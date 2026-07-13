"""Transfer-evaluation assembler (ADR 0008).

Reads whatever producer cells are present under ``cache/transfer/`` (curve,
reference, and equation if it exists) and builds the descriptive source × target
matrix. The matrix is **read off, not selected**: every source policy and every
seed is kept, and each cell reports the spread across its transferred seeds as the
regime's generalization consistency. Mirrors the ``compile_results_fetch`` /
``compile_results_plot`` split — this is the plot side.
"""

import dataclasses
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro

from util.transfer import assemble_transfer

# A source×target cell is keyed by the source policy and the target regime.
_CELL_KEYS = ["producer", "source_id", "target", "target_eps", "target_T"]

# The cell identity shared across producers (producer itself excluded): a curve cell
# and an equation cell overlay iff they agree on these.
_OVERLAY_KEYS = ["source_id", "target", "target_eps", "target_T"]


def transfer_matrix(assembled: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-seed transfer rows to one row per source×target cell.

    Read off, not selected: every ``source_id`` is kept and each cell reports the
    mean accuracy plus the spread (std) across its transferred seeds — the
    generalization-consistency measure, never a selected best seed.
    """
    return (
        assembled.groupby(_CELL_KEYS, dropna=False)["accuracy"]
        .agg(mean_acc="mean", spread=lambda s: float(s.std(ddof=0)), n="count")
        .reset_index()
    )


def nearest_source(assembled: pd.DataFrame, target_eps: float, target_T: int) -> str:
    """The ``source_id`` whose source regime is nearest the target in (ε, T).

    Distance is *relative* in each axis — ``|Δε|/target_eps`` and ``|ΔT|/target_T``
    combined in quadrature — so ε and T contribute comparably despite their very
    different scales. Ties break deterministically on ``source_id`` (sorted first).
    """
    sources = (
        assembled[["source_id", "source_eps", "source_T"]]
        .drop_duplicates()
        .sort_values("source_id")
    )
    d_eps = (sources["source_eps"] - target_eps) / target_eps
    d_T = (sources["source_T"] - target_T) / target_T
    dist = (d_eps**2 + d_T**2) ** 0.5
    return str(sources.iloc[dist.to_numpy().argmin()]["source_id"])


def _cell_keys(df: pd.DataFrame) -> set[tuple]:
    """The distinct overlay-cell keys present in one producer's frame."""
    distinct = df[_OVERLAY_KEYS].drop_duplicates()
    return {tuple(row) for row in distinct.itertuples(index=False, name=None)}


def overlay_cells(producers: dict[str, pd.DataFrame]) -> list[tuple]:
    """Cells for which BOTH the curve and equation producers have a record.

    A presence-check join (ADR 0008): the curve-vs-equation overlay is drawn only
    where both producers evaluated the same source×target cell. Returns the sorted
    intersection of their cell keys; empty if either producer is absent. Reference
    cells never participate.
    """
    curve = producers.get("curve")
    equation = producers.get("equation")
    if curve is None or equation is None:
        return []
    return sorted(_cell_keys(curve) & _cell_keys(equation))


# ---------------------------------------------------------------------------
# Rendering + IO (integration glue; exercised end-to-end, not unit-tested)
# ---------------------------------------------------------------------------


def _target_label(target: str, eps: float, T: int) -> str:
    """Column label for one target regime."""
    return f"{target}\nε={eps:g} T={int(T)}"


def load_producers(cache_root: Path | str) -> dict[str, pd.DataFrame]:
    """Assemble every producer that has written cells under ``cache/transfer/``.

    Discovers the producer subdirectories (``curve``, ``reference``, ``equation``,
    or any future writer of the shared schema) and assembles each into one frame.
    Whatever is present is returned — the assembler renders what it finds.
    """
    root = Path(cache_root) / "transfer"
    producers: dict[str, pd.DataFrame] = {}
    if not root.is_dir():
        return producers
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if not any(sub.glob("*.parquet")):
            continue
        producers[sub.name] = assemble_transfer(sub.name, cache_root)
    return producers


def plot_matrix(assembled: pd.DataFrame, producer: str, out_stem: Path) -> None:
    """Descriptive source × target accuracy matrix for one producer.

    Rows are source policies, columns are target regimes; each cell shows mean
    transfer accuracy with the generalization-consistency spread beneath it. Every
    column is annotated (in its title) with the source nearest it in (ε, T) — read
    off, not selected.
    """
    if assembled.empty:
        print(f"  [skip] no cells for producer={producer}")
        return

    matrix = transfer_matrix(assembled)
    matrix = matrix[matrix["producer"] == producer]
    if matrix.empty:
        print(f"  [skip] no cells for producer={producer}")
        return

    matrix = matrix.assign(
        _col=[
            _target_label(t, e, tt)
            for t, e, tt in zip(matrix["target"], matrix["target_eps"], matrix["target_T"])
        ]
    )
    sources = sorted(matrix["source_id"].unique())
    columns = sorted(matrix["_col"].unique())
    row_of = {s: i for i, s in enumerate(sources)}
    col_of = {c: j for j, c in enumerate(columns)}

    grid = np.full((len(sources), len(columns)), np.nan)
    text = np.full((len(sources), len(columns)), "", dtype=object)
    for _, r in matrix.iterrows():
        i, j = row_of[r["source_id"]], col_of[r["_col"]]
        grid[i, j] = r["mean_acc"]
        text[i, j] = f"{r['mean_acc']:.3f}\n±{r['spread']:.3f}"

    fig, ax = plt.subplots(figsize=(1.7 * len(columns) + 2.5, 0.55 * len(sources) + 2.0))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="mean transfer accuracy", fraction=0.046, pad=0.04)

    # Annotate each target column with the source nearest it in (ε, T).
    col_titles = []
    for c in columns:
        cell = matrix[matrix["_col"] == c].iloc[0]
        near = nearest_source(assembled, float(cell["target_eps"]), int(cell["target_T"]))
        col_titles.append(f"{c}\n[near: {near}]")

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(col_titles, fontsize=7, rotation=30, ha="right")
    ax.set_yticks(range(len(sources)))
    ax.set_yticklabels(sources, fontsize=7)
    ax.set_ylabel("source policy")
    for i in range(len(sources)):
        for j in range(len(columns)):
            if text[i, j]:
                ax.text(j, i, text[i, j], ha="center", va="center", fontsize=6, color="white")
    ax.set_title(f"Transfer matrix — {producer} (mean ± seed spread)", fontsize=10)
    fig.tight_layout()
    _save(fig, out_stem)


def plot_overlay(producers: dict[str, pd.DataFrame], out_stem: Path) -> None:
    """Per-cell curve-vs-equation accuracy comparison, only where both exist.

    Draws one grouped point per shared source×target cell: curve-transfer mean±spread
    beside equation-transfer mean±spread. Skips entirely when the two producers share
    no cell (the compare-only-when-both-exist rule).
    """
    cells = overlay_cells(producers)
    if not cells:
        print("  [skip] no cells present in both curve and equation")
        return

    curve = transfer_matrix(producers["curve"])
    equation = transfer_matrix(producers["equation"])

    def _lookup(m: pd.DataFrame, cell: tuple) -> tuple[float, float]:
        src, tgt, eps, T = cell
        row = m[
            (m["source_id"] == src)
            & (m["target"] == tgt)
            & (m["target_eps"] == eps)
            & (m["target_T"] == T)
        ].iloc[0]
        return float(row["mean_acc"]), float(row["spread"])

    labels = [f"{s}\n{_target_label(t, e, tt)}" for (s, t, e, tt) in cells]
    x = np.arange(len(cells))
    curve_mu, curve_sd = zip(*(_lookup(curve, c) for c in cells))
    eqn_mu, eqn_sd = zip(*(_lookup(equation, c) for c in cells))

    fig, ax = plt.subplots(figsize=(1.4 * len(cells) + 2.5, 4.0))
    ax.errorbar(
        x - 0.1, curve_mu, yerr=curve_sd, fmt="o", capsize=3, label="curve", color="#1f77b4"
    )
    ax.errorbar(x + 0.1, eqn_mu, yerr=eqn_sd, fmt="s", capsize=3, label="equation", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_xlim(-0.5, len(cells) - 0.5)
    ax.set_ylabel("transfer accuracy")
    ax.set_title("Curve vs equation transfer (shared cells)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, out_stem)


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  ✓ {stem.with_suffix('.pdf').name}")


@dataclasses.dataclass
class PlotConfig:
    """Assemble and plot the transfer cells under ``<cache_root>/transfer/``."""

    cache_root: str = "cache"
    out_dir: str = ""
    """Output dir. Defaults to <cache_root>/transfer/plots/."""


def main(conf: PlotConfig) -> None:
    producers = load_producers(conf.cache_root)
    if not producers:
        raise SystemExit(f"no transfer cells under {Path(conf.cache_root) / 'transfer'}")

    out_root = Path(conf.out_dir) if conf.out_dir else Path(conf.cache_root) / "transfer" / "plots"
    out_root.mkdir(parents=True, exist_ok=True)

    for producer, assembled in producers.items():
        print(f"\n=== producer: {producer} ===")
        plot_matrix(assembled, producer, out_root / f"matrix_{producer}")
        transfer_matrix(assembled).to_csv(out_root / f"matrix_{producer}.csv", index=False)

    print("\n=== overlay ===")
    plot_overlay(producers, out_root / "curve_vs_equation")


if __name__ == "__main__":
    main(tyro.cli(PlotConfig))
