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

# A matrix cell is keyed by the source REGIME-ARM and the target regime (ADR 0018) —
# not by source_id. The regime is the row unit of analysis, so pooling its policies is
# what makes the printed ± generalization consistency rather than evaluation noise.
_CELL_KEYS = [
    "producer",
    "source_dataset",
    "source_eps",
    "source_T",
    "source_arch",
    "source_arm",
    "source_label",
    "target",
    "target_eps",
    "target_T",
]

# The per-policy view: the same cell split by source policy. Its spread is across one
# policy's evaluation reps, which is a different quantity (CONTEXT.md).
_POLICY_KEYS = ["producer", "source_id", "target", "target_eps", "target_T"]

# The cell identity shared across producers: a curve cell and an equation cell
# overlay iff they agree on the source REGIME and the target (ADR 0015). NOT
# source_id — curve's is a W&B run id and equation's is a condition slug, so they can
# never compare equal; and a distilled condition has no per-seed identity to match
# on anyway. target_arch is excluded: ADR 0007 derives it from the target dataset, so
# it adds nothing but a chance of spurious label mismatch. source_arm IS included
# (ADR 0018): a synthesis is scoped to one arm (ADR 0016), so the other arm's curve
# cells have no equation counterpart and must simply not overlay — without it they
# would be compared against a closed form distilled from different shapes entirely.
_OVERLAY_KEYS = [
    "source_dataset",
    "source_eps",
    "source_T",
    "source_arch",
    "source_arm",
    "target",
    "target_eps",
    "target_T",
]


def _collapse(assembled: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean / spread / count of ``accuracy`` over ``keys``."""
    return (
        assembled.groupby(keys, dropna=False)["accuracy"]
        .agg(mean_acc="mean", spread=lambda s: float(s.std(ddof=0)), n="count")
        .reset_index()
    )


def source_labels(assembled: pd.DataFrame) -> pd.Series:
    """The row identity each transfer row pools into.

    Curve rows pool by source **regime-arm**: the regime's seed-policies are what a
    cell averages, so its label names the regime, not one of the run ids inside it.
    Every other producer's ``source_id`` already *is* its row unit — a distilled
    condition has no seeds (ADR 0015), and a native reference is a mechanism rather
    than a regime (CONTEXT.md), so the three references must not merge just because
    their source provenance all mirrors the same target.
    """
    regime = (
        assembled["source_dataset"].astype(str)
        + " ε="
        + assembled["source_eps"].map(lambda e: f"{float(e):g}")
        + " T="
        + assembled["source_T"].map(lambda t: str(int(t)))
        + " "
        + assembled["source_arch"].astype(str)
        + " "
        + assembled["source_arm"].astype(str)
    ).str.strip()
    return regime.where(assembled["producer"] == "curve", assembled["source_id"].astype(str))


def transfer_matrix(assembled: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-seed transfer rows to one row per source regime-arm × target.

    Read off, not selected: every regime-arm is kept, and its cell reports the mean
    accuracy plus the spread over **all** of its rows — every policy in the regime
    crossed with every evaluation rep. That spread is the regime's *generalization
    consistency* (CONTEXT.md).

    Grouping on ``source_id`` instead — as this did before ADR 0018 — makes the ±
    the spread across one policy's reps, i.e. DP-SGD's own run-to-run noise, which
    is a different and much smaller quantity than the one the matrix claims to
    report. That view is still available as :func:`policy_matrix`.
    """
    return _collapse(assembled.assign(source_label=source_labels(assembled)), _CELL_KEYS)


def policy_matrix(assembled: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-seed transfer rows to one row per source *policy* × target.

    The diagnostic companion to :func:`transfer_matrix`: same cells, split by source
    policy, so each row's spread is that policy's **evaluation noise** rather than
    its regime's generalization consistency. Kept distinct because CONTEXT.md
    requires every ± to name which of the two it is.
    """
    return _collapse(assembled, _POLICY_KEYS)


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

    A presence-check join (ADR 0008), keyed on the source *regime* rather than the
    source policy (ADR 0015): the overlay is drawn only where both producers
    evaluated the same source regime × target. Returns the sorted intersection of
    their cell keys; empty if either producer is absent. Reference cells never
    participate.
    """
    curve = producers.get("curve")
    equation = producers.get("equation")
    if curve is None or equation is None:
        return []
    return sorted(_cell_keys(curve) & _cell_keys(equation))


def overlay_stats(assembled: pd.DataFrame, cell: tuple) -> tuple[float, float]:
    """One producer's mean and spread for an overlay cell, pooled over the regime.

    The overlay joins on the source regime (ADR 0015), so the curve side — whose
    row unit is one seed's policy — is pooled across *every* policy in the regime
    before it is compared with the equation side's single distilled condition.
    Consequence: the curve error bar mixes seed noise with across-policy spread
    while the equation's is seed noise alone, so the two bars are not like-for-like
    and must not be read as a significance test.
    """
    mask = pd.Series(True, index=assembled.index)
    for key, value in zip(_OVERLAY_KEYS, cell):
        mask &= assembled[key] == value
    accuracy = assembled.loc[mask, "accuracy"]
    return float(accuracy.mean()), float(accuracy.std(ddof=0))


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


def plot_matrix(
    assembled: pd.DataFrame,
    producer: str,
    out_stem: Path,
    by_policy: bool = False,
) -> None:
    """Descriptive source × target accuracy matrix for one producer.

    Rows are source regime-arms, columns are target regimes; each cell shows mean
    transfer accuracy with the **generalization-consistency** spread beneath it —
    the spread across the regime's source policies (ADR 0018). Every column is
    annotated (in its title) with the source nearest it in (ε, T) — read off, not
    selected.

    ``by_policy`` draws the diagnostic companion instead: one row per source policy,
    whose ± is that policy's **evaluation noise**. The two are different quantities
    and the title says which is being shown.
    """
    if assembled.empty:
        print(f"  [skip] no cells for producer={producer}")
        return

    if by_policy:
        matrix = policy_matrix(assembled).rename(columns={"source_id": "source_label"})
        spread_name = "policy evaluation noise"
    else:
        matrix = transfer_matrix(assembled)
        spread_name = "regime generalization consistency"
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
    sources = sorted(matrix["source_label"].unique())
    columns = sorted(matrix["_col"].unique())
    row_of = {s: i for i, s in enumerate(sources)}
    col_of = {c: j for j, c in enumerate(columns)}

    grid = np.full((len(sources), len(columns)), np.nan)
    text = np.full((len(sources), len(columns)), "", dtype=object)
    for _, r in matrix.iterrows():
        i, j = row_of[r["source_label"]], col_of[r["_col"]]
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
    ax.set_ylabel("source policy" if by_policy else "source regime-arm")
    for i in range(len(sources)):
        for j in range(len(columns)):
            if text[i, j]:
                ax.text(j, i, text[i, j], ha="center", va="center", fontsize=6, color="white")
    ax.set_title(f"Transfer matrix — {producer} (mean ± {spread_name})", fontsize=10)
    fig.tight_layout()
    _save(fig, out_stem)


def plot_overlay(producers: dict[str, pd.DataFrame], out_stem: Path) -> None:
    """Per-regime curve-vs-equation accuracy comparison, only where both exist.

    Draws one grouped point per shared source-regime × target cell (ADR 0015):
    curve-transfer mean±spread beside equation-transfer mean±spread. Skips entirely
    when the two producers share no regime (the compare-only-when-both-exist rule).
    The curve side is pooled across the regime's seed-policies, so its error bar is
    wider by construction than the equation's — see ``overlay_stats``.
    """
    cells = overlay_cells(producers)
    if not cells:
        print("  [skip] no source regimes present in both curve and equation")
        return

    labels = [
        f"{s_ds} ε={s_eps:g} T={int(s_T)} {s_arch} {s_arm}\n→ {_target_label(t, e, tt)}"
        for (s_ds, s_eps, s_T, s_arch, s_arm, t, e, tt) in cells
    ]
    x = np.arange(len(cells))
    curve_mu, curve_sd = zip(*(overlay_stats(producers["curve"], c) for c in cells))
    eqn_mu, eqn_sd = zip(*(overlay_stats(producers["equation"], c) for c in cells))

    fig, ax = plt.subplots(figsize=(1.4 * len(cells) + 2.5, 4.0))
    ax.errorbar(
        x - 0.1, curve_mu, yerr=curve_sd, fmt="o", capsize=3, label="curve", color="#1f77b4"
    )
    ax.errorbar(x + 0.1, eqn_mu, yerr=eqn_sd, fmt="s", capsize=3, label="equation", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_xlim(-0.5, len(cells) - 0.5)
    ax.set_ylabel("transfer accuracy")
    ax.set_title("Curve vs equation transfer (shared source regimes)", fontsize=10)
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
        # The per-policy companion: same cells, but its ± is evaluation noise.
        plot_matrix(assembled, producer, out_root / f"matrix_{producer}_by_policy", by_policy=True)
        policy_matrix(assembled).to_csv(out_root / f"matrix_{producer}_by_policy.csv", index=False)

    print("\n=== overlay ===")
    plot_overlay(producers, out_root / "curve_vs_equation")


if __name__ == "__main__":
    main(tyro.cli(PlotConfig))
