#!/usr/bin/env python3
"""compile_results_plot.py — Read parquets produced by compile_results_fetch.py
and emit per-optimizer figures + tables.

Per optimizer the script writes:

    out/<optimizer>/
        t_sweep_main.{pdf,png}
        t_sweep_delta_vs_constant.{pdf,png}
        t_sweep_delta_vs_dynamic.{pdf,png}
        arch_sweep_main.{pdf,png}
        arch_sweep_delta_vs_constant.{pdf,png}
        arch_sweep_delta_vs_dynamic.{pdf,png}
        sigma_shape.{pdf,png}
        clip_shape.{pdf,png}
        t_sweep_table.{csv,tex}
        arch_sweep_table.{csv,tex}

Usage (from src/):
    uv run compile_results_plot.py --in-dir cache/results/<entity>__<project>
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro
from matplotlib.lines import Line2D

LEARNED = "Learned Schedule"
CONSTANT = "Constant σ/clip"  # noqa: RUF001
MEDIAN = "Clip to Median Gradient Norm"
DYNAMIC = "Dynamic-DPSGD"

SCHEDULE_ORDER: list[str] = [LEARNED, DYNAMIC, MEDIAN, CONSTANT]
SCHEDULE_SHORT: dict[str, str] = {
    LEARNED: "Learned",
    DYNAMIC: "Dynamic-DPSGD",
    MEDIAN: "Median-Clip",
    CONSTANT: "Constant",
}
SCHEDULE_COLORS: dict[str, str] = {
    LEARNED: "#1f77b4",
    DYNAMIC: "#d62728",
    MEDIAN: "#2ca02c",
    CONSTANT: "#7f7f7f",
}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _stderr(s: pd.Series) -> float:
    n = len(s)
    if n < 2:
        return 0.0
    return float(s.std(ddof=1) / math.sqrt(n))


def aggregate_across_seeds(
    df: pd.DataFrame,
    x_col: str,
    metric: str = "mean_acc",
) -> pd.DataFrame:
    """Collapse seeds to mean ± stderr + min/max per (dataset, eps, schedule, x)."""
    return (
        df.groupby(["dataset", "eps", "schedule", x_col], dropna=False)[metric]
        .agg(mean="mean", stderr=_stderr, lo="min", hi="max", n="count")
        .reset_index()
    )


def paired_delta(
    df: pd.DataFrame,
    baseline: str,
    x_col: str,
    metric: str = "mean_acc",
) -> pd.DataFrame:
    """Per-seed Δ = Learned − baseline, then aggregate across seeds."""
    keys = ["dataset", "eps", x_col, "seed"]
    learned = df[df["schedule"] == LEARNED][*keys, metric].rename(columns={metric: "learned"})
    base = df[df["schedule"] == baseline][*keys, metric].rename(columns={metric: "base"})
    merged = learned.merge(base, on=keys, how="inner")
    merged["delta"] = merged["learned"] - merged["base"]
    return (
        merged.groupby(["dataset", "eps", x_col], dropna=False)["delta"]
        .agg(mean="mean", stderr=_stderr, lo="min", hi="max", n="count")
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Faceting helpers
# ---------------------------------------------------------------------------


def _facet_grid(datasets: list[str], epsilons: list[float]) -> tuple[plt.Figure, np.ndarray]:
    nrows, ncols = len(datasets), len(epsilons)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.0 * ncols + 1, 2.4 * nrows + 0.5),
        squeeze=False,
        sharex=True,
        sharey="row",
    )
    return fig, axes


def _label_facets(
    axes: np.ndarray, datasets: list[str], epsilons: list[float], xlabel: str, ylabel: str
) -> None:
    for j, eps in enumerate(epsilons):
        axes[0, j].set_title(f"ε = {eps:g}")
    for i, ds in enumerate(datasets):
        axes[i, 0].set_ylabel(f"{ds}\n{ylabel}")
    for j in range(axes.shape[1]):
        axes[-1, j].set_xlabel(xlabel)


def _legend_handles(schedules: list[str]) -> list[Line2D]:
    return [
        Line2D([], [], color=SCHEDULE_COLORS[s], marker="o", label=SCHEDULE_SHORT[s])
        for s in schedules
    ]


# ---------------------------------------------------------------------------
# Main / delta plots
# ---------------------------------------------------------------------------


def _x_axis_for_arch(scalars_df: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    """Return arch labels sorted ascending by param count, and label→x-pos map."""
    pairs = (
        scalars_df.dropna(subset=["arch_param_count"])
        .groupby("arch_label")["arch_param_count"]
        .first()
        .sort_values()
    )
    labels = list(pairs.index)
    return labels, {lbl: i for i, lbl in enumerate(labels)}


def plot_main(
    scalars_df: pd.DataFrame,
    axis: str,
    metric: str,
    out_path_stem: Path,
) -> None:
    df = scalars_df[scalars_df["axis"] == axis].copy()
    if df.empty:
        print(f"  [skip] no rows for axis={axis}")
        return

    datasets = sorted(df["dataset"].unique())
    epsilons = sorted(df["eps"].dropna().unique())

    if axis == "T-sweep":
        x_col = "T"
        x_vals = sorted(df[x_col].dropna().unique())
        x_to_pos = {v: float(v) for v in x_vals}
        xlabel = "T (inner-loop steps)"
    else:
        x_col = "arch_label"
        x_vals, x_to_pos = _x_axis_for_arch(df)
        x_to_pos = {lbl: float(i) for i, lbl in enumerate(x_vals)}
        xlabel = "architecture (sorted by param count)"

    agg = aggregate_across_seeds(df, x_col, metric)
    fig, axes = _facet_grid(datasets, epsilons)
    ylabel = "val accuracy" if metric == "mean_acc" else "val loss"

    for i, ds in enumerate(datasets):
        for j, eps in enumerate(epsilons):
            ax = axes[i, j]
            cell = agg[(agg["dataset"] == ds) & (agg["eps"] == eps)]
            for sched in SCHEDULE_ORDER:
                rows = cell[cell["schedule"] == sched].copy()
                if rows.empty:
                    continue
                rows["xpos"] = rows[x_col].map(x_to_pos)
                rows = rows.sort_values("xpos")
                ax.plot(
                    rows["xpos"],
                    rows["mean"],
                    color=SCHEDULE_COLORS[sched],
                    marker="o",
                    markersize=4,
                    linewidth=1.2,
                    label=SCHEDULE_SHORT[sched],
                )
                ax.fill_between(
                    rows["xpos"],
                    rows["lo"],
                    rows["hi"],
                    color=SCHEDULE_COLORS[sched],
                    alpha=0.12,
                    linewidth=0,
                )
            if axis == "arch-sweep":
                ax.set_xticks([x_to_pos[v] for v in x_vals])
                ax.set_xticklabels(x_vals, rotation=40, ha="right", fontsize=7)
            ax.grid(True, alpha=0.3, linewidth=0.5)

    _label_facets(axes, datasets, epsilons, xlabel, ylabel)
    fig.legend(
        handles=_legend_handles(SCHEDULE_ORDER),
        loc="lower center",
        ncol=len(SCHEDULE_ORDER),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, out_path_stem)


def plot_delta(
    scalars_df: pd.DataFrame,
    axis: str,
    baseline: str,
    metric: str,
    out_path_stem: Path,
) -> None:
    df = scalars_df[scalars_df["axis"] == axis].copy()
    if df.empty:
        print(f"  [skip] no rows for axis={axis}")
        return

    datasets = sorted(df["dataset"].unique())
    epsilons = sorted(df["eps"].dropna().unique())

    if axis == "T-sweep":
        x_col = "T"
        x_vals = sorted(df[x_col].dropna().unique())
        x_to_pos = {v: float(v) for v in x_vals}
        xlabel = "T (inner-loop steps)"
    else:
        x_col = "arch_label"
        x_vals, _ = _x_axis_for_arch(df)
        x_to_pos = {lbl: float(i) for i, lbl in enumerate(x_vals)}
        xlabel = "architecture (sorted by param count)"

    delta = paired_delta(df, baseline, x_col, metric)
    if delta.empty:
        print(f"  [skip] no paired data Learned vs {baseline}")
        return

    fig, axes = _facet_grid(datasets, epsilons)
    ylabel = (
        f"Learned - {SCHEDULE_SHORT[baseline]} (Δ acc)"
        if metric == "mean_acc"
        else f"Learned - {SCHEDULE_SHORT[baseline]} (Δ loss)"
    )

    for i, ds in enumerate(datasets):
        for j, eps in enumerate(epsilons):
            ax = axes[i, j]
            cell = delta[(delta["dataset"] == ds) & (delta["eps"] == eps)].copy()
            cell["xpos"] = cell[x_col].map(x_to_pos)
            cell = cell.sort_values("xpos")
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
            if not cell.empty:
                ax.plot(
                    cell["xpos"],
                    cell["mean"],
                    color="#1f77b4",
                    marker="o",
                    markersize=4,
                    linewidth=1.2,
                )
                ax.fill_between(
                    cell["xpos"],
                    cell["lo"],
                    cell["hi"],
                    color="#1f77b4",
                    alpha=0.15,
                    linewidth=0,
                )
            if axis == "arch-sweep":
                ax.set_xticks([x_to_pos[v] for v in x_vals])
                ax.set_xticklabels(x_vals, rotation=40, ha="right", fontsize=7)
            ax.grid(True, alpha=0.3, linewidth=0.5)

    _label_facets(axes, datasets, epsilons, xlabel, ylabel)
    fig.tight_layout()
    _save(fig, out_path_stem)


# ---------------------------------------------------------------------------
# Schedule-shape plots
# ---------------------------------------------------------------------------


def plot_shape(
    schedules_df: pd.DataFrame,
    var: str,  # "sigma" or "clip"
    out_path_stem: Path,
) -> None:
    if schedules_df.empty:
        print(f"  [skip] no schedule rows for {var}")
        return

    datasets = sorted(schedules_df["dataset"].unique())
    epsilons = sorted(schedules_df["eps"].dropna().unique())
    cmap = plt.get_cmap("viridis")
    eps_to_color = {eps: cmap(i / max(1, len(epsilons) - 1)) for i, eps in enumerate(epsilons)}

    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(4.5 * len(datasets), 3.2),
        squeeze=False,
        sharey=True,
    )
    axes_row = axes[0]

    for j, ds in enumerate(datasets):
        ax = axes_row[j]
        sub = schedules_df[schedules_df["dataset"] == ds]
        for (eps, _run_id), grp in sub.groupby(["eps", "run_id"]):
            grp = grp.sort_values("inner_step")
            ax.plot(
                grp["step_norm"],
                grp[var],
                color=eps_to_color.get(eps, "gray"),
                alpha=0.35,
                linewidth=0.7,
            )
        ax.set_title(ds)
        ax.set_xlabel("t / T")
        ax.grid(True, alpha=0.3, linewidth=0.5)
    axes_row[0].set_ylabel(var)

    legend_handles = [Line2D([], [], color=eps_to_color[e], label=f"ε = {e:g}") for e in epsilons]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(epsilons),
        bbox_to_anchor=(0.5, -0.05),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, out_path_stem)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _fmt_cell(mean: float | None, stderr: float | None) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "—"
    if stderr is None or (isinstance(stderr, float) and math.isnan(stderr)) or stderr == 0:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {stderr:.3f}"


def build_table(
    scalars_df: pd.DataFrame,
    axis: str,
) -> pd.DataFrame:
    df = scalars_df[scalars_df["axis"] == axis].copy()
    if df.empty:
        return pd.DataFrame()

    x_col = "T" if axis == "T-sweep" else "arch_label"

    agg = aggregate_across_seeds(df, x_col, "mean_acc")
    pivot_mean = agg.pivot_table(
        index=["dataset", "eps", x_col],
        columns="schedule",
        values="mean",
        aggfunc="first",
    )
    pivot_se = agg.pivot_table(
        index=["dataset", "eps", x_col],
        columns="schedule",
        values="stderr",
        aggfunc="first",
    )

    cols = [s for s in SCHEDULE_ORDER if s in pivot_mean.columns]
    pivot_mean = pivot_mean[cols]
    pivot_se = pivot_se.reindex(columns=cols)

    if axis == "arch-sweep":
        param_map = (
            df.dropna(subset=["arch_param_count"]).groupby("arch_label")["arch_param_count"].first()
        )
        idx = pivot_mean.reset_index()
        idx["arch_param_count"] = idx["arch_label"].map(param_map)
        idx = idx.sort_values(["dataset", "eps", "arch_param_count"])
        pivot_mean = pivot_mean.loc[list(zip(idx["dataset"], idx["eps"], idx["arch_label"]))]
        pivot_se = pivot_se.reindex(pivot_mean.index)
    else:
        pivot_mean = pivot_mean.sort_index()
        pivot_se = pivot_se.reindex(pivot_mean.index)

    return _format_table_with_winners(pivot_mean, pivot_se)


def _format_table_with_winners(pivot_mean: pd.DataFrame, pivot_se: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=pivot_mean.index, columns=pivot_mean.columns, dtype=object)
    for idx, row in pivot_mean.iterrows():
        winner = row.idxmax(skipna=True) if not row.isna().all() else None
        for col in pivot_mean.columns:
            cell = _fmt_cell(row[col], pivot_se.loc[idx, col])
            if col == winner and cell != "—":
                cell = f"**{cell}**"
            out.loc[idx, col] = cell
    out.columns = [SCHEDULE_SHORT[c] for c in out.columns]
    return out


def write_tables(table: pd.DataFrame, stem: Path) -> None:
    if table.empty:
        return
    table.to_csv(stem.with_suffix(".csv"))
    _write_latex(table, stem.with_suffix(".tex"))


def _write_latex(table: pd.DataFrame, path: Path) -> None:
    cols = list(table.columns)
    idx_names = list(table.index.names)
    n_idx = len(idx_names)
    col_spec = "l" * n_idx + "c" * len(cols)

    lines: list[str] = []
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\toprule")
    header = " & ".join(idx_names + cols) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    for idx, row in table.iterrows():
        idx_tuple = idx if isinstance(idx, tuple) else (idx,)
        idx_cells = [_latex_escape(str(v)) for v in idx_tuple]
        body_cells = [_latex_cell(v) for v in row.tolist()]
        lines.append(" & ".join(idx_cells + body_cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n")


def _latex_cell(s: str) -> str:
    if s.startswith("**") and s.endswith("**"):
        inner = s[2:-2].replace("±", "$\\pm$")
        return f"\\textbf{{{inner}}}"
    if s == "—":
        return "---"
    return s.replace("±", "$\\pm$")


def _latex_escape(s: str) -> str:
    return s.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  ✓ {stem.with_suffix('.pdf').name}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class PlotConfig:
    in_dir: str
    """Cache dir written by compile_results_fetch.py."""
    out_dir: str = ""
    """Output dir. Defaults to <in_dir>/plots/."""
    optimizers: tuple[str, ...] = ()
    """Restrict to these optimizers. Empty = all present."""


def main(conf: PlotConfig) -> None:
    in_dir = Path(conf.in_dir)
    scalars = pd.read_parquet(in_dir / "scalars.parquet")
    schedules = pd.read_parquet(in_dir / "schedules.parquet")
    if scalars.empty:
        raise SystemExit(f"scalars.parquet at {in_dir} is empty")

    out_root = Path(conf.out_dir) if conf.out_dir else in_dir / "plots"
    optimizers = list(conf.optimizers) if conf.optimizers else sorted(scalars["optimizer"].unique())

    for opt in optimizers:
        print(f"\n=== optimizer: {opt} ===")
        s = scalars[scalars["optimizer"] == opt].copy()
        sch = schedules[schedules["optimizer"] == opt].copy()
        if s.empty:
            print(f"  [skip] no rows for optimizer={opt}")
            continue
        out = out_root / opt
        out.mkdir(parents=True, exist_ok=True)

        for axis_tag, prefix in (("T-sweep", "t_sweep"), ("arch-sweep", "arch_sweep")):
            plot_main(s, axis_tag, "mean_acc", out / f"{prefix}_main")
            plot_delta(s, axis_tag, CONSTANT, "mean_acc", out / f"{prefix}_delta_vs_constant")
            plot_delta(s, axis_tag, DYNAMIC, "mean_acc", out / f"{prefix}_delta_vs_dynamic")
            tbl = build_table(s, axis_tag)
            if not tbl.empty:
                write_tables(tbl, out / f"{prefix}_table")
                print(f"  ✓ {prefix}_table.csv / .tex")

        plot_shape(sch, "sigma", out / "sigma_shape")
        plot_shape(sch, "clip", out / "clip_shape")


if __name__ == "__main__":
    main(tyro.cli(PlotConfig))
