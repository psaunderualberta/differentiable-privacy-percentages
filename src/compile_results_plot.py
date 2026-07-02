#!/usr/bin/env python3
"""compile_results_plot.py — Read parquets produced by compile_results_fetch.py
and emit per-optimizer figures + tables.

Per optimizer the script writes:

    out/<optimizer>/
        t_sweep_main.{pdf,png}
        t_sweep_delta_vs_constant.{pdf,png}
        t_sweep_delta_vs_dynamic.{pdf,png}
        t_sweep_table.{csv,tex}
        sigma_shape.{pdf,png}
        clip_shape.{pdf,png}
        curves/
            t_sweep_loss__<dataset>.{pdf,png}
            t_sweep_acc__<dataset>.{pdf,png}
        shape_variants/
            <var>_shape__T_sweep__by_T.{pdf,png}
            <var>_shape__T_sweep__by_seed.{pdf,png}
            (var ∈ {sigma, clip})
        ladders/
            overall/
                arch_forest_delta.{pdf,png}
                arch_forest_abs.{pdf,png}
            <ladder-name>/
                main.{pdf,png}
                delta_vs_constant.{pdf,png}
                delta_vs_dynamic.{pdf,png}
                table.{csv,tex}
                sigma_shape_by_rung.{pdf,png}
                clip_shape_by_rung.{pdf,png}

The architecture sweep is no longer plotted as a single lumped param-count axis.
Instead each ladder (``experiments.architectures.LADDERS``) gets its own
subdirectory with its rungs on a categorical x-axis, and ``ladders/overall/``
holds cross-ladder **forest plots** (rungs on a categorical y-axis grouped into
per-ladder blocks; datasets as columns) answering the architecture-robustness
question rather than a scaling question.

Usage (from src/):
    uv run compile_results_plot.py --in-dir cache/results/<entity>__<project>
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from scipy.stats import t as t_dist  # aliased: many local `t` vars in this file

# Ladder definitions are the single source of truth for rung ordering and the
# arch_label of each rung. This couples the plot script to the training code —
# unlike compile_results_fetch.py, which deliberately stays independent so it can
# run against any project. The plot script has no such constraint and importing
# LADDERS avoids re-deriving rung order by fragile string parsing. The import is
# cheap (pure dataclasses, no jax).
from experiments.architectures import LADDERS
from networks.cnn.config import CNNConfig
from networks.mlp.config import MLPConfig

LEARNED = "Learned Schedule"
CONSTANT = "Constant σ/clip"
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
# Axis selection
# ---------------------------------------------------------------------------

# compile_results_fetch._axis() writes "arch" for the architecture sweep, but
# accepts the legacy "arch-sweep" tag for back-compat with pre-ladder projects.
# Mirror that tolerance here so plots render against either vintage of cache.
_ARCH_AXES: tuple[str, ...] = ("arch", "arch-sweep")


def _axis_mask(df: pd.DataFrame, axis: str) -> pd.Series:
    """Boolean mask selecting rows for ``axis``; "arch" also matches legacy "arch-sweep"."""
    if axis == "arch":
        return df["axis"].isin(_ARCH_AXES)
    return df["axis"] == axis


# ---------------------------------------------------------------------------
# Ladder bookkeeping
# ---------------------------------------------------------------------------


def _arch_label(arch: MLPConfig | CNNConfig) -> str:
    """Mirror create_experiments._arch_label / compile_results_fetch._arch_info."""
    if isinstance(arch, MLPConfig):
        return "mlp-" + "x".join(str(h) for h in arch.hidden_sizes)
    ch = "x".join(str(c) for c in arch.channels)
    head = "x".join(str(h) for h in arch.mlp.hidden_sizes)
    return f"cnn-{ch}-head{head}"


def _ladder_specs() -> dict[str, list[str]]:
    """``{ladder_name: [arch_label per rung, in ladder order]}`` from LADDERS."""
    return {name: [_arch_label(a) for a in archs] for name, archs in LADDERS.items()}


def _ladder_col(ladder_name: str) -> str:
    """The ``in_<ladder>`` membership column compile_results_fetch writes for this ladder."""
    return f"in_{ladder_name.replace('-', '_')}"


def _ladder_member_mask(df: pd.DataFrame, ladder_name: str) -> pd.Series:
    """Rows belonging to ``ladder_name``; all-False if the column is absent."""
    col = _ladder_col(ladder_name)
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    # The column is object dtype with True on members and None/NaN elsewhere;
    # .eq(True) maps both missing forms to False without dtype-downcast warnings.
    return df[col].eq(True)


# ---------------------------------------------------------------------------
# X-axis specification
# ---------------------------------------------------------------------------


@dataclass
class XAxis:
    """How to place and label the x-axis for a scalar plot.

    ``to_pos`` maps each x value to a float position; ``vals`` is the ordered
    list of values present. ``categorical`` requests explicit ticks labelled by
    ``vals`` (used for ladder rungs); numeric axes (T) leave ticks automatic.
    """

    col: str
    vals: list
    to_pos: dict
    label: str
    categorical: bool


def _xaxis_t(df: pd.DataFrame) -> XAxis:
    vals = sorted(df["T"].dropna().unique())
    return XAxis("T", vals, {v: float(v) for v in vals}, "T (inner-loop steps)", False)


def _xaxis_ladder(df: pd.DataFrame, ordered_labels: list[str]) -> XAxis:
    present = set(df["arch_label"].dropna().unique())
    vals = [lbl for lbl in ordered_labels if lbl in present]
    return XAxis(
        "arch_label",
        vals,
        {lbl: float(i) for i, lbl in enumerate(vals)},
        "architecture (ladder rung)",
        True,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _stderr(s: pd.Series) -> float:
    n = len(s)
    if n < 2:
        return 0.0
    return float(s.std(ddof=1) / math.sqrt(n))


def _add_ci(agg: pd.DataFrame) -> pd.DataFrame:
    """Add a 95% CI half-width column: t(0.975, n−1)·SEM, per-row from each cell's n.

    n<2 guard is required: ``t.ppf(·, df=0)`` returns nan and ``nan * stderr(=0)``
    would break fill_between / forest bars. At n<2 → ci=0.0 (point renders, no
    shading; matches ``_stderr``'s n<2 → 0.0 behaviour).
    """
    agg["ci"] = np.where(
        agg["n"] < 2,
        0.0,
        t_dist.ppf(0.975, agg["n"] - 1) * agg["stderr"],
    )
    return agg


def aggregate_across_seeds(
    df: pd.DataFrame,
    x_col: str,
    metric: str = "mean_acc",
) -> pd.DataFrame:
    """Collapse seeds to mean ± stderr + 95% CI + min/max per (dataset, eps, schedule, x)."""
    agg = (
        df.groupby(["dataset", "eps", "schedule", x_col], dropna=False)[metric]
        .agg(mean="mean", stderr=_stderr, lo="min", hi="max", n="count")
        .reset_index()
    )
    return _add_ci(agg)


def paired_delta(
    df: pd.DataFrame,
    baseline: str,
    x_col: str,
    metric: str = "mean_acc",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-seed Δ = Learned − baseline, aggregated across seeds.

    Returns ``(agg, merged)``: ``agg`` is mean/stderr/ci/lo/hi/n per
    (dataset, eps, x_col); ``merged`` is the pre-aggregation per-seed frame
    (one row per seed, with ``seed`` and ``delta`` columns) so callers that need
    the individual seed points (e.g. forest plots) don't have to re-pair.
    """
    keys = ["dataset", "eps", x_col, "seed"]
    learned = df[df["schedule"] == LEARNED][[*keys, metric]].rename(columns={metric: "learned"})
    base = df[df["schedule"] == baseline][[*keys, metric]].rename(columns={metric: "base"})
    merged = learned.merge(base, on=keys, how="inner")
    merged["delta"] = merged["learned"] - merged["base"]
    agg = (
        merged.groupby(["dataset", "eps", x_col], dropna=False)["delta"]
        .agg(mean="mean", stderr=_stderr, lo="min", hi="max", n="count")
        .reset_index()
    )
    return _add_ci(agg), merged


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


def _apply_categorical_ticks(ax: plt.Axes, xaxis: XAxis) -> None:
    if not xaxis.categorical:
        return
    ax.set_xticks([xaxis.to_pos[v] for v in xaxis.vals])
    ax.set_xticklabels(xaxis.vals, rotation=40, ha="right", fontsize=7)


# ---------------------------------------------------------------------------
# Main / delta plots
# ---------------------------------------------------------------------------


def plot_main(
    df: pd.DataFrame,
    xaxis: XAxis,
    metric: str,
    out_path_stem: Path,
) -> None:
    """Per-(dataset, eps) grid of mean metric vs ``xaxis`` for every schedule.

    ``df`` is already filtered to the relevant rows (T-sweep or one ladder).
    """
    if df.empty:
        print(f"  [skip] no rows for {out_path_stem.name}")
        return

    datasets = sorted(df["dataset"].unique())
    epsilons = sorted(df["eps"].dropna().unique())

    agg = aggregate_across_seeds(df, xaxis.col, metric)
    fig, axes = _facet_grid(datasets, epsilons)
    ylabel = "test accuracy" if metric == "mean_acc" else "test loss"

    for i, ds in enumerate(datasets):
        for j, eps in enumerate(epsilons):
            ax = axes[i, j]
            cell = agg[(agg["dataset"] == ds) & (agg["eps"] == eps)]
            for sched in SCHEDULE_ORDER:
                rows = cell[cell["schedule"] == sched].copy()
                if rows.empty:
                    continue
                rows["xpos"] = rows[xaxis.col].map(xaxis.to_pos)
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
                    rows["mean"] - rows["ci"],
                    rows["mean"] + rows["ci"],
                    color=SCHEDULE_COLORS[sched],
                    alpha=0.12,
                    linewidth=0,
                )
            _apply_categorical_ticks(ax, xaxis)
            ax.grid(True, alpha=0.3, linewidth=0.5)

    _label_facets(axes, datasets, epsilons, xaxis.label, ylabel)
    fig.legend(
        handles=_legend_handles(SCHEDULE_ORDER),
        loc="lower center",
        ncol=len(SCHEDULE_ORDER),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.text(0.5, 0.005, "shaded = 95% CI (n=5 seeds)", ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, out_path_stem)


def plot_delta(
    df: pd.DataFrame,
    xaxis: XAxis,
    baseline: str,
    metric: str,
    out_path_stem: Path,
) -> None:
    """Per-(dataset, eps) grid of Learned − baseline vs ``xaxis``."""
    if df.empty:
        print(f"  [skip] no rows for {out_path_stem.name}")
        return

    datasets = sorted(df["dataset"].unique())
    epsilons = sorted(df["eps"].dropna().unique())

    delta, _ = paired_delta(df, baseline, xaxis.col, metric)
    if delta.empty:
        print(f"  [skip] no paired data Learned vs {baseline} for {out_path_stem.name}")
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
            cell["xpos"] = cell[xaxis.col].map(xaxis.to_pos)
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
                    cell["mean"] - cell["ci"],
                    cell["mean"] + cell["ci"],
                    color="#1f77b4",
                    alpha=0.15,
                    linewidth=0,
                )
            _apply_categorical_ticks(ax, xaxis)
            ax.grid(True, alpha=0.3, linewidth=0.5)

    _label_facets(axes, datasets, epsilons, xaxis.label, ylabel)
    fig.text(0.5, 0.005, "shaded = 95% CI (n=5 seeds)", ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, out_path_stem)


# ---------------------------------------------------------------------------
# Cross-ladder forest plots (ladders/overall/)
# ---------------------------------------------------------------------------
#
# The overall view answers an architecture-*robustness* question — does Learned
# beat Constant at *every* architecture — not a scaling question (the T-sweep
# owns scaling). The ladders vary different knobs (width/depth/pm-depth/channels)
# and are not comparable on a shared parameter-count axis (ADR 0002), so the
# param axis is dropped entirely. Instead rungs go on a categorical y-axis,
# grouped into per-ladder blocks; datasets are columns sharing the row order.
# Each rung shows its individual seeds as dots, plus a mean marker and a 95% CI
# bar (mean ± t(0.975, n−1)·SEM) — a forest plot, not a box plot: the bar is an
# inference interval on the mean, and the raw per-seed dots stay visible so the
# tighter CI is read alongside the honest seed spread.


def _param_map(df: pd.DataFrame) -> pd.Series:
    """arch_label → parameter count (first non-null per label)."""
    return df.dropna(subset=["arch_param_count"]).groupby("arch_label")["arch_param_count"].first()


# Vertical layout (data units; one rung occupies ~1.0). The y-axis runs from 0
# at the top downward into negatives, so the first ladder block sits at the top.
_FOREST_HEADER_GAP = 0.9  # header row → first rung
_FOREST_ROW_GAP = 1.0  # rung → rung
_FOREST_BLOCK_GAP = 0.8  # last rung of a block → next header


def _forest_columns(arch_df: pd.DataFrame) -> list[tuple[str, float]]:
    """Ordered (dataset, eps) column keys present in the data.

    Datasets are the primary column axis (the ladder sweep is ε=8 only, so this
    is one column per dataset); including eps keeps the layout correct if a
    multi-ε arch cache is ever passed.
    """
    combos = arch_df[["dataset", "eps"]].dropna().drop_duplicates().sort_values(["dataset", "eps"])
    return [(str(d), float(e)) for d, e in combos.itertuples(index=False, name=None)]


def _forest_blocks(
    arch_df: pd.DataFrame, ladder_specs: dict[str, list[str]]
) -> list[tuple[str, list[str]]]:
    """``[(ladder_name, [present rung labels in ladder order])]`` for non-empty ladders."""
    blocks: list[tuple[str, list[str]]] = []
    for name, ordered_labels in ladder_specs.items():
        member = arch_df[_ladder_member_mask(arch_df, name)]
        present = set(member["arch_label"].dropna().unique())
        labels = [lbl for lbl in ordered_labels if lbl in present]
        if labels:
            blocks.append((name, labels))
    return blocks


def _forest_layout(
    blocks: list[tuple[str, list[str]]],
) -> tuple[dict[tuple[str, str], float], dict[str, float], float]:
    """Assign y positions to each (ladder, rung) and each block header.

    Returns ``(rung_y, header_y, y_bottom)`` where ``y_bottom`` is the lowest y
    used (already past the trailing block gap).
    """
    rung_y: dict[tuple[str, str], float] = {}
    header_y: dict[str, float] = {}
    y = 0.0
    for name, labels in blocks:
        header_y[name] = y
        y -= _FOREST_HEADER_GAP
        for lbl in labels:
            rung_y[(name, lbl)] = y
            y -= _FOREST_ROW_GAP
        y -= _FOREST_BLOCK_GAP
    return rung_y, header_y, y


def _forest_figure(
    ncols: int,
    blocks: list[tuple[str, list[str]]],
    rung_y: dict[tuple[str, str], float],
    header_y: dict[str, float],
    y_bottom: float,
    sharex: bool,
) -> tuple[plt.Figure, np.ndarray]:
    """One tall row of ``ncols`` forest panels with shared rung order.

    Sets the y-axis to the rung labels (leftmost panel only), draws the bold
    per-block sub-headers and faint block separators on every panel.
    """
    n_rungs = len(rung_y)
    height = max(3.5, 0.40 * (n_rungs + len(blocks)) + 1.0)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(4.2 * ncols + 1.0, height),
        squeeze=False,
        sharey=True,
        sharex=sharex,
    )
    row = axes[0]

    yticks = list(rung_y.values())
    yticklabels = [lbl for (_name, lbl) in rung_y]
    top = _FOREST_HEADER_GAP
    for ax in row:
        ax.set_ylim(y_bottom + _FOREST_BLOCK_GAP * 0.5, top)
        ax.set_yticks(yticks)
        ax.grid(True, axis="x", alpha=0.3, linewidth=0.5)
        # Block separators + bold sub-headers on every panel.
        for name, _labels in blocks:
            hy = header_y[name]
            ax.axhline(hy - _FOREST_HEADER_GAP * 0.5, color="0.85", linewidth=0.8)
            ax.text(
                0.01,
                hy,
                name,
                transform=blended_transform_factory(ax.transAxes, ax.transData),
                fontsize=8,
                fontweight="bold",
                va="center",
                ha="left",
            )
    row[0].set_yticklabels(yticklabels, fontsize=7)
    return fig, row


def _forest_marker(
    ax: plt.Axes,
    y: float,
    values: np.ndarray,
    mean: float,
    lo: float,
    hi: float,
    color,
    marker: str = "D",
    filled: bool = True,
) -> None:
    """Draw one rung: 95% CI bar (``lo``/``hi`` = mean±ci) + seed dots + mean marker."""
    n = len(values)
    jit = np.linspace(-0.12, 0.12, n) if n > 1 else np.zeros(1)
    ax.plot([lo, hi], [y, y], color=color, linewidth=1.1, alpha=0.7, zorder=1)
    ax.scatter(
        values,
        y + jit,
        s=14,
        facecolors=color,
        edgecolors="none",
        alpha=0.45,
        zorder=2,
    )
    ax.scatter(
        [mean],
        [y],
        s=46,
        marker=marker,
        facecolors=color if filled else "white",
        edgecolors=color,
        linewidths=1.2,
        zorder=3,
    )


def plot_forest_delta(
    arch_df: pd.DataFrame,
    ladder_specs: dict[str, list[str]],
    baseline: str,
    out_path_stem: Path,
) -> None:
    """Forest of per-seed Δacc = Learned − baseline, one panel per dataset.

    Shared x across panels (the robustness message reads as "every interval
    right of the dashed Δ=0 line"). The anchor rung (mlp-128) recurs once in each
    MLP block by construction — expected.
    """
    if arch_df.empty:
        print(f"  [skip] no arch rows for {out_path_stem.name}")
        return

    columns = _forest_columns(arch_df)
    blocks = _forest_blocks(arch_df, ladder_specs)
    if not columns or not blocks:
        print(f"  [skip] empty forest for {out_path_stem.name}")
        return

    agg, merged = paired_delta(arch_df, baseline, "arch_label")
    if agg.empty:
        print(f"  [skip] no paired data Learned vs {baseline} for {out_path_stem.name}")
        return

    rung_y, header_y, y_bottom = _forest_layout(blocks)
    fig, row = _forest_figure(len(columns), blocks, rung_y, header_y, y_bottom, sharex=True)
    color = SCHEDULE_COLORS[LEARNED]

    for c, (ds, eps) in enumerate(columns):
        ax = row[c]
        ax.axvline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(ds if len(columns) == len({d for d, _ in columns}) else f"{ds}, ε={eps:g}")
        for name, labels in blocks:
            for lbl in labels:
                y = rung_y[(name, lbl)]
                arow = agg[
                    (agg["dataset"] == ds) & (agg["eps"] == eps) & (agg["arch_label"] == lbl)
                ]
                if arow.empty:
                    continue
                seeds = merged[
                    (merged["dataset"] == ds)
                    & (merged["eps"] == eps)
                    & (merged["arch_label"] == lbl)
                ]["delta"].to_numpy()
                r = arow.iloc[0]
                _forest_marker(
                    ax,
                    y,
                    seeds,
                    r["mean"],
                    r["mean"] - r["ci"],
                    r["mean"] + r["ci"],
                    color,
                    marker="D",
                )

    for ax in row:
        ax.set_xlabel(f"Learned − {SCHEDULE_SHORT[baseline]} (Δ acc, %)")

    handles = [
        Line2D([], [], color=color, marker="D", linestyle="none", label="seed mean"),
        Line2D([], [], color=color, marker="o", linestyle="none", alpha=0.45, label="per seed"),
        Line2D([], [], color=color, linewidth=1.1, alpha=0.7, label="95% CI"),
        Line2D(
            [],
            [],
            color="black",
            linestyle="--",
            linewidth=0.8,
            label=f"Δ=0 ({SCHEDULE_SHORT[baseline]})",
        ),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), frameon=False
    )
    fig.suptitle("Architecture robustness — Learned vs Constant Δ accuracy", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    _save(fig, out_path_stem)


def plot_forest_abs(
    arch_df: pd.DataFrame,
    ladder_specs: dict[str, list[str]],
    out_path_stem: Path,
) -> None:
    """Forest of paired absolute accuracy (Constant ○ / Learned ●) per rung.

    Independent x per dataset panel (mnist and fashion-mnist sit in different
    accuracy ranges), so each column is auto-scaled.
    """
    if arch_df.empty:
        print(f"  [skip] no arch rows for {out_path_stem.name}")
        return

    columns = _forest_columns(arch_df)
    blocks = _forest_blocks(arch_df, ladder_specs)
    if not columns or not blocks:
        print(f"  [skip] empty forest for {out_path_stem.name}")
        return

    agg = aggregate_across_seeds(arch_df, "arch_label", "mean_acc")
    rung_y, header_y, y_bottom = _forest_layout(blocks)
    fig, row = _forest_figure(len(columns), blocks, rung_y, header_y, y_bottom, sharex=False)

    # Learned filled, Constant open; offset in y so the two don't overlap.
    schedule_style = [(LEARNED, "o", True, 0.16), (CONSTANT, "o", False, -0.16)]

    for c, (ds, eps) in enumerate(columns):
        ax = row[c]
        ax.set_title(ds if len(columns) == len({d for d, _ in columns}) else f"{ds}, ε={eps:g}")
        for name, labels in blocks:
            for lbl in labels:
                y = rung_y[(name, lbl)]
                for sched, marker, filled, yoff in schedule_style:
                    arow = agg[
                        (agg["dataset"] == ds)
                        & (agg["eps"] == eps)
                        & (agg["schedule"] == sched)
                        & (agg["arch_label"] == lbl)
                    ]
                    if arow.empty:
                        continue
                    seeds = arch_df[
                        (arch_df["dataset"] == ds)
                        & (arch_df["eps"] == eps)
                        & (arch_df["schedule"] == sched)
                        & (arch_df["arch_label"] == lbl)
                    ]["mean_acc"].to_numpy()
                    r = arow.iloc[0]
                    _forest_marker(
                        ax,
                        y + yoff,
                        seeds,
                        r["mean"],
                        r["mean"] - r["ci"],
                        r["mean"] + r["ci"],
                        SCHEDULE_COLORS[sched],
                        marker=marker,
                        filled=filled,
                    )

    for ax in row:
        ax.set_xlabel("test accuracy (%)")

    handles = [
        Line2D(
            [],
            [],
            color=SCHEDULE_COLORS[LEARNED],
            marker="o",
            linestyle="none",
            label=SCHEDULE_SHORT[LEARNED],
        ),
        Line2D(
            [],
            [],
            color=SCHEDULE_COLORS[CONSTANT],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            label=SCHEDULE_SHORT[CONSTANT],
        ),
        Line2D([], [], color="0.4", marker="o", linestyle="none", alpha=0.45, label="per seed"),
        Line2D([], [], color="0.4", linewidth=1.1, alpha=0.7, label="95% CI"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), frameon=False
    )
    fig.suptitle("Architecture robustness — paired absolute accuracy", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
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


def plot_shape_variant(
    df: pd.DataFrame,
    var: str,  # "sigma" or "clip"
    color_col: str,  # "T", "arch_label", or "seed"
    out_path_stem: Path,
    value_order: list | None = None,
    title: str | None = None,
) -> None:
    """Shape plot: thin per-run lines colored by ``color_col``, bold per-value mean.

    Grid: rows = ε, cols = dataset. ``df`` is already filtered to the relevant
    rows (T-sweep, or one ladder). For structural axes (T, arch_label) a bold
    per-value mean across seeds is drawn; ``seed`` is a sanity-check axis (thin
    lines only). ``value_order`` pins the colour/legend order (used for ladder
    rungs, whose order is not the param-count order).
    """
    if df.empty:
        print(f"  [skip] no schedule rows for {out_path_stem.name}")
        return

    datasets = sorted(df["dataset"].unique())
    epsilons = sorted(df["eps"].dropna().unique())
    if not datasets or not epsilons:
        print(f"  [skip] empty grid for {out_path_stem.name}")
        return

    if color_col == "seed":
        values = sorted(df["seed"].dropna().unique())
        cmap = plt.get_cmap("tab10")
        value_to_color = {v: cmap(i % 10) for i, v in enumerate(values)}
        draw_mean = False
        legend_label = lambda v: f"seed={int(v)}"  # noqa: E731
    elif color_col == "T":
        values = sorted(df["T"].dropna().unique())
        cmap = plt.get_cmap("viridis")
        value_to_color = {v: cmap(i / max(1, len(values) - 1)) for i, v in enumerate(values)}
        draw_mean = True
        legend_label = lambda v: f"T={int(v)}"  # noqa: E731
    elif color_col == "arch_label":
        if value_order is not None:
            present = set(df["arch_label"].dropna().unique())
            values = [v for v in value_order if v in present]
        else:
            values = list(_param_map(df).sort_values().index)
        cmap = plt.get_cmap("viridis")
        value_to_color = {v: cmap(i / max(1, len(values) - 1)) for i, v in enumerate(values)}
        draw_mean = True
        legend_label = lambda v: str(v)  # noqa: E731
    else:
        raise ValueError(f"unknown color_col: {color_col}")

    if not values:
        print(f"  [skip] no {color_col} values for {out_path_stem.name}")
        return

    nrows, ncols = len(epsilons), len(datasets)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.5 * ncols + 1, 2.6 * nrows + 0.5),
        squeeze=False,
        sharex=True,
        sharey="row",
    )

    for i, eps in enumerate(epsilons):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            cell = df[(df["dataset"] == ds) & (df["eps"] == eps)]
            if cell.empty:
                continue
            for val in values:
                grp_df = cell[cell[color_col] == val]
                if grp_df.empty:
                    continue
                color = value_to_color[val]
                for _run_id, run_grp in grp_df.groupby("run_id"):
                    run_grp = run_grp.sort_values("inner_step")
                    ax.plot(
                        run_grp["step_norm"],
                        run_grp[var],
                        color=color,
                        alpha=0.35,
                        linewidth=0.7,
                    )
                if draw_mean:
                    mean_vals = grp_df.groupby("inner_step")[var].mean().sort_index()
                    mean_x = grp_df.groupby("inner_step")["step_norm"].first().sort_index()
                    ax.plot(
                        mean_x.to_numpy(),
                        mean_vals.to_numpy(),
                        color=color,
                        linewidth=1.8,
                    )
            ax.grid(True, alpha=0.3, linewidth=0.5)

    for j, ds in enumerate(datasets):
        axes[0, j].set_title(ds)
    for i, eps in enumerate(epsilons):
        axes[i, 0].set_ylabel(f"ε = {eps:g}\n{var}")
    for j in range(ncols):
        axes[-1, j].set_xlabel("t / T")

    legend_handles = [
        Line2D([], [], color=value_to_color[v], label=legend_label(v)) for v in values
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(values), 6),
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
    )
    suptitle = title or (
        f"{var} schedules — colored by {color_col}"
        + ("" if draw_mean else " (thin lines only — sanity check)")
    )
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    _save(fig, out_path_stem)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _fmt_cell(mean: float | None, ci: float | None) -> str:
    if mean is None or (isinstance(mean, float) and math.isnan(mean)):
        return "—"
    if ci is None or (isinstance(ci, float) and math.isnan(ci)) or ci == 0:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {ci:.3f}"


def build_table(
    df: pd.DataFrame,
    xaxis: XAxis,
) -> pd.DataFrame:
    """Winner-bolded mean ± 95% CI table, indexed by (dataset, eps, x) ordered by ``xaxis``."""
    if df.empty:
        return pd.DataFrame()

    x_col = xaxis.col
    agg = aggregate_across_seeds(df, x_col, "mean_acc")
    pivot_mean = agg.pivot_table(
        index=["dataset", "eps", x_col],
        columns="schedule",
        values="mean",
        aggfunc="first",
    )
    pivot_ci = agg.pivot_table(
        index=["dataset", "eps", x_col],
        columns="schedule",
        values="ci",
        aggfunc="first",
    )

    cols = [s for s in SCHEDULE_ORDER if s in pivot_mean.columns]
    pivot_mean = pivot_mean[cols]
    pivot_ci = pivot_ci.reindex(columns=cols)

    idx = pivot_mean.reset_index()
    idx["xpos"] = idx[x_col].map(xaxis.to_pos)
    idx = idx.sort_values(["dataset", "eps", "xpos"])
    order = list(zip(idx["dataset"], idx["eps"], idx[x_col]))
    pivot_mean = pivot_mean.loc[order]
    pivot_ci = pivot_ci.reindex(pivot_mean.index)

    return _format_table_with_winners(pivot_mean, pivot_ci)


def _format_table_with_winners(pivot_mean: pd.DataFrame, pivot_ci: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=pivot_mean.index, columns=pivot_mean.columns, dtype=object)
    for idx, row in pivot_mean.iterrows():
        winner = row.idxmax(skipna=True) if not row.isna().all() else None
        for col in pivot_mean.columns:
            cell = _fmt_cell(row[col], pivot_ci.loc[idx, col])
            if col == winner and cell != "—":
                cell = f"**{cell}**"
            out.loc[idx, col] = cell
    out.columns = [SCHEDULE_SHORT[c] for c in out.columns]
    return out


def write_tables(table: pd.DataFrame, stem: Path) -> None:
    if table.empty:
        return
    stem.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(stem.with_suffix(".csv"))
    _write_latex(table, stem.with_suffix(".tex"))


def _write_latex(table: pd.DataFrame, path: Path) -> None:
    cols = list(table.columns)
    idx_names = list(table.index.names)
    n_idx = len(idx_names)
    n_total = n_idx + len(cols)
    col_spec = "l" * n_idx + "c" * len(cols)

    lines: list[str] = []
    lines.append("\\begin{tabular}{" + col_spec + "}")
    lines.append("\\toprule")
    header = " & ".join(idx_names + cols) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    prev_dataset: object = None
    prev_eps: object = None
    for i, (idx, row) in enumerate(table.iterrows()):
        idx_tuple = idx if isinstance(idx, tuple) else (idx,)
        dataset = idx_tuple[0] if n_idx >= 1 else None
        eps = idx_tuple[1] if n_idx >= 2 else None

        if i > 0:
            if dataset != prev_dataset:
                lines.append("\\midrule")
            elif eps != prev_eps:
                lines.append(f"\\cmidrule(lr){{2-{n_total}}}")

        new_dataset = dataset != prev_dataset
        new_eps = new_dataset or eps != prev_eps

        idx_cells: list[str] = []
        for k, v in enumerate(idx_tuple):
            if k == 0:
                cell = _latex_escape(str(v)) if new_dataset else ""
            elif k == 1:
                cell = _latex_escape(str(v)) if new_eps else ""
            else:
                cell = _latex_escape(str(v))
            idx_cells.append(cell)

        body_cells = [_latex_cell(v) for v in row.tolist()]
        lines.append(" & ".join(idx_cells + body_cells) + " \\\\")

        prev_dataset = dataset
        prev_eps = eps

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
# Training-curve plots (Learned only) — T-sweep
# ---------------------------------------------------------------------------


def plot_curves(
    histories_df: pd.DataFrame,
    metric: str,  # "test_loss" or "test_acc"
    dataset: str,
    out_path_stem: Path,
) -> None:
    """Per-(T, eps) grid of outer-loop training curves for one dataset (T-sweep).

    If the held-fixed arch has more than one value in the data, emit one plot per
    value with a filename suffix and warn that the plot was split.
    """
    df = histories_df[
        _axis_mask(histories_df, "T-sweep") & (histories_df["dataset"] == dataset)
    ].copy()
    if df.empty:
        print(f"  [skip] no T-sweep history for dataset={dataset}")
        return

    held_values = sorted(df["arch_label"].dropna().unique())

    if len(held_values) > 1:
        warnings.warn(
            f"dataset={dataset}: held-fixed arch has {len(held_values)} distinct "
            f"values {held_values}; splitting plot into one per value.",
            stacklevel=2,
        )
        for hv in held_values:
            sub = df[df["arch_label"] == hv]
            safe = str(hv).replace("/", "_").replace(" ", "_")
            sub_stem = out_path_stem.with_name(f"{out_path_stem.name}__arch-{safe}")
            _plot_curves_single(sub, metric, dataset, hv, sub_stem)
        return

    held_val = held_values[0] if held_values else None
    _plot_curves_single(df, metric, dataset, held_val, out_path_stem)


def _plot_curves_single(
    df: pd.DataFrame,
    metric: str,
    dataset: str,
    held_val: object,
    out_path_stem: Path,
) -> None:
    row_vals = sorted(df["T"].dropna().unique())
    epsilons = sorted(df["eps"].dropna().unique())
    if not row_vals or not epsilons:
        print(f"  [skip] empty grid for dataset={dataset}")
        return

    nrows, ncols = len(row_vals), len(epsilons)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.0 * ncols + 1, 2.0 * nrows + 0.5),
        squeeze=False,
        sharex=True,
        sharey="row",
    )

    ylabel = "test loss" if metric == "test_loss" else "test accuracy"

    for i, rv in enumerate(row_vals):
        for j, eps in enumerate(epsilons):
            ax = axes[i, j]
            cell = df[(df["T"] == rv) & (df["eps"] == eps)]
            if cell.empty:
                ax.set_visible(False)
                continue

            seed_groups = list(cell.groupby("seed", dropna=False))
            seed_lengths = {s: len(g) for s, g in seed_groups}
            if len(set(seed_lengths.values())) > 1:
                warnings.warn(
                    f"seeds disagree on outer-step count for "
                    f"dataset={dataset}, eps={eps}, T={rv}: {seed_lengths}",
                    stacklevel=2,
                )
                max_len = max(seed_lengths.values())
                seed_groups = [(s, g) for (s, g) in seed_groups if len(g) == max_len]

            sorted_groups = [g.sort_values("outer_step") for _, g in seed_groups]
            xs = sorted_groups[0]["outer_step"].to_numpy()
            for grp in sorted_groups:
                ax.plot(
                    xs,
                    grp[metric].to_numpy(dtype=float),
                    color="#1f77b4",
                    alpha=0.35,
                    linewidth=0.7,
                )
            stacked = np.vstack([g[metric].to_numpy(dtype=float) for g in sorted_groups])
            ax.plot(xs, np.nanmean(stacked, axis=0), color="#1f77b4", linewidth=1.6)

            ax.grid(True, alpha=0.3, linewidth=0.5)

    for j, eps in enumerate(epsilons):
        axes[0, j].set_title(f"ε = {eps:g}")
    for i, rv in enumerate(row_vals):
        axes[i, 0].set_ylabel(f"T={int(rv)}\n{ylabel}")
    for j in range(ncols):
        axes[-1, j].set_xlabel("outer step")

    held_str = "" if held_val is None else f", arch={held_val}"
    fig.suptitle(f"{dataset} — Learned schedule training curves (T-sweep{held_str})", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out_path_stem)


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


def _plot_t_sweep(s: pd.DataFrame, sch: pd.DataFrame, histories: pd.DataFrame, out: Path) -> None:
    """T-sweep scalar plots, tables, all-runs shape, shape-variants and curves."""
    t = s[_axis_mask(s, "T-sweep")]
    if t.empty:
        print("  [skip] no T-sweep rows")
    else:
        xaxis_t = _xaxis_t(t)
        plot_main(t, xaxis_t, "mean_acc", out / "t_sweep_main")
        plot_delta(t, xaxis_t, CONSTANT, "mean_acc", out / "t_sweep_delta_vs_constant")
        plot_delta(t, xaxis_t, DYNAMIC, "mean_acc", out / "t_sweep_delta_vs_dynamic")
        tbl = build_table(t, xaxis_t)
        if not tbl.empty:
            write_tables(tbl, out / "t_sweep_table")
            print("  ✓ t_sweep_table.csv / .tex")

    plot_shape(sch, "sigma", out / "sigma_shape")
    plot_shape(sch, "clip", out / "clip_shape")

    sch_t = sch[_axis_mask(sch, "T-sweep")]
    variants_dir = out / "shape_variants"
    for var in ("sigma", "clip"):
        plot_shape_variant(sch_t, var, "T", variants_dir / f"{var}_shape__T_sweep__by_T")
        plot_shape_variant(sch_t, var, "seed", variants_dir / f"{var}_shape__T_sweep__by_seed")

    if not histories.empty:
        curves_dir = out / "curves"
        for dataset in sorted(histories["dataset"].unique()):
            plot_curves(histories, "test_loss", dataset, curves_dir / f"t_sweep_loss__{dataset}")
            plot_curves(histories, "test_acc", dataset, curves_dir / f"t_sweep_acc__{dataset}")


def _plot_ladders(s: pd.DataFrame, sch: pd.DataFrame, out: Path) -> None:
    """Per-ladder scalar/table/shape plots and cross-ladder overlays."""
    ladder_specs = _ladder_specs()
    s_arch = s[_axis_mask(s, "arch")]
    sch_arch = sch[_axis_mask(sch, "arch")]
    if s_arch.empty:
        print("  [skip] no arch rows — no ladder plots")
        return

    ladders_dir = out / "ladders"
    for name, ordered_labels in ladder_specs.items():
        member = s_arch[_ladder_member_mask(s_arch, name)]
        if member.empty:
            print(f"  [skip] ladder {name}: no member runs")
            continue
        d = ladders_dir / name
        xaxis_l = _xaxis_ladder(member, ordered_labels)
        plot_main(member, xaxis_l, "mean_acc", d / "main")
        plot_delta(member, xaxis_l, CONSTANT, "mean_acc", d / "delta_vs_constant")
        plot_delta(member, xaxis_l, DYNAMIC, "mean_acc", d / "delta_vs_dynamic")
        tbl = build_table(member, xaxis_l)
        if not tbl.empty:
            write_tables(tbl, d / "table")
            print(f"  ✓ {name}/table.csv / .tex")

        sch_member = sch_arch[_ladder_member_mask(sch_arch, name)]
        for var in ("sigma", "clip"):
            plot_shape_variant(
                sch_member,
                var,
                "arch_label",
                d / f"{var}_shape_by_rung",
                value_order=ordered_labels,
                title=f"{name} — {var} schedules by rung",
            )

    overall = ladders_dir / "overall"
    # Remove the superseded param-count overlays so stale figures don't linger.
    for stale in ("arch_overlay_delta_vs_constant", "arch_overlay_acc"):
        for ext in ("pdf", "png"):
            (overall / stale).with_suffix(f".{ext}").unlink(missing_ok=True)
    plot_forest_delta(s_arch, ladder_specs, CONSTANT, overall / "arch_forest_delta")
    plot_forest_abs(s_arch, ladder_specs, overall / "arch_forest_abs")


def main(conf: PlotConfig) -> None:
    in_dir = Path(conf.in_dir)
    scalars = pd.read_parquet(in_dir / "scalars.parquet")
    schedules = pd.read_parquet(in_dir / "schedules.parquet")
    histories_path = in_dir / "histories.parquet"
    histories = pd.read_parquet(histories_path) if histories_path.exists() else pd.DataFrame()
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

        h = histories[histories["optimizer"] == opt] if not histories.empty else pd.DataFrame()
        _plot_t_sweep(s, sch, h, out)
        _plot_ladders(s, sch, out)


if __name__ == "__main__":
    main(tyro.cli(PlotConfig))
