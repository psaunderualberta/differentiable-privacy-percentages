#!/usr/bin/env python3
"""symbolic_regression_eval.py — Assess how well the equations found by
``symbolic_regression.py`` match the learned σ/C/μ schedules.

Reads the artefacts ``symbolic_regression.py`` persists under ``<cache_dir>/pysr_eval``:

    manifest.json          filters + included runs that produced the models
    features_full.parquet  per-run, EVERY inner step: features + actual σ/C/μ
    <target>/model.pkl     pickled PySRRegressor (carries the whole Pareto front)
    <target>/equations.csv distilled front (complexity, loss, selected)
    <target>/feature_names.json  exact feature columns the target was fit on

Evaluation is **per-run, in-sample, full-T**: for each run the selected equation
is evaluated across all inner steps to form a *predicted schedule*, compared to
that run's *actual* schedule. Per target it writes::

    <out>/<target>/
        pareto.{pdf,png}              fit-vs-complexity over the whole front
        overlay_grid.{pdf,png}        all runs: actual (solid) vs predicted (dashed)
        spotlights.{pdf,png}          best / median / worst run by NRMSE
        metric_distributions.{pdf,png}
        residual_vs_step.{pdf,png}
        metrics_per_run.csv
        summary_table.{csv,tex}
    <out>/privacy/                    only when BOTH σ and clip equations exist
        privacy_validity.{pdf,png}
        privacy_per_run.csv

Metrics: NRMSE-by-mean (headline), RMSLE, RMSE, caveated R² (flat runs flagged
and excluded from summaries), plus extremum location/value errors. No
correlation scalars — see docs/adr/0001. Privacy validity uses the budget
recovered from the (on-budget) actual schedule, so it needs neither δ nor p.

Usage (from src/):
    uv run symbolic_regression_eval.py --eval-dir cache/results/<entity>__<project>/pysr_eval
"""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro
from matplotlib.lines import Line2D

TARGETS: tuple[str, ...] = ("sigma", "clip", "mu")
_MAX_MU2 = 80.0  # mirror gdp_privacy._MAX_MU2 — cap (C/σ)² before exp() to avoid overflow
_EPS_POS = 1e-12  # floor for log/ratio of strictly-positive quantities


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class TargetModel:
    target: str
    model: object  # PySRRegressor
    feature_names: list[str]
    equations: pd.DataFrame  # the distilled front (incl. "selected" column)


def _load_target(eval_dir: Path, target: str) -> TargetModel | None:
    tdir = eval_dir / target
    if not (tdir / "model.pkl").exists():
        return None
    with (tdir / "model.pkl").open("rb") as f:
        model = pickle.load(f)
    feature_names = json.loads((tdir / "feature_names.json").read_text())
    equations = pd.read_csv(tdir / "equations.csv")
    return TargetModel(target, model, feature_names, equations)


def _predict(tm: TargetModel, df: pd.DataFrame, index: int | None) -> np.ndarray:
    """Evaluate the equation (selected when ``index is None``) on ``df``'s rows."""
    X = df[tm.feature_names].to_numpy()
    if index is None:
        return np.asarray(tm.model.predict(X), dtype=float)
    return np.asarray(tm.model.predict(X, index=index), dtype=float)


def _selected_index(tm: TargetModel) -> int:
    sel = tm.equations.index[tm.equations["selected"]]
    return int(sel[0]) if len(sel) else int(len(tm.equations) - 1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, _EPS_POS, None))


def per_run_metrics(actual: np.ndarray, pred: np.ndarray, step_norm: np.ndarray) -> dict:
    """Position- and scale-aware fit metrics for one run's schedule.

    Returns NaNs for entries that are undefined (e.g. non-finite predictions);
    callers drop / flag those rather than letting them dominate summaries.
    """
    finite = np.isfinite(pred) & np.isfinite(actual)
    out: dict[str, float] = {"n": len(actual), "n_finite": int(finite.sum())}
    if finite.sum() < 2:
        return {**out, **dict.fromkeys(_METRIC_KEYS, np.nan), "flat": True}

    a, p, s = actual[finite], pred[finite], step_norm[finite]
    resid = p - a
    mean_a = float(np.mean(a))
    std_a = float(np.std(a))

    rmse = float(np.sqrt(np.mean(resid**2)))
    out["rmse"] = rmse
    out["nrmse"] = rmse / abs(mean_a) if mean_a != 0 else np.nan
    out["rmsle"] = float(np.sqrt(np.mean((_safe_log(p) - _safe_log(a)) ** 2)))

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((a - mean_a) ** 2))
    out["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # Near-flat actual schedule → R² denominator collapses; flag and exclude later.
    out["flat"] = bool(abs(mean_a) < _EPS_POS or std_a / abs(mean_a) < _FLAT_REL_STD)

    # Shape: extremum location (in t/T) and value errors — what correlation can't see.
    out["peak_loc_err"] = float(s[np.argmax(p)] - s[np.argmax(a)])
    out["trough_loc_err"] = float(s[np.argmin(p)] - s[np.argmin(a)])
    out["max_val_rel_err"] = float((np.max(p) - np.max(a)) / (abs(np.max(a)) + _EPS_POS))
    out["min_val_rel_err"] = float((np.min(p) - np.min(a)) / (abs(np.min(a)) + _EPS_POS))
    return out


_METRIC_KEYS = (
    "rmse",
    "nrmse",
    "rmsle",
    "r2",
    "peak_loc_err",
    "trough_loc_err",
    "max_val_rel_err",
    "min_val_rel_err",
)

# Set from config in main() before metrics run.
_FLAT_REL_STD = 0.05


# ---------------------------------------------------------------------------
# Privacy-budget fidelity (recovers the budget from the on-budget actual schedule)
# ---------------------------------------------------------------------------


def _expenditure_sq(mu_schedule: np.ndarray) -> float:
    """Σ(exp(μ²) − 1) — proportional to squared privacy expenditure (p² factor)."""
    mu2 = np.clip(mu_schedule**2, None, _MAX_MU2)
    return float(np.sum(np.exp(mu2) - 1.0))


def budget_ratio(mu_pred: np.ndarray, mu_actual: np.ndarray) -> float:
    """Predicted privacy expenditure ÷ target μ.

    The actual schedule sits on the budget surface, so target μ = expenditure of
    the actual schedule; the unknown p and δ cancel in the ratio. >1 ⇒ the
    predicted schedule over-spends the budget (a privacy violation); <1 ⇒ wasteful.
    """
    denom = _expenditure_sq(mu_actual)
    if denom <= 0:
        return np.nan
    return float(np.sqrt(_expenditure_sq(mu_pred) / denom))


def projection_distance(
    sigma_pred: np.ndarray, clip_pred: np.ndarray, mu_actual: np.ndarray
) -> float:
    """Relative L2 distance the predicted schedule must move to become budget-valid.

    Recovers the bound B = Σexp(μ_act²) (the actual schedule sits on it) and
    L2-projects the predicted (σ, C) onto it with the same solver DP-SGD uses.
    """
    import jax.numpy as jnp

    from privacy.gdp_privacy import project_sigma_and_clip_given_bound

    s = np.clip(sigma_pred, _EPS_POS, None)
    c = np.clip(clip_pred, _EPS_POS, None)
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(c))):
        return np.nan
    B = float(np.sum(np.exp(np.clip(mu_actual**2, None, _MAX_MU2))))
    try:
        ps, pc = project_sigma_and_clip_given_bound(jnp.asarray(s), jnp.asarray(c), jnp.asarray(B))
        ps, pc = np.asarray(ps), np.asarray(pc)
    except Exception as exc:
        warnings.warn(f"projection failed: {exc}", stacklevel=2)
        return np.nan
    moved = np.sqrt(np.sum((ps - s) ** 2) + np.sum((pc - c) ** 2))
    orig = np.sqrt(np.sum(s**2) + np.sum(c**2))
    return float(moved / orig) if orig > 0 else np.nan


# ---------------------------------------------------------------------------
# Per-run evaluation driver
# ---------------------------------------------------------------------------

_META_COLS = ("run_id", "dataset", "eps", "T", "arch_label", "seed", "axis", "optimizer")


def evaluate_runs(
    full: pd.DataFrame,
    models: dict[str, TargetModel],
    complexity: int,
    compute_projection: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-run metrics for every target + (when σ and clip both fit) privacy.

    Returns (metrics_df, privacy_df). Predictions are attached to ``full`` in
    place as ``<target>_pred`` so the plotting functions can reuse them.
    """
    for t, tm in models.items():
        idx = None if complexity < 0 else _index_for_complexity(tm, complexity)
        full[f"{t}_pred"] = _predict(tm, full, idx)

    have_priv = "sigma" in models and "clip" in models
    metric_rows: list[dict] = []
    priv_rows: list[dict] = []

    for _run_id, g in full.groupby("run_id", sort=False):
        g = g.sort_values("inner_step")
        meta = {c: g[c].iloc[0] for c in _META_COLS if c in g.columns}
        step_norm = g["step_norm"].to_numpy()

        for t, _tm in models.items():
            m = per_run_metrics(g[t].to_numpy(float), g[f"{t}_pred"].to_numpy(float), step_norm)
            metric_rows.append({**meta, "target": t, **m})

        if have_priv:
            mu_act = (g["clip"] / g["sigma"]).to_numpy(float)
            mu_pred_paired = (g["clip_pred"] / g["sigma_pred"]).to_numpy(float)
            row = {
                **meta,
                "budget_ratio": budget_ratio(mu_pred_paired, mu_act),
            }
            if "mu" in models:  # optional cross-check: μ-equation directly
                row["budget_ratio_mu_eq"] = budget_ratio(g["mu_pred"].to_numpy(float), mu_act)
            if compute_projection:
                row["projection_distance"] = projection_distance(
                    g["sigma_pred"].to_numpy(float), g["clip_pred"].to_numpy(float), mu_act
                )
            priv_rows.append(row)

    return pd.DataFrame(metric_rows), pd.DataFrame(priv_rows)


def _index_for_complexity(tm: TargetModel, complexity: int) -> int:
    eqs = tm.equations
    hits = eqs.index[eqs["complexity"] == complexity]
    if not len(hits):
        warnings.warn(
            f"[{tm.target}] no equation with complexity={complexity}; using selected",
            stacklevel=2,
        )
        return _selected_index(tm)
    return int(hits[0])


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  ✓ {stem.relative_to(stem.parents[1])}.pdf")


def plot_pareto(tm: TargetModel, full: pd.DataFrame, stem: Path) -> None:
    """Pooled NRMSE / RMSLE across the whole front vs complexity — which to trust."""
    actual = full[tm.target].to_numpy(float)
    mean_a = abs(float(np.mean(actual[np.isfinite(actual)]))) or 1.0
    comps, nrmses, rmsles = [], [], []
    for i in tm.equations.index:
        pred = _predict(tm, full, int(i))
        finite = np.isfinite(pred) & np.isfinite(actual)
        if finite.sum() < 2:
            nrmses.append(np.nan)
            rmsles.append(np.nan)
        else:
            a, p = actual[finite], pred[finite]
            nrmses.append(float(np.sqrt(np.mean((p - a) ** 2)) / mean_a))
            rmsles.append(float(np.sqrt(np.mean((_safe_log(p) - _safe_log(a)) ** 2))))
        comps.append(float(tm.equations.loc[i, "complexity"]))

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(comps, nrmses, marker="o", color="#1f77b4", label="pooled NRMSE")
    ax.set_xlabel("equation complexity")
    ax.set_ylabel("pooled NRMSE", color="#1f77b4")
    ax.set_yscale("log")
    ax2 = ax.twinx()
    ax2.plot(comps, rmsles, marker="s", color="#d62728", label="pooled RMSLE")
    ax2.set_ylabel("pooled RMSLE", color="#d62728")
    sel = _selected_index(tm)
    ax.axvline(float(tm.equations.loc[sel, "complexity"]), color="black", ls="--", lw=0.8)
    ax.set_title(f"{tm.target}: fit vs complexity (dashed = selected)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, stem)


def _facet_eps_dataset(full: pd.DataFrame) -> tuple[list, list]:
    return sorted(full["eps"].dropna().unique()), sorted(full["dataset"].dropna().unique())


def plot_overlay_grid(full: pd.DataFrame, target: str, stem: Path) -> None:
    """All runs: actual (solid) vs predicted (dashed), faceted ε × dataset."""
    epsilons, datasets = _facet_eps_dataset(full)
    if not epsilons or not datasets:
        return
    fig, axes = plt.subplots(
        len(epsilons),
        len(datasets),
        figsize=(3.5 * len(datasets) + 1, 2.6 * len(epsilons) + 0.5),
        squeeze=False,
        sharex=True,
        sharey="row",
    )
    for i, eps in enumerate(epsilons):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            cell = full[(full["eps"] == eps) & (full["dataset"] == ds)]
            for _rid, g in cell.groupby("run_id", sort=False):
                g = g.sort_values("inner_step")
                ax.plot(g["step_norm"], g[target], color="#1f77b4", alpha=0.3, lw=0.6)
                ax.plot(
                    g["step_norm"], g[f"{target}_pred"], color="#d62728", alpha=0.3, lw=0.6, ls="--"
                )
            ax.grid(True, alpha=0.3, lw=0.5)
    for j, ds in enumerate(datasets):
        axes[0, j].set_title(ds)
    for i, eps in enumerate(epsilons):
        axes[i, 0].set_ylabel(f"ε = {eps:g}\n{target}")
    for j in range(len(datasets)):
        axes[-1, j].set_xlabel("t / T")
    fig.legend(
        handles=[
            Line2D([], [], color="#1f77b4", label="actual"),
            Line2D([], [], color="#d62728", ls="--", label="predicted"),
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.04),
        frameon=False,
    )
    fig.suptitle(f"{target}: actual vs predicted schedules (all runs)", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    _save(fig, stem)


def plot_spotlights(full: pd.DataFrame, metrics: pd.DataFrame, target: str, stem: Path) -> None:
    """Best / median / worst run by NRMSE — actual vs predicted, clean panels."""
    sub = metrics[(metrics["target"] == target) & np.isfinite(metrics["nrmse"])].copy()
    if sub.empty:
        return
    sub = sub.sort_values("nrmse").reset_index(drop=True)
    picks = {
        "best": sub.iloc[0],
        "median": sub.iloc[len(sub) // 2],
        "worst": sub.iloc[-1],
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2), squeeze=False)
    for ax, (label, row) in zip(axes[0], picks.items()):
        g = full[full["run_id"] == row["run_id"]].sort_values("inner_step")
        ax.plot(g["step_norm"], g[target], color="#1f77b4", lw=1.4, label="actual")
        ax.plot(
            g["step_norm"], g[f"{target}_pred"], color="#d62728", lw=1.4, ls="--", label="predicted"
        )
        ax.set_title(
            f"{label}: {row['dataset']} ε={row['eps']:g} T={int(row['T'])}\nNRMSE={row['nrmse']:.3f}",
            fontsize=9,
        )
        ax.set_xlabel("t / T")
        ax.grid(True, alpha=0.3, lw=0.5)
    axes[0, 0].set_ylabel(target)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"{target}: best / median / worst fit by NRMSE", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, stem)


def plot_metric_distributions(metrics: pd.DataFrame, target: str, stem: Path) -> None:
    """Box plots of each scalar metric grouped by (dataset, ε). R² excludes flat runs."""
    sub = metrics[metrics["target"] == target].copy()
    if sub.empty:
        return
    sub["group"] = sub["dataset"] + "\nε=" + sub["eps"].map(lambda e: f"{e:g}")
    groups = sorted(sub["group"].unique())
    panels = [
        ("nrmse", "NRMSE", sub),
        ("rmsle", "RMSLE", sub),
        ("r2", "R² (flat runs excluded)", sub[~sub["flat"]]),
        ("peak_loc_err", "peak-location error (t/T)", sub),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(max(6, 1.1 * len(groups)), 3.0 * len(panels)))
    for ax, (key, ylabel, data) in zip(np.atleast_1d(axes), panels):
        box = [
            data[(data["group"] == grp)][key].replace([np.inf, -np.inf], np.nan).dropna()
            for grp in groups
        ]
        ax.boxplot(box, tick_labels=groups, showfliers=False)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3, lw=0.5)
        if key in ("peak_loc_err",):
            ax.axhline(0.0, color="black", lw=0.6, alpha=0.6)
    for lab in axes[-1].get_xticklabels():
        lab.set_rotation(30)
        lab.set_ha("right")
    fig.suptitle(f"{target}: per-run fit metric distributions", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, stem)


def plot_residual_vs_step(full: pd.DataFrame, target: str, stem: Path) -> None:
    """Mean residual (predicted − actual) vs t/T ± std across runs, faceted by dataset."""
    datasets = sorted(full["dataset"].dropna().unique())
    if not datasets:
        return
    full = full.copy()
    full["resid"] = full[f"{target}_pred"] - full[target]
    full["bin"] = (full["step_norm"] * 20).round() / 20.0  # 20 bins over t/T
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(4.2 * len(datasets), 3.2), squeeze=False, sharey=True
    )
    for ax, ds in zip(axes[0], datasets):
        cell = full[full["dataset"] == ds]
        for eps, ce in cell.groupby("eps"):
            agg = ce.groupby("bin")["resid"].agg(["mean", "std"]).reset_index()
            ax.plot(agg["bin"], agg["mean"], lw=1.3, label=f"ε={eps:g}")
            ax.fill_between(
                agg["bin"], agg["mean"] - agg["std"], agg["mean"] + agg["std"], alpha=0.15, lw=0
            )
        ax.axhline(0.0, color="black", lw=0.6, alpha=0.6)
        ax.set_title(ds)
        ax.set_xlabel("t / T")
        ax.grid(True, alpha=0.3, lw=0.5)
    axes[0, 0].set_ylabel(f"residual ({target}_pred − {target})")  # noqa: RUF001
    axes[0, -1].legend(fontsize=8)
    fig.suptitle(f"{target}: residual structure vs training progress", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, stem)


def plot_privacy(priv: pd.DataFrame, stem: Path) -> None:
    """Budget ratio (E_pred / target μ) and projection distance distributions."""
    if priv.empty:
        return
    priv = priv.copy()
    priv["group"] = priv["dataset"] + "\nε=" + priv["eps"].map(lambda e: f"{e:g}")
    groups = sorted(priv["group"].unique())
    has_proj = "projection_distance" in priv.columns
    fig, axes = plt.subplots(
        2 if has_proj else 1,
        1,
        figsize=(max(6, 1.1 * len(groups)), 6 if has_proj else 3.2),
        squeeze=False,
    )

    ax = axes[0, 0]
    box = [
        priv[priv["group"] == grp]["budget_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        for grp in groups
    ]
    ax.boxplot(box, tick_labels=groups, showfliers=False)
    ax.axhline(1.0, color="#d62728", lw=1.0, label="on budget (=1)")
    ax.set_ylabel("budget ratio  E_pred / μ")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3, lw=0.5)
    ax.set_title("Privacy-budget fidelity (>1 ⇒ over budget)", fontsize=10)

    if has_proj:
        ax2 = axes[1, 0]
        box2 = [
            priv[priv["group"] == grp]["projection_distance"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            for grp in groups
        ]
        ax2.boxplot(box2, tick_labels=groups, showfliers=False)
        ax2.set_ylabel("relative projection distance")
        ax2.grid(True, axis="y", alpha=0.3, lw=0.5)
    for lab in axes[-1, 0].get_xticklabels():
        lab.set_rotation(30)
        lab.set_ha("right")
    fig.tight_layout()
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def summarise(metrics: pd.DataFrame, target: str) -> pd.DataFrame:
    """Mean ± std per (dataset, ε) for the headline metrics (flat runs out of R²)."""
    sub = metrics[metrics["target"] == target].copy()
    if sub.empty:
        return pd.DataFrame()

    def fmt(s: pd.Series) -> str:
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            return "—"
        m, sd = s.mean(), s.std()
        return f"{m:.3f}" if np.isnan(sd) else f"{m:.3f} ± {sd:.3f}"

    def agg(df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "n_runs": len(df),
                "NRMSE": fmt(df["nrmse"]),
                "RMSLE": fmt(df["rmsle"]),
                "R2": fmt(df.loc[~df["flat"], "r2"]),
                "n_flat": int(df["flat"].sum()),
            }
        )

    return sub.groupby(["dataset", "eps"]).apply(agg, include_groups=False).reset_index()


def write_table(table: pd.DataFrame, stem: Path) -> None:
    if table.empty:
        return
    table.to_csv(stem.with_suffix(".csv"), index=False)
    cols = list(table.columns)
    lines = ["\\begin{tabular}{" + "l" * len(cols) + "}", "\\toprule"]
    lines.append(" & ".join(c.replace("_", "\\_") for c in cols) + " \\\\")
    lines.append("\\midrule")
    for _, row in table.iterrows():
        cells = [str(row[c]).replace("±", "$\\pm$").replace("_", "\\_") for c in cols]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    stem.with_suffix(".tex").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    eval_dir: str
    """The pysr_eval dir written by symbolic_regression.py."""
    out_dir: str = ""
    """Output dir. Defaults to <eval_dir>."""
    complexity: int = -1
    """Evaluate the equation of this complexity for overlays/metrics. -1 = PySR-selected."""
    flat_rel_std: float = 0.05
    """Runs whose actual-schedule std/mean is below this are flagged and excluded from R²."""
    compute_projection: bool = True
    """Compute the (slower) per-run projection-distance privacy metric."""


def main(conf: EvalConfig) -> None:
    global _FLAT_REL_STD
    _FLAT_REL_STD = conf.flat_rel_std

    eval_dir = Path(conf.eval_dir)
    out_dir = Path(conf.out_dir) if conf.out_dir else eval_dir
    full = pd.read_parquet(eval_dir / "features_full.parquet")

    models = {t: tm for t in TARGETS if (tm := _load_target(eval_dir, t)) is not None}
    if not models:
        raise SystemExit(f"no fitted models found under {eval_dir} (expected <target>/model.pkl)")
    print(f"Loaded models: {sorted(models)} · {full['run_id'].nunique()} runs, {len(full)} rows")

    metrics, privacy = evaluate_runs(full, models, conf.complexity, conf.compute_projection)

    for t, tm in models.items():
        print(f"\n=== {t} ===")
        tout = out_dir / t
        plot_pareto(tm, full, tout / "pareto")
        plot_overlay_grid(full, t, tout / "overlay_grid")
        plot_spotlights(full, metrics, t, tout / "spotlights")
        plot_metric_distributions(metrics, t, tout / "metric_distributions")
        plot_residual_vs_step(full, t, tout / "residual_vs_step")
        metrics[metrics["target"] == t].to_csv(tout / "metrics_per_run.csv", index=False)
        write_table(summarise(metrics, t), tout / "summary_table")
        print("  ✓ metrics_per_run.csv / summary_table.csv,.tex")

    if not privacy.empty:
        print("\n=== privacy ===")
        pout = out_dir / "privacy"
        pout.mkdir(parents=True, exist_ok=True)
        plot_privacy(privacy, pout / "privacy_validity")
        privacy.to_csv(pout / "privacy_per_run.csv", index=False)
        print("  ✓ privacy_per_run.csv")
    else:
        print("\n[skip] privacy validity — needs both σ and clip equations")  # noqa: RUF001


if __name__ == "__main__":
    main(tyro.cli(EvalConfig))
