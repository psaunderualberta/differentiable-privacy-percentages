import json
import os
import pickle
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import tyro
from pysr import PySRRegressor, TemplateExpressionSpec

from sr_category import (
    build_category_map,
    build_constants_table,
    category_series,
    save_category_map,
    template_param_names,
)
from sr_identity import canonical_identity, derive_slug, identity_flags

# Fixed PySR run_id so a synthesis's run directory is reused across chained jobs.
_RUN_ID = "pysr_run"
_THIS_SCRIPT = Path(__file__).resolve()
_SR_RUN_STARTER = _THIS_SCRIPT.parent.parent / "cc" / "slurm" / "sr-run-starter.py"

_NON_FEATURE_COLS: tuple[str, ...] = (
    "run_id",
    "run_name",
    "dataset",
    "arch_label",
    "optimizer",
    "seed",
    "axis",
)


@dataclass
class PySRConfig:
    cache_dir: str
    """Path to the <entity>__<project> directory produced by compile_results_fetch.py."""
    targets: tuple[Literal["sigma", "clip", "mu"], ...] = ("sigma",)
    datapoint_frequency: int = (
        100  # Frequency to use inner step datapoints in regression (i.e. every 100 steps)
    )
    datasets: tuple[str, ...] = ()
    arch_labels: tuple[str, ...] = ()
    optimizers: tuple[str, ...] = ()
    run_ids: tuple[str, ...] = ()
    keep_features: tuple[str, ...] = ()
    include_nonfinite_schedules: bool = False
    include_diverged_training: bool = False
    out_dir: str = ""
    """Base for persisted models + eval inputs; the synthesis-group slug is appended.
    Defaults to <cache_dir>/pysr_eval/, so artefacts land in <cache_dir>/pysr_eval/<slug>/."""
    niterations: int = 100_000
    """PySR search iterations. Lower (e.g. 5) for quick smoke tests."""
    maxsize: int = 25
    """Max equation complexity PySR will consider."""

    # --- Template mode (per-condition free constants; see docs/adr/0006) ---
    template_mode: bool = True
    """Fit one schedule shape f(step_norm) shared across runs plus K per-condition
    constants (the default). Pass --no-template_mode for the pooled scalar fit."""
    n_template_params: int = 3
    """K — number of free per-condition constants the schedule shape is modulated by."""

    # --- SLURM job-chaining (see docs/adr/0002) ---
    procs: int = 0
    """PySR worker processes. 0 ⇒ read $SLURM_NTASKS on SLURM, else cpu_count."""
    timeout_in_seconds: int = 9900
    """Per-job PySR search budget (2h45m), set below the ~2h55m SLURM wall time."""
    pad_seconds: int = 600
    """Slack (10m) below timeout: fit() finishing earlier counts as natural completion."""
    max_chain_jobs: int = 16
    """Hard cap on chain depth; the chain stops resubmitting once reached."""


def build_template_spec(n_conditions: int, n_template_params: int) -> TemplateExpressionSpec:
    """A ``TemplateExpressionSpec`` for the universal schedule shape ``f``.

    ``f``'s only real input is ``step_norm``; the K per-condition constants are
    indexed by the 1-indexed ``category`` column and passed in as extra arguments
    so PySR discovers how the shape is modulated per condition (ADR 0006). Each of
    the K parameter slots holds one constant per condition.
    """
    names = template_param_names(n_template_params)
    const_args = ", ".join(f"{name}[category]" for name in names)
    return TemplateExpressionSpec(
        expressions=["f"],
        variable_names=["step_norm", "category"],
        parameters=dict.fromkeys(names, n_conditions),
        combine=f"f(step_norm, {const_args})",
    )


def extract_template_constants(
    equation_row, param_names: tuple[str, ...] | list[str]
) -> dict[str, np.ndarray]:
    """Read the fitted per-condition constants off one template equation row.

    ``equation_row`` is a row of ``model.equations_`` (or ``model.get_best()``)
    whose ``"julia_expression"`` carries ``.metadata.parameters`` — a Julia
    ``NamedTuple`` with one field per declared template parameter. Each field is
    a length-``n_conditions`` vector. Returns ``{name: float64 ndarray}`` in the
    order of ``param_names``.

    Fields are read by attribute (``getattr``); the NamedTuple has no string
    ``getindex`` (``params["p1"]`` raises a Julia ``MethodError``). Values come
    back float32 and are cast to float64. See docs/adr/0006.
    """
    params = equation_row["julia_expression"].metadata.parameters
    return {name: np.asarray(getattr(params, name), dtype=np.float64) for name in param_names}


def _validate_targets(targets: tuple[str, ...]) -> None:
    if not 1 <= len(targets) <= 3:
        raise ValueError(f"targets must have length 1–3, got {len(targets)}: {targets}")
    if len(set(targets)) != len(targets):
        raise ValueError(f"targets must be unique, got {targets}")


def _apply_row_filters(df: pd.DataFrame, conf: PySRConfig) -> pd.DataFrame:
    if conf.datasets:
        df = df[df["dataset"].isin(conf.datasets)]
    if conf.arch_labels:
        dfs = []
        for label in conf.arch_labels:
            dfs.append(df[df["arch_label"].apply(lambda lbl, label=label: label in lbl)])
        df = pd.concat(dfs, axis=0)
    if conf.optimizers:
        df = df[df["optimizer"].isin(conf.optimizers)]
    if conf.run_ids:
        df = df[df["run_id"].isin(conf.run_ids)]
    return df


def _runs_with_finite_schedules(schedules: pd.DataFrame) -> set[str]:
    finite_mask = np.isfinite(schedules["sigma"]) & np.isfinite(schedules["clip"])
    bad_runs = set(schedules.loc[~finite_mask, "run_id"].unique())
    return set(schedules["run_id"].unique()) - bad_runs


def _runs_with_finite_learned(scalars: pd.DataFrame) -> set[str]:
    learned = scalars[scalars["schedule"] == "Learned Schedule"]
    finite_mask = np.isfinite(learned["mean_acc"]) & np.isfinite(learned["mean_loss"])
    return set(learned.loc[finite_mask, "run_id"].unique())


def _filter_features(df: pd.DataFrame, conf: PySRConfig, target: str) -> pd.DataFrame:
    df = df[~(df[target].isna() | np.isinf(df[target]))]
    df = df.dropna(axis=1)

    # Drop constant columns (excluding the target)
    non_target = [c for c in df.columns if c != target]
    keep_mask = (df[non_target] != df[non_target].iloc[0]).any()
    df = df[[target, *keep_mask[keep_mask].index]]

    if conf.keep_features:
        df = df[[target, *(c for c in conf.keep_features if c != target)]]
    return df


def should_resubmit(
    elapsed_seconds: float,
    timeout_seconds: float,
    pad_seconds: float,
    chain_depth: int,
    max_chain_jobs: int,
) -> bool:
    """Decide whether a finished job should resubmit a chain successor.

    Resubmit only when the job hit its PySR timeout (i.e. did NOT naturally
    complete: ``elapsed >= timeout - pad``) AND the chain has not reached its
    depth cap. A naturally completed synthesis never resubmits.
    """
    naturally_completed = elapsed_seconds < timeout_seconds - pad_seconds
    if naturally_completed:
        return False
    return chain_depth < max_chain_jobs


def _resolve_procs(conf: PySRConfig) -> int:
    """Worker-process count: explicit config, else $SLURM_NTASKS, else cpu_count."""
    if conf.procs > 0:
        return conf.procs
    ntasks = os.environ.get("SLURM_NTASKS")
    if ntasks:
        return int(ntasks)
    return 0


def run_regression(
    df: pd.DataFrame,
    target_col: str,
    conf: PySRConfig,
    output_directory: Path,
    procs: int,
    expression_spec: TemplateExpressionSpec | None = None,
) -> tuple[PySRRegressor, float]:
    """Fit (or resume) one synthesis. Returns the model and the ``fit()`` wall time.

    The run directory ``output_directory/_RUN_ID`` is pinned so chained jobs reuse
    it. If it already holds PySR state we resume via ``from_file`` + ``warm_start``;
    otherwise we start a fresh search writing into that fixed location.

    When ``expression_spec`` is given (template mode) the fresh search fits that
    spec — the universal shape ``f`` plus per-condition constants — instead of a
    pooled scalar equation.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    runtime_kwargs = {
        "timeout_in_seconds": conf.timeout_in_seconds,
        "niterations": conf.niterations,
    }

    if procs > 0:
        runtime_kwargs |= {
            "parallelism": "multiprocessing",
            "procs": procs,
            "populations": 3 * procs,
        }

    on_slurm = "SLURM_NTASKS" in os.environ
    if on_slurm:
        runtime_kwargs |= {
            "cluster_manager": "slurm",
        }

    run_directory = output_directory / _RUN_ID
    hall_of_fame = run_directory / "hall_of_fame.csv"
    if hall_of_fame.exists():  # created by previous run, not from anything else in this script
        print(f"  resuming synthesis from {run_directory}")
        # 1.5.10 forwards **kwargs straight to set_params (no nested pysr_kwargs arg).
        model = PySRRegressor.from_file(
            run_directory=str(run_directory),
            warm_start=True,
            **runtime_kwargs,
        )
    else:
        builder_kwargs: dict = {
            "output_directory": str(output_directory),
            "run_id": _RUN_ID,
            "batching": True,
            "parsimony": 1e-3,
            "maxsize": conf.maxsize,
            "binary_operators": ["*", "/", "+", "-"],
            "unary_operators": ["sqrt", "exp", "log"],
        }
        if expression_spec is not None:
            builder_kwargs["expression_spec"] = expression_spec
        builder_kwargs["elementwise_loss"] = (
            "loss(prediction, target) = ((prediction - target) / target)^2"
        )
        model = PySRRegressor(**builder_kwargs, **runtime_kwargs)

    start = time.monotonic()
    model.fit(X, y, variable_names=list(X.columns))
    elapsed = time.monotonic() - start
    return model, elapsed


def _resubmit_chain(conf: PySRConfig, target: str) -> None:
    """Resubmit the next job in this synthesis's chain via sr-run-starter.py.

    Reads chain context from ``CHAIN_*`` env vars injected by sr-run-starter.py and
    re-emits the synthesis identity from ``conf`` so the successor stays on the same slug
    directory; no-ops (with a warning) when run outside that launcher.
    """
    resubmit_script = os.environ.get("CHAIN_RESUBMIT_SCRIPT")
    if resubmit_script is None:
        print("WARNING: CHAIN_RESUBMIT_SCRIPT not set — synthesis chain ends here.")
        return

    depth = int(os.environ.get("CHAIN_DEPTH", "0"))
    prereqs: list[str] = []
    if "SLURM_JOB_ID" in os.environ:
        prereqs = ["--prerequisites", os.environ["SLURM_JOB_ID"]]

    # Re-emit the synthesis identity from conf so the successor lands on the same slug
    # directory (and warm-starts the same PySR state) — no identity is threaded through
    # CHAIN_* env vars. See docs/adr/0005.
    cmd = [
        "uv",
        "run",
        resubmit_script,
        "--cache_dir",
        os.environ.get("CHAIN_CACHE_DIR", ""),
        "--targets",
        target,
        "--chain-depth",
        str(depth + 1),
        "--max-chain-jobs",
        os.environ.get("CHAIN_MAX_JOBS", "16"),
        "--ntasks",
        os.environ.get("CHAIN_NTASKS", "32"),
        "--account",
        os.environ.get("CHAIN_ACCOUNT", ""),
        "--jobname",
        os.environ.get("CHAIN_JOBNAME", f"sr-{target}"),
        *identity_flags(asdict(conf)),
        *prereqs,
    ]
    print(f"Resubmitting chain (depth {depth + 1}): {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: resubmit failed:\n{result.stderr}")
    else:
        print(f"Resubmit successful: {result.stdout.strip()}")


def _equations_table(model: PySRRegressor) -> pd.DataFrame:
    """Distil the PySR Pareto front to a human-readable, serialisable table.

    The evaluator reads this CSV's ``selected`` column to pick the default
    equation; ``_selected_index`` falls back to the last (highest-complexity)
    row when no mask is present.
    """
    eqs = model.equations_
    cols = [c for c in ("complexity", "loss", "score", "equation") if c in eqs.columns]
    table = eqs[cols].copy()
    # Mark which front row PySR would select by default (used in .predict()).
    selected = getattr(model, "selection_mask_", None)
    table["selected"] = False
    if selected is not None:
        table.loc[np.asarray(selected), "selected"] = True
    else:
        table.iloc[-1, table.columns.get_loc("selected")] = True
    return table.reset_index(drop=True)


def _persist_target(model: PySRRegressor, feature_names: list[str], target_dir: Path) -> None:
    """Write the per-target artefacts the evaluator loads (see symbolic_regression_eval.py).

    PySR already checkpoints its own state into the run directory during ``fit``,
    but the evaluator loads a self-contained, relocatable bundle keyed by target:
    ``model.pkl`` (the pickled regressor, carrying the whole Pareto front),
    ``equations.csv`` (the distilled front incl. the ``selected`` row), and
    ``feature_names.json`` (the exact columns the target was fit on).
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    with (target_dir / "model.pkl").open("wb") as f:
        pickle.dump(model, f)
    _equations_table(model).to_csv(target_dir / "equations.csv", index=False)
    (target_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    print(f"  → persisted {target_dir}")


def _write_manifest(
    out_dir: Path, conf: PySRConfig, keep_runs: set[str], full_df: pd.DataFrame
) -> None:
    manifest = {
        "config": asdict(conf),
        "targets": list(conf.targets),
        "datapoint_frequency": conf.datapoint_frequency,
        "n_runs": len(keep_runs),
        "n_rows_full": len(full_df),
        "run_ids": sorted(keep_runs),
        "datasets": sorted(full_df["dataset"].dropna().unique().tolist()),
        "arch_labels": sorted(full_df["arch_label"].dropna().unique().tolist()),
        "optimizers": sorted(full_df["optimizer"].dropna().unique().tolist()),
        "eps": sorted(full_df["eps"].dropna().unique().tolist()),
        "T": sorted(int(t) for t in full_df["T"].dropna().unique()),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main(conf: PySRConfig):
    _validate_targets(conf.targets)

    cache_dir = Path(conf.cache_dir)
    schedules = pd.read_parquet(cache_dir / "schedules.parquet")
    scalars = pd.read_parquet(cache_dir / "scalars.parquet")

    schedules = _apply_row_filters(schedules, conf)
    scalars = _apply_row_filters(scalars, conf)
    schedules["mu"] = schedules["clip"] / schedules["sigma"]

    keep_runs = set(schedules["run_id"].unique())
    if not conf.include_nonfinite_schedules:
        keep_runs &= _runs_with_finite_schedules(schedules)
    if not conf.include_diverged_training:
        keep_runs &= _runs_with_finite_learned(scalars)

    schedules = schedules[schedules["run_id"].isin(keep_runs)].reset_index(drop=True)
    if schedules.empty:
        raise ValueError("No rows remain after filtering.")

    print(f"Regressing on {len(keep_runs)} runs ({len(schedules)} rows)")

    # This synthesis group's identity slug disambiguates it from other filterings of the
    # same sweep (datasets, arch, ...). It is one path segment above the per-target dir on
    # both the scratch run directory and the persistent mirror. See docs/adr/0005.
    slug = derive_slug(canonical_identity(asdict(conf)))

    # Persist the inputs the evaluator needs to reconstruct predicted schedules:
    # the full-resolution (every inner step) per-run feature+actual table, plus a
    # manifest recording exactly which runs/filters produced these models. The slug dir is
    # a self-contained synthesis-group artifact: point the evaluator's --eval-dir here.
    out_base = Path(conf.out_dir) if conf.out_dir else cache_dir / "pysr_eval"
    out_dir = out_base / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    schedules.to_parquet(out_dir / "features_full.parquet", index=False)
    _write_manifest(out_dir, conf, keep_runs, schedules)

    sampled = schedules[schedules["inner_step"] % conf.datapoint_frequency == 0].copy()

    # Template mode indexes per-condition constants by a 1-indexed `category`. Build
    # and persist that map (so the evaluator rebuilds the same column) BEFORE dropping
    # the condition columns dataset/arch_label below. See docs/adr/0006.
    category_map = None
    if conf.template_mode:
        category_map = build_category_map(sampled)
        save_category_map(category_map, out_dir / "category_map.json")
        sampled["category"] = category_series(sampled, category_map)
        print(f"Template mode: {len(category_map)} conditions, {conf.n_template_params} constants")

    feature_df = sampled.drop(columns=list(_NON_FEATURE_COLS), errors="ignore")

    for target in conf.targets:
        target_out_dir = out_dir / target

        procs = _resolve_procs(conf)
        print(f"=== {target} regression ({procs} procs) ===")

        if conf.template_mode:
            target_df = feature_df[[target, "step_norm", "category"]].copy()
            target_df = target_df[
                ~(target_df[target].isna() | np.isinf(target_df[target]))
            ].reset_index(drop=True)
            feature_names = [c for c in target_df.columns if c != target]
            spec = build_template_spec(len(category_map), conf.n_template_params)
            model, elapsed = run_regression(
                target_df, target, conf, target_out_dir, procs, expression_spec=spec
            )
            names = template_param_names(conf.n_template_params)
            constants = extract_template_constants(model.get_best(), names)
            build_constants_table(constants, category_map).to_csv(
                target_out_dir / "constants.csv", index=False
            )
            print(f"  ✓ constants.csv ({len(category_map)} conditions × {len(names)} params)")
        else:
            others = {"sigma", "clip", "mu"} - {target}
            target_df = feature_df.drop(columns=[*others], errors="ignore").copy()
            target_df = _filter_features(target_df, conf, target)
            feature_names = [c for c in target_df.columns if c != target]
            print(f"=== Features: {feature_names} ===")
            model, elapsed = run_regression(target_df, target, conf, target_out_dir, procs)

        print(model)
        print(f"  fit() took {elapsed:.0f}s (timeout {conf.timeout_in_seconds}s)")
        _persist_target(model, feature_names, target_out_dir)

        depth = int(os.environ.get("CHAIN_DEPTH", "0"))
        if should_resubmit(
            elapsed, conf.timeout_in_seconds, conf.pad_seconds, depth, conf.max_chain_jobs
        ):
            _resubmit_chain(conf, target)
        else:
            print("  synthesis complete or chain cap reached — not resubmitting.")


if __name__ == "__main__":
    main(tyro.cli(PySRConfig))
