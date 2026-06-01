import json
import multiprocessing  # used to get cpu count (local fallback only)
import os
import pickle
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import tyro
from pysr import PySRRegressor

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
    """Where to persist fitted models + eval inputs. Defaults to <cache_dir>/pysr_eval/."""
    niterations: int = 100_000
    """PySR search iterations. Lower (e.g. 5) for quick smoke tests."""
    maxsize: int = 25
    """Max equation complexity PySR will consider."""

    # --- SLURM job-chaining (see docs/adr/0002) ---
    scratch_dir: str = "/scratch/$USER/pysr"
    """Base dir for the live run directory (fast shared tier). Per-target subdir
    is appended; env vars like $USER are expanded. NEVER use $SLURM_TMPDIR."""
    procs: int = 0
    """PySR worker processes. 0 ⇒ read $SLURM_NTASKS on SLURM, else cpu_count."""
    timeout_in_seconds: int = 9900
    """Per-job PySR search budget (2h45m), set below the ~2h55m SLURM wall time."""
    pad_seconds: int = 600
    """Slack (10m) below timeout: fit() finishing earlier counts as natural completion."""
    max_chain_jobs: int = 16
    """Hard cap on chain depth; the chain stops resubmitting once reached."""
    mirror_sync_secs: int = 900
    """Interval (15m) for the background rsync of run directory → persistent mirror."""


def _validate_targets(targets: tuple[str, ...]) -> None:
    if not 1 <= len(targets) <= 3:
        raise ValueError(f"targets must have length 1–3, got {len(targets)}: {targets}")  # noqa: RUF001
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


def should_restore_mirror(scratch_run_dir_exists: bool, mirror_exists: bool) -> bool:
    """Restore the persistent mirror onto scratch only if scratch was purged.

    True iff the scratch run directory is gone but a mirror exists — the case
    where a chained job landed after /scratch was wiped. If scratch survived
    (back-to-back jobs) we resume in place; with no mirror we start fresh.
    """
    return (not scratch_run_dir_exists) and mirror_exists


def _resolve_procs(conf: PySRConfig) -> int:
    """Worker-process count: explicit config, else $SLURM_NTASKS, else cpu_count."""
    if conf.procs > 0:
        return conf.procs
    ntasks = os.environ.get("SLURM_NTASKS")
    if ntasks:
        return int(ntasks)
    return multiprocessing.cpu_count()


def _rsync(src: Path, dst: Path) -> None:
    """Mirror ``src`` directory into ``dst`` (durable storage), deleting stale files."""
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rsync", "-a", "--delete", f"{src}/", f"{dst}/"],
        check=False,
    )


def _start_mirror_daemon(
    run_directory: Path, mirror: Path, interval_secs: int
) -> tuple[threading.Event, threading.Thread]:
    """Background daemon: rsync run directory → mirror every ``interval_secs``.

    Returns a (stop_event, thread) pair; set the event then call ``_rsync`` once
    more for a final sync after ``fit()`` returns.
    """
    stop = threading.Event()

    def loop(stop_event: threading.Event = stop) -> None:
        while not stop_event.wait(interval_secs):
            _rsync(run_directory, mirror)

    thread = threading.Thread(target=loop, daemon=True, name="pysr-mirror")
    thread.start()
    return stop, thread


def run_regression(
    df: pd.DataFrame,
    target_col: str,
    conf: PySRConfig,
    output_directory: Path,
    procs: int,
) -> tuple[PySRRegressor, float]:
    """Fit (or resume) one synthesis. Returns the model and the ``fit()`` wall time.

    The run directory ``output_directory/_RUN_ID`` is pinned so chained jobs reuse
    it. If it already holds PySR state we resume via ``from_file`` + ``warm_start``;
    otherwise we start a fresh search writing into that fixed location.
    """
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    on_slurm = "SLURM_NTASKS" in os.environ
    populations = 3 * procs
    runtime_kwargs = {
        "parallelism": "multiprocessing",
        "procs": procs,
        "cluster_manager": "slurm" if on_slurm else None,
        "timeout_in_seconds": conf.timeout_in_seconds,
        "niterations": conf.niterations,
        "populations": populations,
        "warm_start": True,
    }

    run_directory = output_directory / _RUN_ID
    if run_directory.exists():
        print(f"  resuming synthesis from {run_directory}")
        # 1.5.10 forwards **kwargs straight to set_params (no nested pysr_kwargs arg).
        model = PySRRegressor.from_file(
            run_directory=str(run_directory),
            **runtime_kwargs,
        )
    else:
        model = PySRRegressor(
            output_directory=str(output_directory),
            run_id=_RUN_ID,
            batching=True,
            turbo=True,
            parsimony=1e-4,
            maxsize=conf.maxsize,
            binary_operators=["*", "/"],
            unary_operators=[
                "sqrt",
                "exp",
                "log",
            ],
            elementwise_loss="loss(prediction, target) = ((prediction - target) / target)^2",
            **runtime_kwargs,
        )

    start = time.monotonic()
    model.fit(X, y, variable_names=list(X.columns))
    elapsed = time.monotonic() - start
    return model, elapsed


def _fit_with_mirror(
    df: pd.DataFrame,
    target_col: str,
    conf: PySRConfig,
    output_directory: Path,
    run_directory: Path,
    mirror: Path,
    procs: int,
) -> tuple[PySRRegressor, float]:
    """Run ``run_regression`` with the background mirror daemon active.

    The final ``_rsync`` runs in a ``finally`` so the freshest checkpoint is
    mirrored even if ``fit()`` raises — though a crash still ends the chain (no
    successor is submitted; see docs/adr/0002).
    """
    stop_daemon, daemon = _start_mirror_daemon(run_directory, mirror, conf.mirror_sync_secs)
    try:
        return run_regression(df, target_col, conf, output_directory, procs)
    finally:
        stop_daemon.set()
        daemon.join(timeout=5)
        _rsync(run_directory, mirror)  # final sync of the freshest checkpoint


def _resubmit_chain(target: str) -> None:
    """Resubmit the next job in this synthesis's chain via sr-run-starter.py.

    Reads chain context from ``CHAIN_*`` env vars injected by sr-run-starter.py;
    no-ops (with a warning) when run outside that launcher.
    """
    resubmit_script = os.environ.get("CHAIN_RESUBMIT_SCRIPT")
    if resubmit_script is None:
        print("WARNING: CHAIN_RESUBMIT_SCRIPT not set — synthesis chain ends here.")
        return

    depth = int(os.environ.get("CHAIN_DEPTH", "0"))
    prereqs: list[str] = []
    if "SLURM_JOB_ID" in os.environ:
        prereqs = ["--prerequisites", os.environ["SLURM_JOB_ID"]]

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
        *prereqs,
    ]
    print(f"Resubmitting chain (depth {depth + 1}): {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: resubmit failed:\n{result.stderr}")
    else:
        print(f"Resubmit successful: {result.stdout.strip()}")


def _equations_table(model: PySRRegressor) -> pd.DataFrame:
    """Distil the PySR Pareto front to a human-readable, serialisable table."""
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

    # Persist the inputs the evaluator needs to reconstruct predicted schedules:
    # the full-resolution (every inner step) per-run feature+actual table, plus a
    # manifest recording exactly which runs/filters produced these models.
    out_dir = Path(conf.out_dir) if conf.out_dir else cache_dir / "pysr_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    schedules.to_parquet(out_dir / "features_full.parquet", index=False)
    _write_manifest(out_dir, conf, keep_runs, schedules)

    feature_df = schedules[schedules["inner_step"] % conf.datapoint_frequency == 0]
    feature_df = feature_df.drop(columns=list(_NON_FEATURE_COLS), errors="ignore")

    procs = _resolve_procs(conf)
    scratch_base = Path(os.path.expandvars(conf.scratch_dir))

    for target in conf.targets:
        others = {"sigma", "clip", "mu"} - {target}
        target_df = feature_df.drop(columns=[*others], errors="ignore").copy()
        target_df = _filter_features(target_df, conf, target)
        feature_names = [c for c in target_df.columns if c != target]

        # Live run dir on fast scratch; durable mirror under the persisted target dir.
        output_directory = scratch_base / target
        run_directory = output_directory / _RUN_ID
        mirror = out_dir / target / _RUN_ID

        if should_restore_mirror(run_directory.exists(), mirror.exists()):
            print(f"  /scratch purged — restoring mirror {mirror} → {run_directory}")
            _rsync(mirror, run_directory)

        print(f"=== {target} regression ({procs} procs) ===")
        print(f"=== Features: {feature_names} ===")

        model, elapsed = _fit_with_mirror(
            target_df, target, conf, output_directory, run_directory, mirror, procs
        )

        print(model)
        print(f"  fit() took {elapsed:.0f}s (timeout {conf.timeout_in_seconds}s)")
        _persist_target(model, feature_names, out_dir / target)

        depth = int(os.environ.get("CHAIN_DEPTH", "0"))
        if should_resubmit(
            elapsed, conf.timeout_in_seconds, conf.pad_seconds, depth, conf.max_chain_jobs
        ):
            _resubmit_chain(target)
        else:
            print("  synthesis complete or chain cap reached — not resubmitting.")


if __name__ == "__main__":
    main(tyro.cli(PySRConfig))
