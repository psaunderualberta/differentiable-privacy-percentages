import json
import multiprocessing  # used to get cpu count
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import tyro
from pysr import PySRRegressor

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


def run_regression(
    df: pd.DataFrame, target_col: str, niterations: int = 2000, maxsize: int = 20
) -> PySRRegressor:
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    model = PySRRegressor(
        batching=True,
        turbo=True,
        populations=3 * multiprocessing.cpu_count(),
        parsimony=1e-4,
        maxsize=maxsize,
        niterations=niterations,
        timeout_in_seconds=60 * 60,
        binary_operators=["*", "/"],
        unary_operators=[
            "sqrt",
            "exp",
            "log",
        ],
        elementwise_loss="loss(prediction, target) = ((prediction - target) / target)^2",
    )
    model.fit(X, y, variable_names=list(X.columns))
    return model


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

    for target in conf.targets:
        others = {"sigma", "clip", "mu"} - {target}
        target_df = feature_df.drop(columns=[*others], errors="ignore").copy()
        target_df = _filter_features(target_df, conf, target)
        feature_names = [c for c in target_df.columns if c != target]

        print(f"=== {target} regression ===")
        print(f"=== Features: {feature_names} ===")
        model = run_regression(target_df, target, conf.niterations, conf.maxsize)
        print(model)
        _persist_target(model, feature_names, out_dir / target)


if __name__ == "__main__":
    main(tyro.cli(PySRConfig))
