from dataclasses import dataclass
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
    "inner_step",
)


@dataclass
class PySRConfig:
    cache_dir: str
    """Path to the <entity>__<project> directory produced by compile_results_fetch.py."""
    targets: tuple[Literal["sigma", "clip"], ...] = ("sigma",)
    datasets: tuple[str, ...] = ()
    arch_labels: tuple[str, ...] = ()
    optimizers: tuple[str, ...] = ()
    run_ids: tuple[str, ...] = ()
    keep_features: tuple[str, ...] = ()
    include_nonfinite_schedules: bool = False
    include_diverged_training: bool = False


def _validate_targets(targets: tuple[str, ...]) -> None:
    if not 1 <= len(targets) <= 2:
        raise ValueError(f"targets must have length 1 or 2, got {len(targets)}: {targets}")
    if len(set(targets)) != len(targets):
        raise ValueError(f"targets must be unique, got {targets}")


def _apply_row_filters(df: pd.DataFrame, conf: PySRConfig) -> pd.DataFrame:
    if conf.datasets:
        df = df[df["dataset"].isin(conf.datasets)]
    if conf.arch_labels:
        df = df[df["arch_label"].isin(conf.arch_labels)]
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


def run_regression(df: pd.DataFrame, target_col: str) -> PySRRegressor:
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    model = PySRRegressor(
        batching=True,
        maxsize=30,
        niterations=2000,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[
            "sqrt",
            "exp",
            "log",
            "inv(x) = 1/x",
        ],
        extra_sympy_mappings={
            "inv": lambda x: 1 / x,
        },
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    )
    model.fit(X, y, variable_names=list(X.columns))
    return model


def main(conf: PySRConfig):
    _validate_targets(conf.targets)

    cache_dir = Path(conf.cache_dir)
    schedules = pd.read_parquet(cache_dir / "schedules.parquet")
    scalars = pd.read_parquet(cache_dir / "scalars.parquet")

    schedules = _apply_row_filters(schedules, conf)
    scalars = _apply_row_filters(scalars, conf)

    keep_runs = set(schedules["run_id"].unique())
    if not conf.include_nonfinite_schedules:
        keep_runs &= _runs_with_finite_schedules(schedules)
    if not conf.include_diverged_training:
        keep_runs &= _runs_with_finite_learned(scalars)

    schedules = schedules[schedules["run_id"].isin(keep_runs)].reset_index(drop=True)
    if schedules.empty:
        raise ValueError("No rows remain after filtering.")

    print(f"Regressing on {len(keep_runs)} runs ({len(schedules)} rows)")

    feature_df = schedules.drop(columns=list(_NON_FEATURE_COLS), errors="ignore")

    for target in conf.targets:
        other = "clip" if target == "sigma" else "sigma"
        target_df = feature_df.drop(columns=[other], errors="ignore").copy()
        target_df = _filter_features(target_df, conf, target)

        print(f"=== {target} regression ===")
        print(f"=== Features: {[c for c in target_df.columns if c != target]} ===")
        model = run_regression(target_df, target)
        print(model)


if __name__ == "__main__":
    main(tyro.cli(PySRConfig))
