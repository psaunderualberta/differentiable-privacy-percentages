"""Curve-transfer producer (ADR 0008).

For one source × target cell: resample a source run's raw length-T sigma/clip
schedule onto the target T, seat the sigma shape on the target's DP-PSAC budget,
carry the (privacy-neutral) clip curve across, and run the shared eval core.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from jax import random as jr
from jaxtyping import Array

from privacy.gdp_privacy import GDPPrivacyParameters
from util.transfer import (
    RawArraySchedule,
    SourcePolicy,
    TargetSpec,
    build_target_config,
    seat_on_budget,
    transfer_rows,
    write_transfer_cell,
)


def load_source_policies(
    schedules_parquet: Path | str,
) -> list[tuple[SourcePolicy, np.ndarray, np.ndarray]]:
    """Read ``schedules.parquet`` into per-run ``(SourcePolicy, sigmas, clips)``.

    One record per source ``run_id``; the length-T sigma/clip curves are ordered by
    ``inner_step``. ``delta`` and ``p`` are source-side provenance that the parquet
    does not carry (only the target's budget matters for seating), so they default
    to NaN.
    """
    df = pd.read_parquet(schedules_parquet)
    records: list[tuple[SourcePolicy, np.ndarray, np.ndarray]] = []
    for run_id, group in df.groupby("run_id", sort=True):
        group = group.sort_values("inner_step")
        first = group.iloc[0]
        source = SourcePolicy(
            run_id=str(run_id),
            dataset=str(first["dataset"]),
            eps=float(first["eps"]),
            delta=float("nan"),
            T=int(first["T"]),
            p=float("nan"),
            arch=str(first["arch_label"]),
        )
        sigmas = group["sigma"].to_numpy(dtype=float)
        clips = group["clip"].to_numpy(dtype=float)
        records.append((source, sigmas, clips))
    return records


def schedule_data_to_results(df: pd.DataFrame) -> list[tuple[int, float, float]]:
    """Adapt ``Baseline.generate_schedule_data`` rows to ``(seed, accuracy, loss)``.

    Each row is one evaluation rep; ``Baseline`` derives that rep's PRNG key
    sequentially, so the rep index is the cell's per-seed identifier.
    """
    return [
        (i, float(acc), float(loss))
        for i, (acc, loss) in enumerate(zip(df["accuracy"], df["loss"]))
    ]


def build_curve_schedule(
    source_sigmas: Array,
    source_clips: Array,
    privacy_params: GDPPrivacyParameters,
) -> RawArraySchedule:
    """Resample a source curve onto the target T and seat it on the target budget.

    The sigma shape is resampled to the target T and scaled so it *binds* the
    target's DP-PSAC boundary (``seat_on_budget``) — its magnitude is set entirely
    by the target ε. The clip curve is a privacy-neutral per-step LR multiplier, so
    it is resampled and carried across untouched.
    """
    target_T = int(privacy_params.T)
    sigmas = seat_on_budget(resample_curve(source_sigmas, target_T), privacy_params)
    clips = resample_curve(source_clips, target_T)
    return RawArraySchedule(sigmas, clips)


def resample_curve(values: Array, target_T: int) -> np.ndarray:
    """Resample a length-T curve onto ``target_T`` points, endpoint-preserving.

    Linear interpolation over normalized position ``linspace(0, 1, T)``: a pure
    shape-preserving reshape that keeps the first and last learned step and never
    extrapolates a fabricated tail (unlike the ``i/T`` grid, whose last source
    point sits below 1 and would force interp to clamp the target tail).
    """
    values = np.asarray(values, dtype=float)
    src_pos = np.linspace(0.0, 1.0, len(values))
    tgt_pos = np.linspace(0.0, 1.0, target_T)
    return np.interp(tgt_pos, src_pos, values)


# ---------------------------------------------------------------------------
# Cell orchestration + CLI (integration glue; exercised end-to-end, not unit-tested)
# ---------------------------------------------------------------------------


def run_curve_cell(
    source: SourcePolicy,
    source_sigmas: Array,
    source_clips: Array,
    target: TargetSpec,
    cache_root: Path | str,
    batch_size: int = 250,
    num_reps: int = 8,
    seed: int = 0,
) -> Path:
    """Transfer one source policy onto the target and write its per-seed cell.

    Seats the resampled source curve on the target budget, evaluates it natively on
    the target for ``num_reps`` seeds via the shared eval core, and writes the
    ``producer="curve"`` parquet cell.
    """
    from conf.scope import RunContext, using
    from conf.singleton_conf import SingletonConfig
    from environments.dp_params import DPTrainingParams
    from privacy.gdp_privacy import get_privacy_params
    from util.baselines import Baseline
    from util.dataloaders import get_dataset_shapes

    config = build_target_config(target, batch_size)
    # The inner DP-SGD path also reads the singleton / RunContext, so the whole
    # eval (not just param construction) must stay inside both scopes — otherwise
    # a training-time singleton read finds it reset and re-parses sys.argv.
    with SingletonConfig.override(config), using(RunContext(config)):
        X_shape, *_ = get_dataset_shapes()
        gdp_params = get_privacy_params(X_shape[0])
        env_params = DPTrainingParams.create_direct_from_config()

        schedule = build_curve_schedule(source_sigmas, source_clips, gdp_params)
        baseline = Baseline(env_params, gdp_params, jr.PRNGKey(seed), num_reps=num_reps)
        df = baseline.generate_schedule_data(schedule, name=f"Curve Transfer ({source.run_id})")

    rows = transfer_rows("curve", source, target, schedule_data_to_results(df))
    return write_transfer_cell(rows, cache_root)


@dataclasses.dataclass
class CurveCellConfig:
    """One curve-transfer cell (one source policy × one target) per invocation."""

    schedules_parquet: str
    """Path to the source ``schedules.parquet`` from compile_results_fetch."""
    target: str
    """Target dataset name (eyepacs, imagenet, chexpert, ...)."""
    target_eps: float
    target_T: int
    target_delta: float = 1e-7
    target_arch: str = ""
    """Arch label recorded on the cell rows; the arch itself is auto-derived from the dataset."""
    batch_size: int = 250
    cache_root: str = "cache"
    num_reps: int = 8
    seed: int = 0
    source_run_id: str = ""
    """If set, transfer only this source run; otherwise every source policy in the parquet."""


def main(conf: CurveCellConfig) -> None:
    target = TargetSpec(
        name=conf.target,
        eps=conf.target_eps,
        delta=conf.target_delta,
        T=conf.target_T,
        arch=conf.target_arch,
    )
    records = load_source_policies(conf.schedules_parquet)
    if conf.source_run_id:
        records = [r for r in records if r[0].run_id == conf.source_run_id]
        if not records:
            raise SystemExit(
                f"source_run_id {conf.source_run_id!r} not in {conf.schedules_parquet}"
            )

    for source, sigmas, clips in records:
        out = run_curve_cell(
            source,
            sigmas,
            clips,
            target,
            cache_root=conf.cache_root,
            batch_size=conf.batch_size,
            num_reps=conf.num_reps,
            seed=conf.seed,
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main(tyro.cli(CurveCellConfig))
