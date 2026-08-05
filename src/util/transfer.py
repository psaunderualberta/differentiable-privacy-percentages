"""Shared core for the policy-transfer evaluation (ADR 0008).

The curve and equation producers both feed a length-T sigma/clip schedule through
the same seating + eval core (``util/baselines.py``'s ``generate_schedule_data``).
``RawArraySchedule`` is the lossless wrapper that carries an arbitrary schedule into
that core; ``seat_on_budget`` binds a source sigma curve onto the target's DP-PSAC
budget before it is evaluated.
"""

import dataclasses
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Self

import jax.numpy as jnp
import numpy as np
import optimistix as optx
import pandas as pd
from jaxtyping import Array

from policy.schedules.abstract import AbstractNoiseAndClipSchedule
from privacy.gdp_privacy import GDPPrivacyParameters


@dataclasses.dataclass(frozen=True)
class SourcePolicy:
    """A learned source run's schedule and the regime it was trained in.

    The row unit of the transfer matrix (CONTEXT.md), identified by ``run_id``.
    """

    run_id: str
    dataset: str
    eps: float
    delta: float
    T: int
    p: float
    arch: str
    arm: str = ""
    """Which momentum arm the policy was learned in (``sgd-m0.9`` / ``sgd-m0.0``).

    Part of the source regime's identity (ADR 0018): both arms are transferred and
    their shapes differ enough that pooling them would report arm separation as
    generalization consistency. ``""`` means "no arm" — a native reference is not
    learned in one at all — and is an empty *string* rather than NaN because the
    arm is a grouping/join key, and NaN never compares equal to itself.
    """


@dataclasses.dataclass(frozen=True)
class TargetSpec:
    """A transfer target dataset and the privacy budget it is evaluated under."""

    name: str
    eps: float
    delta: float
    T: int
    arch: str


# Exact parquet schema (ADR 0008): one row per (cell, seed).
_TRANSFER_COLUMNS = [
    "producer",
    "source_id",
    "source_dataset",
    "source_eps",
    "source_delta",
    "source_T",
    "source_p",
    "source_arch",
    "source_arm",
    "target",
    "target_eps",
    "target_delta",
    "target_T",
    "target_arch",
    "seed",
    "accuracy",
    "loss",
]


def transfer_rows(
    producer: str,
    source: SourcePolicy,
    target: TargetSpec,
    results: Iterable[tuple[int, float, float]],
) -> pd.DataFrame:
    """Build the per-seed transfer rows for one source×target cell.

    ``results`` is an iterable of ``(seed, accuracy, loss)`` — the per-seed output
    of the eval core. Source/target metadata is broadcast across the seed rows.
    """
    rows = [
        {
            "producer": producer,
            "source_id": source.run_id,
            "source_dataset": source.dataset,
            "source_eps": source.eps,
            "source_delta": source.delta,
            "source_T": source.T,
            "source_p": source.p,
            "source_arch": source.arch,
            "source_arm": source.arm,
            "target": target.name,
            "target_eps": target.eps,
            "target_delta": target.delta,
            "target_T": target.T,
            "target_arch": target.arch,
            "seed": seed,
            "accuracy": accuracy,
            "loss": loss,
        }
        for seed, accuracy, loss in results
    ]
    return pd.DataFrame(rows, columns=_TRANSFER_COLUMNS)


# Assembling into a stable order: a cell's rows are identified by these keys, so
# sorting on them makes the concatenated matrix independent of filesystem glob order.
_ASSEMBLE_SORT_KEYS = ["source_id", "target", "target_eps", "target_T", "seed"]


def write_transfer_cell(df: pd.DataFrame, cache_root: Path | str) -> Path:
    """Write one cell's rows to ``<cache_root>/transfer/<producer>/<cell>.parquet``.

    The cell filename embeds the (source_id, target, target_eps, target_T) key so
    each SLURM cell owns a distinct file; the assembler later globs them together.
    The name comes from ``transfer_launch.cell_filename`` — the launcher's skip
    filter looks for exactly this path, so the two must not drift apart.

    The write is atomic (temp file in the same directory + ``os.replace``): array
    tasks are killed at the wall clock, and a half-written parquet would be
    indistinguishable from a finished cell to both the assembler and the skip filter.
    """
    from transfer_launch import Target, cell_filename

    producer = df["producer"].iloc[0]
    src_id = str(df["source_id"].iloc[0])
    target = Target(
        dataset=str(df["target"].iloc[0]),
        eps=float(df["target_eps"].iloc[0]),
        T=int(df["target_T"].iloc[0]),
    )

    out_dir = Path(cache_root) / "transfer" / str(producer)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cell_filename(src_id, target)

    with tempfile.NamedTemporaryFile(dir=out_dir, suffix=".parquet", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return out_path


def _candidate_dir(cache_root: Path | str) -> Path:
    from transfer_launch import CANDIDATE_DIR

    return Path(cache_root) / CANDIDATE_DIR


def write_candidate_record(
    reference: str,
    target: "TargetSpec",
    candidate: int,
    mean_accuracy: float,
    n: int,
    cache_root: Path | str,
) -> Path:
    """Record one reference-sweep candidate's score (ADR 0019).

    An *intermediate* artifact, not a transfer cell: it lives outside
    ``<cache_root>/transfer/`` so the assembler cannot mistake nineteen
    deliberately under-evaluated candidates for nineteen extra reference columns.
    Only :func:`write_transfer_cell` produces matrix rows.

    Atomic, for the same reason cell writes are: an array task killed at the wall
    clock must not leave a half-written score that the skip filter reads as done.
    """
    from transfer_launch import Target, candidate_filename

    out_dir = _candidate_dir(cache_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / candidate_filename(
        reference, Target(dataset=target.name, eps=target.eps, T=target.T), candidate
    )
    payload = {
        "reference": reference,
        "target": target.name,
        "target_eps": float(target.eps),
        "target_T": int(target.T),
        "candidate": int(candidate),
        "mean_accuracy": float(mean_accuracy),
        "n": int(n),
    }

    with tempfile.NamedTemporaryFile(
        "w", dir=out_dir, suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(payload, tmp)
        tmp_path = Path(tmp.name)
    try:
        os.replace(tmp_path, out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return out_path


def read_candidate_records(
    reference: str, target: "TargetSpec", cache_root: Path | str
) -> list[dict]:
    """Every scored candidate for one (reference × target), in candidate order.

    The selector's input. Ordered by candidate index so the winner is a
    deterministic function of what is on disk, whatever order the tasks finished in.
    """
    from transfer_launch import Target, candidate_filename

    out_dir = _candidate_dir(cache_root)
    if not out_dir.is_dir():
        return []
    launch_target = Target(dataset=target.name, eps=target.eps, T=target.T)
    prefix = candidate_filename(reference, launch_target, 0).rsplit("__cand", 1)[0]
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(out_dir.glob(f"{prefix}__cand*.json"))
    ]
    return sorted(records, key=lambda record: record["candidate"])


def assemble_transfer(producer: str, cache_root: Path | str) -> pd.DataFrame:
    """Glob every cell parquet for ``producer`` and concat into one matrix frame.

    Deterministic regardless of write/glob order: rows are sorted on the cell keys.
    """
    cell_dir = Path(cache_root) / "transfer" / str(producer)
    frames = [pd.read_parquet(p) for p in sorted(cell_dir.glob("*.parquet"))]
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(_ASSEMBLE_SORT_KEYS).reset_index(drop=True)


def build_target_config(target: "TargetSpec", batch_size: int):
    """A minimal singleton Config pinned to the target regime.

    ``get_privacy_params`` / ``DPTrainingParams.create_direct_from_config`` /
    ``get_dataset_shapes`` all read the singleton, so a producer wraps this config
    around them to build the target's env/GDP params. The network arch is left to
    ``AutoNetworkConfig``, which derives the surrogate arch from the dataset
    (ADR 0007), matching the target's ``arch`` label.
    """
    from conf.config import Config, EnvConfig, ScheduleOptimizerConfig, SweepConfig, WandbConfig

    env = EnvConfig(
        eps=target.eps,
        delta=target.delta,
        batch_size=batch_size,
        num_training_steps=target.T,
    )
    sweep = SweepConfig(
        env=env,
        schedule_optimizer=ScheduleOptimizerConfig(),
        dataset=target.name,
    )
    return Config(wandb_conf=WandbConfig(), sweep=sweep)


def seat_on_budget(sigmas: Array, privacy_params: GDPPrivacyParameters) -> Array:
    """Scale a **noise-multiplier** curve onto the target DP-PSAC budget, then project.

    ``sigmas`` here is the per-step *multiplier* ``s = sigma_noise / clip``, the same
    unit ``project_inverse_sigmas`` takes — **not** the raw noise scale. The GDP
    budget is ``sum_i exp((C_i/sigma_i)^2) = sum_i exp(1/s_i^2)``, so callers holding
    raw sigmas must divide by their clips first and multiply the result back
    (see ``transfer_curve.build_curve_schedule``). Passing raw sigma silently
    substitutes ``C_i := 1`` and over-noises the curve by ~10x.

    ``project_inverse_sigmas`` enforces only the *inequality*
    ``sum_i exp(1/s_i^2) <= (mu/p)^2 + T`` — a feasible-but-slack (over-noised)
    source curve passes through untouched, under-spending the target budget. So we
    first bind the boundary by a single monotonic scale factor ``c`` solving
    ``sum_i exp(1/(c*s_i)^2) = (mu/p)^2 + T`` (the sum is strictly decreasing in
    ``c``), which preserves the curve's shape, then apply ``project_inverse_sigmas``
    to land exactly on the feasible boundary.

    Raises:
        ValueError: if the seated curve does not actually bind the budget. The
            bisection is bracketed and runs with ``throw=False``, so an unreachable
            root would otherwise be returned as the bracket ceiling and ship a
            silently over-noised schedule.
    """
    sigmas = jnp.asarray(sigmas)
    bound = (privacy_params.mu / privacy_params.p) ** 2 + privacy_params.T

    def residual(c, args):
        return jnp.sum(jnp.exp(1.0 / (c * sigmas) ** 2)) - bound

    # residual is strictly decreasing in c (flip=True); expand the bracket if the
    # root lies beyond the initial guess.
    bisection = optx.Bisection(rtol=1e-6, atol=1e-6, flip=True, expand_if_necessary=True)
    c = optx.root_find(
        residual,
        bisection,
        1.0,
        options={"lower": 1e-6, "upper": 10.0},
        max_steps=100,
        throw=False,
    ).value
    # Land marginally on the feasible side of the boundary: sum is decreasing in c,
    # so a hair-larger c gives sum <= bound and the projection retraction below is an
    # exact no-op. Without this the scaled sum can sit ~1e-7 *over* bound, tripping
    # project_inverse_sigmas' exact feasibility test and its fragile over-correction.
    c = c * (1.0 + 1e-6)
    seated = privacy_params.project_inverse_sigmas(c * sigmas)

    # The bisection cannot signal failure (throw=False), so verify the postcondition
    # the whole function exists to establish: the budget is actually spent.
    used = float(jnp.sum(jnp.exp(1.0 / jnp.asarray(seated) ** 2)))
    if not np.isfinite(used) or used < 0.99 * float(bound):
        raise ValueError(
            f"seat_on_budget did not bind the budget: spent {used:.6g} of "
            f"{float(bound):.6g} ({100 * used / float(bound):.4f}%). The bracketed "
            f"bisection returned c={float(c):.6g}, likely its ceiling. Check that "
            f"`sigmas` is the multiplier sigma/clip and not the raw noise scale."
        )
    return seated


class RawArraySchedule(AbstractNoiseAndClipSchedule):
    """Wrap explicit length-T sigma/clip arrays as a schedule the eval core can run.

    Evaluation-only: the outer-loop methods (``apply_updates`` / ``project``) are
    never exercised on a transferred schedule, so they raise.
    """

    sigmas: Array
    clips: Array

    def __init__(self, sigmas: Array, clips: Array):
        self.sigmas = jnp.asarray(sigmas)
        self.clips = jnp.asarray(clips)

    def get_private_noise_scales(self) -> Array:
        return self.sigmas

    def get_private_clips(self) -> Array:
        return self.clips

    def get_private_weights(self) -> Array:
        return self.get_private_clips() / self.get_private_noise_scales()

    def apply_updates(self, updates) -> Self:  # pragma: no cover - eval-only
        raise NotImplementedError("RawArraySchedule is evaluation-only.")

    def project(self) -> Self:  # pragma: no cover - eval-only
        raise NotImplementedError("RawArraySchedule is evaluation-only.")

    def _get_log_arrays(self) -> dict[str, Array]:
        return {
            "sigmas": self.get_private_noise_scales(),
            "clips": self.get_private_clips(),
        }
