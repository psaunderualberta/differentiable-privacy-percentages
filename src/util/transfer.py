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
    """The momentum arm of the whole transfer (``sgd-m0.9`` / ``sgd-m0.0``).

    Part of the source regime's identity (ADR 0018): both arms are transferred and
    their shapes differ enough that pooling them would report arm separation as
    generalization consistency.

    ADR 0021 widened it from a source-only property: the *target* now runs at the
    source's momentum too, so the arm names the source and target configuration
    together. A native reference has no learned source, so its arm is simply the
    target momentum it was tuned and evaluated at. An empty *string* rather than NaN
    because the arm is a grouping/join key, and NaN never compares equal to itself —
    it now marks only pre-ADR-0011 data carrying no arm, which is filtered, not
    plotted.
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
    "tuned_scale",
    "tuned_constants",
]


def describe_knobs(knobs) -> tuple[float, str]:
    """A cell's winning tuning knobs as the two columns the row schema records.

    ``(tuned_scale, tuned_constants)``, where the constants render as a flat
    ``"sigma.p2=1.5,clip.p1=3.0"`` — legible on a plot annotation and enough to
    reconstruct the schedule, without a nested column type in the parquet. The
    equation is named because sigma's ``p2`` and clip's ``p2`` are different numbers
    in different closed forms.

    ``None`` gives the identity knobs, so an untuned cell records ``(1.0, "")`` rather
    than a null: direct transfer *is* tuned transfer that chose to change nothing, and
    a column that is null for one producer and populated for another cannot be grouped
    on (ADR 0024).
    """
    if knobs is None:
        return 1.0, ""
    overrides = [
        # repr, not %g: the column exists so a tuned cell can be reproduced, and %g
        # would quietly round a constant like -0.004737322 to six significant digits.
        f"{equation}.{name}={float(value)!r}"
        for equation, constants in (
            ("sigma", knobs.sigma_constants),
            ("clip", knobs.clip_constants),
        )
        for name, value in constants
    ]
    return float(knobs.scale), ",".join(overrides)


def transfer_rows(
    producer: str,
    source: SourcePolicy,
    target: TargetSpec,
    results: Iterable[tuple[int, float, float]],
    knobs=None,
) -> pd.DataFrame:
    """Build the per-seed transfer rows for one source×target cell.

    ``results`` is an iterable of ``(seed, accuracy, loss)`` — the per-seed output
    of the eval core. Source/target metadata is broadcast across the seed rows.

    ``knobs`` are the tuning knobs the cell won under (ADR 0024), broadcast across the
    seed rows like the rest of the metadata; omitting them records the untuned
    identity. They belong on the row because the accuracy is now the accuracy of a
    *tuned* schedule — without them the number cannot be reproduced, and "which scale
    did each target prefer" is itself a result.
    """
    tuned_scale, tuned_constants = describe_knobs(knobs)
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
            "tuned_scale": tuned_scale,
            "tuned_constants": tuned_constants,
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

    Whether the name also carries the arm is decided by the single shared predicate
    ``transfer_launch.ARM_IN_CELL_NAME`` rather than here, so the launcher and this
    writer cannot reach different answers (ADR 0021).

    The write is atomic (temp file in the same directory + ``os.replace``): array
    tasks are killed at the wall clock, and a half-written parquet would be
    indistinguishable from a finished cell to both the assembler and the skip filter.
    """
    from transfer_launch import ARM_IN_CELL_NAME, Target, cell_filename

    producer = df["producer"].iloc[0]
    src_id = str(df["source_id"].iloc[0])
    target = Target(
        dataset=str(df["target"].iloc[0]),
        eps=float(df["target_eps"].iloc[0]),
        T=int(df["target_T"].iloc[0]),
    )
    arm = str(df["source_arm"].iloc[0]) if producer in ARM_IN_CELL_NAME else ""

    out_dir = Path(cache_root) / "transfer" / str(producer)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / cell_filename(src_id, target, arm)

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
    sweep: str,
    target: "TargetSpec",
    candidate: int,
    mean_accuracy: float,
    n: int,
    cache_root: Path | str,
    arm: str = "",
) -> Path:
    """Record one candidate's score in a tuning sweep (ADR 0019, ADR 0024).

    An *intermediate* artifact, not a transfer cell: it lives outside
    ``<cache_root>/transfer/`` so the assembler cannot mistake nineteen
    deliberately under-evaluated candidates for nineteen extra reference columns.
    Only :func:`write_transfer_cell` produces matrix rows.

    ``sweep`` is the candidate pool this score belongs to
    (``transfer_launch.sweep_id``). ADR 0019 built this for the reference stage's
    random search; ADR 0024 reuses it for the tuned curve and equation stages, which
    is why the parameter names a sweep rather than a reference mechanism.

    Atomic, for the same reason cell writes are: an array task killed at the wall
    clock must not leave a half-written score that the skip filter reads as done.
    """
    from transfer_launch import Target, candidate_filename

    out_dir = _candidate_dir(cache_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / candidate_filename(
        sweep, Target(dataset=target.name, eps=target.eps, T=target.T), candidate, arm
    )
    payload = {
        "sweep": sweep,
        "target": target.name,
        "target_eps": float(target.eps),
        "target_T": int(target.T),
        "arm": str(arm),
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
    sweep: str, target: "TargetSpec", cache_root: Path | str, arm: str = ""
) -> list[dict]:
    """Every scored candidate for one (sweep × target × arm), in candidate order.

    The selector's input. Ordered by candidate index so the winner is a
    deterministic function of what is on disk, whatever order the tasks finished in.
    Scoped to one ``arm``: after ADR 0021 a reference is swept once per target
    momentum, and a pool mixing the two would let one arm's tuning decide the other's
    baseline. Scoped to one ``sweep`` for the same reason across producers (ADR 0024):
    a tuned curve's scale search and a reference's mechanism search are different
    pools, and mixing them would have one decide the other's winner.
    """
    from transfer_launch import Target, candidate_filename

    out_dir = _candidate_dir(cache_root)
    if not out_dir.is_dir():
        return []
    launch_target = Target(dataset=target.name, eps=target.eps, T=target.T)
    prefix = candidate_filename(sweep, launch_target, 0, arm).rsplit("__cand", 1)[0]
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


def optimizer_for_arm(arm: str):
    """The inner-SGD optimizer config an arm label denotes (ADR 0011).

    ``sgd-m0.9`` / ``sgd-m0.0`` are the arm labels the ``optimizer`` column carries;
    the suffix *is* the private network's momentum. Unknown labels raise rather than
    fall back to a default — falling back is precisely the failure ADR 0021 records,
    where an unset optimizer silently ran every target at momentum 0.9.
    """
    from conf.config_util import dist_config_helper
    from conf.optimizer_config import SGDConfig

    prefix, _, momentum = str(arm).partition("-m")
    if prefix != "sgd" or not momentum:
        raise ValueError(
            f"unknown arm {arm!r}; expected an ADR 0011 label of the form 'sgd-m<momentum>' "
            "(e.g. 'sgd-m0.9' / 'sgd-m0.0')"
        )
    return SGDConfig(momentum=dist_config_helper(value=float(momentum), distribution="constant"))


def build_target_config(target: "TargetSpec", batch_size: int, arm: str):
    """A minimal singleton Config pinned to the target regime and its arm.

    ``get_privacy_params`` / ``DPTrainingParams.create_direct_from_config`` /
    ``get_dataset_shapes`` all read the singleton, so a producer wraps this config
    around them to build the target's env/GDP params. The network arch is left to
    ``AutoNetworkConfig``, which derives the surrogate arch from the dataset
    (ADR 0007), matching the target's ``arch`` label.

    ``arm`` is **required**, not defaulted (ADR 0021): the target regime inherits the
    source policy's inner momentum, and a target config that silently accepted
    ``SGDConfig``'s ``momentum=0.9`` for a caller obliged to specify it is what ran
    every ``sgd-m0.0`` transfer against a mismatched optimizer.
    """
    from conf.config import Config, EnvConfig, ScheduleOptimizerConfig, SweepConfig, WandbConfig

    env = EnvConfig(
        eps=target.eps,
        delta=target.delta,
        batch_size=batch_size,
        num_training_steps=target.T,
        optimizer=optimizer_for_arm(arm),
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
