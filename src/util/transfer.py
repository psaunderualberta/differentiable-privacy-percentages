"""Shared core for the policy-transfer evaluation (ADR 0008).

The curve and equation producers both feed a length-T sigma/clip schedule through
the same seating + eval core (``util/baselines.py``'s ``generate_schedule_data``).
``RawArraySchedule`` is the lossless wrapper that carries an arbitrary schedule into
that core; ``seat_on_budget`` binds a source sigma curve onto the target's DP-PSAC
budget before it is evaluated.
"""

import dataclasses
from collections.abc import Iterable
from pathlib import Path
from typing import Self

import jax.numpy as jnp
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
    """
    producer = df["producer"].iloc[0]
    src_id = df["source_id"].iloc[0]
    target = df["target"].iloc[0]
    t_eps = df["target_eps"].iloc[0]
    t_T = df["target_T"].iloc[0]

    out_dir = Path(cache_root) / "transfer" / str(producer)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{src_id}__{target}__eps{t_eps}_T{t_T}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def assemble_transfer(producer: str, cache_root: Path | str) -> pd.DataFrame:
    """Glob every cell parquet for ``producer`` and concat into one matrix frame.

    Deterministic regardless of write/glob order: rows are sorted on the cell keys.
    """
    cell_dir = Path(cache_root) / "transfer" / str(producer)
    frames = [pd.read_parquet(p) for p in sorted(cell_dir.glob("*.parquet"))]
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(_ASSEMBLE_SORT_KEYS).reset_index(drop=True)


def seat_on_budget(sigmas: Array, privacy_params: GDPPrivacyParameters) -> Array:
    """Scale a sigma curve onto the target DP-PSAC budget, then project.

    ``project_inverse_sigmas`` enforces only the *inequality*
    ``sum_i exp(1/sigma_i) <= (mu/p)^2 + T`` — a feasible-but-slack (over-noised)
    source curve passes through untouched, under-spending the target budget. So we
    first bind the boundary by a single monotonic scale factor ``c`` solving
    ``sum_i exp(1/(c*sigma_i)) = (mu/p)^2 + T`` (the sum is strictly decreasing in
    ``c``), which preserves the curve's shape, then apply ``project_inverse_sigmas``
    to land exactly on the feasible boundary.
    """
    sigmas = jnp.asarray(sigmas)
    bound = (privacy_params.mu / privacy_params.p) ** 2 + privacy_params.T

    def residual(c, args):
        return jnp.sum(jnp.exp(1.0 / (c * sigmas))) - bound

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
    return privacy_params.project_inverse_sigmas(c * sigmas)


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
