"""Reference-transfer producer (ADR 0008).

The three *native* references (Constant, DynamicDPSGD, StatefulMedianGradient) are
not transferred from a source — they are swept/evaluated directly on the target
regime, and written under the shared transfer schema (``util/transfer.py``) so the
assembler treats them as extra columns of the same matrix.

**Two phases** (ADR 0019). A reference's random search is 20 candidates, which as
one blocking call is ~87 GPU-hours per target — every task would have been killed
at roughly 13% of its work. So the stage is split along a *candidate* dimension:

    run_candidate_cell   one (reference x target x candidate), ~1.3h, writes an
                         intermediate score OUTSIDE cache/transfer/
    run_selector_cell    reads those scores, picks the winner, runs the final
                         evaluation, writes the one producer="reference" cell

Total compute is unchanged; only its packaging is, and the full 20-candidate search
survives — shrinking it instead would make the references a straw man against the
very claim they exist to test.
"""

import contextlib
import dataclasses
from pathlib import Path

import pandas as pd
import tyro
from jax import random as jr

from util.transfer import (
    SourcePolicy,
    TargetSpec,
    build_target_config,
    transfer_rows,
    write_transfer_cell,
)

# The ``type`` strings Baseline.generate_baseline_data tags each native reference
# with, mapped to clean, filesystem-safe slugs used as the cell's ``source_id``.
# These name *references*, not regimes (CONTEXT.md): a regime is a (dataset, eps, T,
# arch) tuple, whereas each of these is one baseline mechanism.
_REFERENCE_SLUGS = {
    "Constant σ/clip": "Constant",
    "Dynamic-DPSGD": "Dynamic-DPSGD",
    "Adaptive Clip (Andrew et al.)": "Median",
}


def reference_slugs() -> list[str]:
    """The clean, filesystem-safe slug for each native reference."""
    return list(_REFERENCE_SLUGS.values())


def _reference_for_slug(slug: str) -> str:
    """The Baseline reference name a filesystem-safe slug refers to."""
    for reference, reference_slug in _REFERENCE_SLUGS.items():
        if reference_slug == slug:
            return reference
    raise SystemExit(f"unknown reference {slug!r}; expected one of {reference_slugs()}")


def reference_source(reference_slug: str, target: TargetSpec) -> SourcePolicy:
    """The SourcePolicy for a native reference evaluated on ``target``.

    A reference is not transferred from a learned run, so its source provenance IS the
    target regime (dataset/eps/delta/T/arch), tagged by the reference slug. ``p`` is NaN:
    there is no source run to read a sampling rate from. ``arm`` is ``""`` — the arm is
    an outer-loop condition (ADR 0011) and a reference was never learned in one. It is
    the empty string rather than NaN because ``source_arm`` is a grouping and
    overlay-join key, and NaN never compares equal to itself.
    """
    return SourcePolicy(
        run_id=reference_slug,
        dataset=target.name,
        eps=target.eps,
        delta=target.delta,
        T=target.T,
        p=float("nan"),
        arch=target.arch,
        arm="",
    )


def select_candidate(records: list[dict]) -> int:
    """The winning candidate index from a reference's per-candidate scores (ADR 0019).

    Highest 3-run mean accuracy, ties broken on the lowest candidate index so the
    winner is a function of the scores alone and not of which SLURM task finished
    first. An empty score set raises: a selector whose candidates all died would
    otherwise silently report an untuned reference as the tuned baseline, which is
    exactly the straw-man result the split exists to avoid.
    """
    if not records:
        raise SystemExit(
            "no candidate scores to select from; the candidate phase for this "
            "reference × target either has not run or failed entirely"
        )
    best = max(records, key=lambda record: (record["mean_accuracy"], -record["candidate"]))
    return int(best["candidate"])


def baseline_data_to_results(df: pd.DataFrame) -> dict[str, list[tuple[int, float, float]]]:
    """Split a generate_baseline_data frame into per-reference ``(seed, acc, loss)``.

    Rows are grouped by the ``type`` column (one group per native reference) and
    remapped to a clean reference slug; each group's reps are seed-indexed 0..N-1 in
    frame order.
    """
    results: dict[str, list[tuple[int, float, float]]] = {}
    for reference_type, group in df.groupby("type", sort=False):
        slug = _REFERENCE_SLUGS[str(reference_type)]
        results[slug] = [
            (i, float(acc), float(loss))
            for i, (acc, loss) in enumerate(zip(group["accuracy"], group["loss"]))
        ]
    return results


# ---------------------------------------------------------------------------
# Phase orchestration + CLI (integration glue; exercised end-to-end, not unit-tested)
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _baseline_on_target(target: TargetSpec, batch_size: int, num_reps: int, seed: int):
    """A ``Baseline`` bound to the target regime, inside both config scopes.

    The inner DP-SGD path also reads the singleton / RunContext, so everything the
    caller does with the Baseline (not just param construction) has to stay inside
    both — otherwise a training-time singleton read finds it reset and re-parses
    ``sys.argv``.
    """
    from conf.scope import RunContext, using
    from conf.singleton_conf import SingletonConfig
    from environments.dp_params import DPTrainingParams
    from privacy.gdp_privacy import get_privacy_params
    from util.baselines import Baseline
    from util.dataloaders import get_dataset_shapes

    config = build_target_config(target, batch_size)
    with SingletonConfig.override(config), using(RunContext(config)):
        X_shape, *_ = get_dataset_shapes()
        gdp_params = get_privacy_params(X_shape[0])
        env_params = DPTrainingParams.create_direct_from_config()
        yield Baseline(env_params, gdp_params, jr.PRNGKey(seed), num_reps=num_reps)


def _sweep_key(reference_slug: str, seed: int):
    """The reference's sweep key — the one the combined three-reference sweep gives it."""
    from util.baselines import reference_sweep_keys

    return reference_sweep_keys(jr.PRNGKey(seed))[_reference_for_slug(reference_slug)]


def run_candidate_cell(
    reference: str,
    target: TargetSpec,
    candidate: int,
    cache_root: Path | str = "cache",
    batch_size: int = 250,
    num_reps: int = 3,
    seed: int = 0,
    iterations: int = 0,
) -> Path:
    """Score one sweep candidate on the target and write its record (ADR 0019).

    The first phase of the split reference stage: one SLURM task per (reference ×
    target × candidate), bounded at ``iterations`` inner trainings, writing an
    intermediate score rather than a transfer cell. Candidate enumeration is a pure
    function of (reference, key, index), so this task builds exactly the schedule
    the monolithic 20-candidate sweep would have evaluated at that position.
    """
    from util.baselines import SWEEP_SCORING_ITERATIONS
    from util.transfer import write_candidate_record

    iterations = iterations or SWEEP_SCORING_ITERATIONS
    name = _reference_for_slug(reference)
    with _baseline_on_target(target, batch_size, num_reps, seed) as baseline:
        schedules = baseline.candidate_schedules(name, _sweep_key(reference, seed))
        if not 0 <= candidate < len(schedules):
            raise SystemExit(
                f"candidate {candidate} is outside this sweep's 0..{len(schedules) - 1}"
            )
        score = baseline.score_candidate(schedules[candidate], name, iterations)

    return write_candidate_record(reference, target, candidate, score, iterations, cache_root)


def run_selector_cell(
    reference: str,
    target: TargetSpec,
    cache_root: Path | str = "cache",
    batch_size: int = 250,
    num_reps: int = 3,
    seed: int = 0,
) -> Path:
    """Pick the sweep winner off the candidate records and write the reference cell.

    The second phase of the split reference stage, and the only one whose output is
    a transfer cell. Re-evaluates the winner on the held-out split under the
    Baseline's own key — disjoint from the scoring draws, so the reported number is
    not the draw that selected it, and shared with the curve/equation producers, so
    a reference cell and the cells it is compared against see common random numbers.
    """
    from util.transfer import read_candidate_records

    winner = select_candidate(read_candidate_records(reference, target, cache_root))
    name = _reference_for_slug(reference)
    print(f"{reference}: candidate {winner} won the sweep; running the final evaluation")

    with _baseline_on_target(target, batch_size, num_reps, seed) as baseline:
        schedules = baseline.candidate_schedules(name, _sweep_key(reference, seed))
        df = baseline.evaluate_candidate(schedules[winner], name, with_progress_bar=False)

    results = baseline_data_to_results(df)[reference]
    rows = transfer_rows("reference", reference_source(reference, target), target, results)
    return write_transfer_cell(rows, cache_root)


@dataclasses.dataclass
class ReferenceCellConfig:
    """One phase of one native reference's sweep on one target regime (ADR 0019).

    ``--candidate N`` scores sweep candidate N; omitting it runs the selector, which
    reads every candidate's score, picks the winner and writes the transfer cell.
    """

    reference: str
    """Which native reference to sweep (Constant, Dynamic-DPSGD, Median)."""
    target: str
    """Target dataset name (eyepacs, imagenet, chexpert, ...)."""
    target_eps: float
    target_T: int
    target_delta: float = 1e-7
    target_arch: str = ""
    """Arch label recorded on the cell rows; the arch itself is auto-derived from the dataset."""
    batch_size: int = 250
    cache_root: str = "cache"
    num_reps: int = 3
    seed: int = 0
    candidate: int = -1
    """Score this sweep candidate instead of selecting; -1 runs the selector phase."""
    iterations: int = 0
    """Inner trainings per scored candidate; 0 uses baselines.SWEEP_SCORING_ITERATIONS."""


def main(conf: ReferenceCellConfig) -> None:
    target = TargetSpec(
        name=conf.target,
        eps=conf.target_eps,
        delta=conf.target_delta,
        T=conf.target_T,
        arch=conf.target_arch,
    )
    shared = {
        "cache_root": conf.cache_root,
        "batch_size": conf.batch_size,
        "num_reps": conf.num_reps,
        "seed": conf.seed,
    }
    if conf.candidate >= 0:
        out = run_candidate_cell(
            conf.reference, target, conf.candidate, iterations=conf.iterations, **shared
        )
    else:
        out = run_selector_cell(conf.reference, target, **shared)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(tyro.cli(ReferenceCellConfig))
