"""Reference-transfer producer (ADR 0008).

The three *native* references (Constant, DynamicDPSGD, StatefulMedianGradient) are
not transferred from a source — they are swept/evaluated directly on the target
regime. This producer runs ``Baseline.generate_baseline_data`` on a target and
writes one ``producer="reference"`` cell per reference, under the shared transfer
schema (``util/transfer.py``), so the assembler treats them as extra columns of
the same matrix.
"""

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
    there is no source run to read a sampling rate from.
    """
    return SourcePolicy(
        run_id=reference_slug,
        dataset=target.name,
        eps=target.eps,
        delta=target.delta,
        T=target.T,
        p=float("nan"),
        arch=target.arch,
    )


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
# Cell orchestration + CLI (integration glue; exercised end-to-end, not unit-tested)
# ---------------------------------------------------------------------------


def run_reference_cell(
    target: TargetSpec,
    cache_root: Path | str = "cache",
    batch_size: int = 250,
    num_reps: int = 8,
    seed: int = 0,
    reference: str = "",
) -> list[Path]:
    """Sweep the native references on ``target`` and write one cell per reference.

    Runs ``Baseline.generate_baseline_data`` (sweep + final eval) inside the target
    config scope, then writes a ``producer="reference"`` parquet cell per reference
    under the shared transfer schema. ``reference`` narrows the sweep to a single
    reference — the launcher runs each as its own job — without changing the keys
    it draws, so a narrowed run is comparable to a combined one.
    """
    from conf.scope import RunContext, using
    from conf.singleton_conf import SingletonConfig
    from environments.dp_params import DPTrainingParams
    from privacy.gdp_privacy import get_privacy_params
    from util.baselines import REFERENCES, Baseline
    from util.dataloaders import get_dataset_shapes

    references = (_reference_for_slug(reference),) if reference else REFERENCES
    config = build_target_config(target, batch_size)
    # The inner DP-SGD path also reads the singleton / RunContext, so the whole sweep
    # (not just param construction) must stay inside both scopes — otherwise a
    # training-time singleton read finds it reset and re-parses sys.argv.
    with SingletonConfig.override(config), using(RunContext(config)):
        X_shape, *_ = get_dataset_shapes()
        gdp_params = get_privacy_params(X_shape[0])
        env_params = DPTrainingParams.create_direct_from_config()

        baseline = Baseline(env_params, gdp_params, jr.PRNGKey(seed), num_reps=num_reps)
        df = baseline.generate_baseline_data(
            jr.PRNGKey(seed), with_progress_bar=False, references=references
        )

    paths: list[Path] = []
    for slug, results in baseline_data_to_results(df).items():
        rows = transfer_rows("reference", reference_source(slug, target), target, results)
        paths.append(write_transfer_cell(rows, cache_root))
    return paths


@dataclasses.dataclass
class ReferenceCellConfig:
    """Native references swept on one target regime — by default all three, or one
    per invocation when the launcher fans them out with ``--reference``."""

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
    reference: str = ""
    """Sweep only this reference slug (Constant, Dynamic-DPSGD, Median); empty = all three."""


def main(conf: ReferenceCellConfig) -> None:
    target = TargetSpec(
        name=conf.target,
        eps=conf.target_eps,
        delta=conf.target_delta,
        T=conf.target_T,
        arch=conf.target_arch,
    )
    for out in run_reference_cell(
        target,
        cache_root=conf.cache_root,
        batch_size=conf.batch_size,
        num_reps=conf.num_reps,
        seed=conf.seed,
        reference=conf.reference,
    ):
        print(f"wrote {out}")


if __name__ == "__main__":
    main(tyro.cli(ReferenceCellConfig))
