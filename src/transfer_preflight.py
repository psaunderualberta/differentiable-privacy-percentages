"""Warm every target dataset's on-disk cache before the transfer producers run.

``util/dataloaders.py`` resolves one cache directory per dataset (``dataset_dir``:
``src/data/`` in the repo, or ``$SCRATCH/data/`` for the large targets on a cluster)
and its ``_ensure_*`` paths have no locking and no temp-rename, so two producer array
tasks first-touching the same dataset would race a half-written ``.npy``. This stage runs
**once, sequentially**, ahead of the producer arrays (which depend on it via
``-d afterok:``) so every download happens exactly once with no concurrency.

Integration glue; exercised end-to-end, not unit-tested. The launcher builds its
argv via ``transfer_launch.preflight_command``.
"""

import dataclasses

import tyro

from util.transfer import TargetSpec, build_target_config


@dataclasses.dataclass
class PreflightConfig:
    """Datasets to warm, and the batch size the producers will use."""

    datasets: tuple[str, ...]
    """Distinct target dataset names — the launcher deduplicates the cross-product."""
    batch_size: int = 250


def warm(dataset: str, batch_size: int) -> tuple:
    """Force ``dataset`` to be downloaded and cached, returning its shapes.

    Goes through the same ``build_target_config`` + singleton scope the producers
    use, so the cache it warms is exactly the one they will read. The privacy
    budget is irrelevant here — only the dataset is being touched — so a nominal
    (eps, T) is used.
    """
    from conf.scope import RunContext, using
    from conf.singleton_conf import SingletonConfig
    from util.dataloaders import get_dataset_shapes

    target = TargetSpec(name=dataset, eps=1.0, delta=1e-7, T=1, arch="")
    # Nothing here trains, so the arm is inert — but ADR 0021 makes it required rather
    # than defaulted, and a warm-up that silently picked a momentum is exactly the
    # habit that caused the bug. Named explicitly, and it reaches no optimizer.
    config = build_target_config(target, batch_size, arm="sgd-m0.9")
    # Both scopes, exactly as the three producers do it: net_factory reads the
    # SingletonConfig, dataloaders reads the conf.scope RunContext.
    with SingletonConfig.override(config), using(RunContext(config)):
        return get_dataset_shapes()


def main(conf: PreflightConfig) -> None:
    for dataset in conf.datasets:
        print(f"warming {dataset} ...", flush=True)
        shapes = warm(dataset, conf.batch_size)
        print(f"  ✓ {dataset}: {shapes}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(PreflightConfig))
