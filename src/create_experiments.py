#!/usr/bin/env python3
"""
create_experiments.py — Programmatically create W&B config-seed runs for a
structured T × architecture experiment.

Each run stores a complete SweepConfig so the existing SLURM pipeline
(run-starter.py) can submit it via --wandb-conf.restart_run_id.

Usage (from src/):
    uv run create_experiments.py --project "schedule-T-arch" --entity <entity>
    uv run create_experiments.py --project "schedule-T-arch" --entity <entity> --dry-run

The output file is printed at the end; pipe it to run-starter.py:
    cat <output_file> | parallel -q uv run cc/slurm/run-starter.py \\
        --run_id={} --jobname='"schedule-T-arch"'
"""

from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Any

import tyro

import wandb
from networks.net_factory import DATASET_NETWORK_DEFAULTS

# Must be run from src/ — mirrors the PROJECT_ROOT convention in sweep.py.
_CC_ROOT = os.environ["PROJECT_ROOT"] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "cc"),
)

from conf.config import EnvConfig, ScheduleOptimizerConfig, SweepConfig
from conf.config_util import (
    DistributionConfig,
    _is_fixed_field,
    _is_union_field,
    dist_config_helper,
)
from conf.optimizer_config import (
    OptimizerConfig,
    SGDConfig,
)
from experiments.architectures import LADDER_TAG_PREFIX, LADDERS
from networks.cnn.config import CNNConfig
from networks.mlp.config import MLPConfig
from policy.schedules.config import (
    DecoupledSigmaAndClipScheduleConfig,
)

# ---------------------------------------------------------------------------
# Serialiser
# ---------------------------------------------------------------------------


def _to_run_config(
    obj: Any,
    parent_cls: type | None = None,
    field_name: str | None = None,
) -> Any:
    """Serialise a dataclass tree to a W&B run-config-compatible dict.

    Mirrors what the W&B sweep agent stores in run.config:
    - DistributionConfig   → its constant .value scalar
    - Union-typed fields   → nested dict with "_type": ClassName injected
    - Fixed fields         → skipped (as in to_wandb_sweep_params)
    - Other dataclasses    → plain nested dict (recurse)
    - Everything else      → value as-is
    """
    if isinstance(obj, DistributionConfig):
        return obj.value

    if not dataclasses.is_dataclass(obj):
        return obj

    result: dict[str, Any] = {}
    cls = type(obj)

    if (
        parent_cls is not None
        and field_name is not None
        and _is_union_field(parent_cls, field_name)
    ):
        result["_type"] = cls.__name__

    for field in dataclasses.fields(obj):
        if _is_fixed_field(cls, field.name):
            continue
        val = getattr(obj, field.name)
        result[field.name] = _to_run_config(val, parent_cls=cls, field_name=field.name)

    return result


# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------

# Shared privacy / optimisation budget.
DELTA: float = 1e-6
BATCH_SIZE: int = 250  # T=250 ≈ 1 MNIST epoch (N=60 000)
DATASETS: list[str] = ["mnist", "fashion-mnist"]
NUM_OUTER_STEPS: int = 1000
SEEDS: tuple[int, ...] = (0, 1, 2)

# --- Axis 1: vary T, architecture fixed at the dataset-default CNN ---
T_SWEEP_EPSILONS: list[float] = [1, 3, 5, 8]
T_VALUES: list[int] = [1500, 2000, 3000, 5000, 7000]

# --- Axis 2: architecture ladders (experiments/architectures.py), T fixed at ~20 epochs ---
# Exploratory first look — single (loose) privacy budget; T-sweep keeps full eps breadth.
LADDER_EPSILONS: list[float] = [8]
T_FOR_ARCH_SWEEP: int = 5000

OPTIMIZERS: list[OptimizerConfig] = [
    SGDConfig(learning_rate=dist_config_helper(value=0.05, distribution="constant")),
    # AdamConfig(learning_rate=dist_config_helper(value=1e-3, distribution="constant")),
    # AdamWConfig(learning_rate=dist_config_helper(value=1e-3, distribution="constant")),
]


def _opt_tag(opt: OptimizerConfig) -> str:
    return type(opt).__name__.removesuffix("Config").lower()


def _arch_ladder_tags() -> list[tuple[MLPConfig | CNNConfig, list[str]]]:
    """Invert LADDERS into ``[(unique arch, [ladder tags])]``.

    Architectures shared across ladders (e.g. the width/depth anchor) are emitted
    once with the union of their ``ladder:<name>`` tags, so the W&B run is created
    a single time and downstream tooling reads its membership from the tags.
    """
    unique: list[tuple[MLPConfig | CNNConfig, list[str]]] = []
    for ladder_name, archs in LADDERS.items():
        tag = f"{LADDER_TAG_PREFIX}{ladder_name}"
        for arch in archs:
            for existing_arch, tags in unique:
                if existing_arch == arch:
                    if tag not in tags:
                        tags.append(tag)
                    break
            else:
                unique.append((arch, [tag]))
    return unique


def _group_label(ds: str, eps: float, axis: str, opt_tag: str) -> str:
    return f"ds{ds.upper()}/eps={eps}/{axis}/{opt_tag}"


def _arch_label(arch: MLPConfig | CNNConfig) -> str:
    if isinstance(arch, MLPConfig):
        return "mlp-" + "x".join(str(h) for h in arch.hidden_sizes)
    ch = "x".join(str(c) for c in arch.channels)
    head = "x".join(str(h) for h in arch.mlp.hidden_sizes)
    return f"cnn-{ch}-head{head}"


def _make_sweep_config(
    ds: str,
    eps: float,
    T: int,
    network_conf: MLPConfig | CNNConfig,
    seed: int,
    optimizer: OptimizerConfig,
) -> SweepConfig:
    return SweepConfig(
        dataset=ds,
        num_outer_steps=NUM_OUTER_STEPS,
        with_baselines=True,
        baseline_log_interval=200,
        plotting_interval=50,
        prng_seed=dist_config_helper(value=float(seed), distribution="constant"),
        env=EnvConfig(
            network=network_conf,
            eps=eps,
            delta=DELTA,
            batch_size=BATCH_SIZE,
            num_training_steps=T,
            scan_segments=T,  # Loading data as-needed
            optimizer=optimizer,
        ),
        schedule_optimizer=ScheduleOptimizerConfig(
            schedule=DecoupledSigmaAndClipScheduleConfig(),
            lr=dist_config_helper(value=10.0, distribution="constant"),
        ),
    )


def _build_experiments() -> dict[str, list[tuple[list[str], str, str, SweepConfig]]]:
    """Return {opt_tag: [(tags, group, run_name, SweepConfig), ...]} for every condition × seed.

    ``tags`` is the complete W&B tag list for the run, including the axis tag and
    any ``ladder:<name>`` membership tags.
    """
    experiments: dict[str, list[tuple[list[str], str, str, SweepConfig]]] = {}

    def get_name(
        ds: str,
        eps: float,
        tpe: str,
        T: int,
        arch: MLPConfig | CNNConfig,
        seed: int,
        opt_tag: str,
    ):
        return f"{opt_tag}/ds{ds.upper()}/e{eps}/{tpe}-sweep/T={T}/{_arch_label(arch)}/seed={seed}"

    for opt in OPTIMIZERS:
        opt_tag = _opt_tag(opt)
        bucket: list[tuple[list[str], str, str, SweepConfig]] = []

        # Axis 1: T-sweep — dataset-default arch, full eps breadth.
        for ds in DATASETS:
            for eps in T_SWEEP_EPSILONS:
                for T in T_VALUES:
                    arch = DATASET_NETWORK_DEFAULTS[ds]
                    for seed in SEEDS:
                        name = get_name(ds, eps, "T", T, arch, seed, opt_tag)
                        bucket.append(
                            (
                                ["config-seed", "T-sweep", opt_tag],
                                _group_label(ds, eps, "T-sweep", opt_tag),
                                name,
                                _make_sweep_config(ds, eps, T, arch, seed, opt),
                            )
                        )

        # Axis 2: architecture ladders — deduped archs, eps=8 only, T fixed.
        for ds in DATASETS:
            for eps in LADDER_EPSILONS:
                for arch, ladder_tags in _arch_ladder_tags():
                    T = T_FOR_ARCH_SWEEP
                    for seed in SEEDS:
                        name = get_name(ds, eps, "arch", T, arch, seed, opt_tag)
                        bucket.append(
                            (
                                ["config-seed", "arch", opt_tag, *ladder_tags],
                                _group_label(ds, eps, "arch", opt_tag),
                                name,
                                _make_sweep_config(ds, eps, T, arch, seed, opt),
                            )
                        )

        experiments[opt_tag] = bucket

    return experiments


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class LauncherConfig:
    project: str
    entity: str
    dry_run: bool = False
    """Print serialised configs without creating any W&B runs."""
    num_workers: int = 8
    """Number of parallel workers to use when creating W&B runs."""


def _init_run(args: tuple[str, str, str, str, list[str], dict, str]) -> tuple[str, str, str]:
    """Worker: create a single W&B run and return (opt_tag, run_id, name)."""
    project, entity, name, group, tags, config, opt_tag = args
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        config=config,
        job_type="config-seed",
        tags=tags,
    )
    assert run is not None
    run_id = run.id
    run.finish()
    return opt_tag, run_id, name


if __name__ == "__main__":
    conf = tyro.cli(LauncherConfig)

    experiments = _build_experiments()
    total = sum(len(v) for v in experiments.values())

    timestamp = int(time.time())
    safe_name = conf.project.replace("/", "_").replace(" ", "_")
    output_paths: dict[str, str] = {
        opt_tag: os.path.join(_CC_ROOT, "sweeps", f"{safe_name}_{opt_tag}_{timestamp}.txt")
        for opt_tag in experiments
    }

    print(f"{total} runs across {len(experiments)} optimizers  →  project '{conf.project}'")

    if conf.dry_run:
        run_counts = []
        paths = []
        for opt_tag, bucket in experiments.items():
            for _tags, group, name, sweep_conf in bucket:
                print(f"\n{'─' * 64}")
                print(f"  {name}  [group={group}]")
                print(json.dumps(_to_run_config(sweep_conf), indent=2, default=str))
            run_counts.append(len(bucket))
            paths.append(output_paths[opt_tag])

        print(f"Would write {sum(run_counts)} run IDs in total:")
        for count, path in zip(run_counts, paths):
            print(f"\t{count} run IDs to:\n  {path}")
    else:
        tasks = []
        for opt_tag, bucket in experiments.items():
            for tags, group, name, sweep_conf in bucket:
                tasks.append(
                    (
                        conf.project,
                        conf.entity,
                        name,
                        group,
                        tags,
                        _to_run_config(sweep_conf),
                        opt_tag,
                    )
                )

        results_by_opt = {k: [] for k in experiments}
        with mp.get_context("spawn").Pool(processes=conf.num_workers) as pool:
            for opt_tag, run_id, name in pool.imap_unordered(_init_run, tasks):
                results_by_opt[opt_tag].append((run_id, name))
                print(f"  [{opt_tag}] {run_id}  {name}")

        for opt_tag, results in results_by_opt.items():
            output_path = output_paths[opt_tag]
            with open(output_path, "w") as f:
                for run_id, _name in results:
                    f.write(run_id + "\n")
            print(f"\nRun IDs ({opt_tag}) → {output_path}")

        print("\nSubmit to SLURM:")
        for opt_tag, output_path in output_paths.items():
            print(
                f"  cat {output_path} | parallel -q uv run cc/slurm/run-starter.py"
                f" --run_id={{}} --jobname='\"{safe_name}-{opt_tag}\"'"
            )
