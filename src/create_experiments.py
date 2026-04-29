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
    AdamConfig,
    AdamWConfig,
    OptimizerConfig,
    SGDConfig,
)
from networks.cnn.config import CNNConfig
from networks.mlp.config import MLPConfig
from policy.schedules.config import (
    ParallelSigmaAndClipScheduleConfig,
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
EPSILONS: list[float] = [1, 3, 5, 8]
DELTA: float = 1e-6
BATCH_SIZE: int = 250  # T=250 ≈ 1 MNIST epoch (N=60 000)
DATASETS: list[str] = ["mnist", "fashion-mnist"]
NUM_OUTER_STEPS: int = 1000
SEEDS: tuple[int, ...] = (447831761, 159020393, 435372193)

# --- Axis 1: vary T, architecture fixed at medium MLP ---
T_VALUES: list[int] = [1000, 1500, 2000, 3000, 5000]

# --- Axis 2: vary architecture, T fixed at ~14 epochs ---
T_FOR_ARCH_SWEEP: int = 3500

# MLP-only architectures
MLP_ARCHS: list[MLPConfig] = [
    MLPConfig(hidden_sizes=(16,)),  # ~13K
    MLPConfig(hidden_sizes=(512, 256)),  # ~535K
    # MLPConfig(hidden_sizes=(128,)) shared with T-sweep; omitted to avoid duplicates
]

# CNN+MLP architectures
OPTIMIZERS: list[OptimizerConfig] = [
    SGDConfig(learning_rate=dist_config_helper(value=0.1, distribution="constant")),
    AdamConfig(learning_rate=dist_config_helper(value=1e-3, distribution="constant")),
    AdamWConfig(learning_rate=dist_config_helper(value=1e-4, distribution="constant")),
]


def _opt_tag(opt: OptimizerConfig) -> str:
    return type(opt).__name__.removesuffix("Config").lower()


CNN_ARCHS: list[CNNConfig] = [
    CNNConfig(  # ~11K
        channels=(16, 32),
        kernel_sizes=(8, 4),
        paddings=(2, 0),
        strides=(2, 2),
        pool_kernel_size=2,
        mlp=MLPConfig(hidden_sizes=(32,)),
    ),
    CNNConfig(  # ~38K
        channels=(32, 64),
        kernel_sizes=(8, 4),
        paddings=(2, 0),
        strides=(2, 2),
        pool_kernel_size=2,
        mlp=MLPConfig(hidden_sizes=(64,)),
    ),
    CNNConfig(  # ~141K
        channels=(64, 128),
        kernel_sizes=(8, 4),
        paddings=(2, 0),
        strides=(2, 2),
        pool_kernel_size=2,
        mlp=MLPConfig(hidden_sizes=(128,)),
    ),
]


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
        baseline_log_interval=100,
        plotting_interval=10,
        prng_seed=dist_config_helper(value=float(seed), distribution="constant"),
        env=EnvConfig(
            network=network_conf,
            eps=eps,
            delta=DELTA,
            batch_size=BATCH_SIZE,
            num_training_steps=T,
            scan_segments=T // 100,  # Loading data as-needed
            optimizer=optimizer,
        ),
        schedule_optimizer=ScheduleOptimizerConfig(
            schedule=ParallelSigmaAndClipScheduleConfig(use_fista=True),
            lr=dist_config_helper(value=0.05, distribution="constant"),
            batch_size=4,
        ),
    )


def _build_experiments() -> dict[str, list[tuple[str, str, str, SweepConfig]]]:
    """Return {opt_tag: [(axis_tag, group, run_name, SweepConfig), ...]} for every condition × seed."""
    experiments: dict[str, list[tuple[str, str, str, SweepConfig]]] = {}

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
        bucket: list[tuple[str, str, str, SweepConfig]] = []
        for ds in DATASETS:
            for eps in EPSILONS:
                # Axis 1: T sweep (medium MLP, all T values including T=3000 anchor)
                for T in T_VALUES:
                    arch = DATASET_NETWORK_DEFAULTS[ds]
                    for seed in SEEDS:
                        name = get_name(ds, eps, "T", T, arch, seed, opt_tag)
                        bucket.append(
                            (
                                "T-sweep",
                                _group_label(ds, eps, "T-sweep", opt_tag),
                                name,
                                _make_sweep_config(ds, eps, T, arch, seed, opt),
                            )
                        )

                # Axis 2: architecture sweep (T=3000, MLP variants excluding the anchor already above)
                for arch in MLP_ARCHS:
                    T = T_FOR_ARCH_SWEEP
                    for seed in SEEDS:
                        name = get_name(ds, eps, "arch", T, arch, seed, opt_tag)
                        bucket.append(
                            (
                                "arch-sweep",
                                _group_label(ds, eps, "arch-sweep", opt_tag),
                                name,
                                _make_sweep_config(ds, eps, T_FOR_ARCH_SWEEP, arch, seed, opt),
                            )
                        )

                # Axis 2: architecture sweep (T=3000, CNN+MLP variants)
                for arch in CNN_ARCHS:
                    T = T_FOR_ARCH_SWEEP
                    for seed in SEEDS:
                        name = get_name(ds, eps, "arch", T, arch, seed, opt_tag)
                        bucket.append(
                            (
                                "arch-sweep",
                                _group_label(ds, eps, "arch-sweep", opt_tag),
                                name,
                                _make_sweep_config(ds, eps, T_FOR_ARCH_SWEEP, arch, seed, opt),
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
            for _axis, group, name, sweep_conf in bucket:
                print(f"\n{'─' * 64}")
                print(f"  {name}  [group={group}]")
                print(json.dumps(_to_run_config(sweep_conf), indent=2, default=str))
            run_counts.append(len(bucket))
            paths.append(output_paths[opt_tag])

        print(f"Would write {sum(run_counts)} run IDs in total:")
        for count, path in zip(run_counts, paths):
            print(f"\t{count} run IDs to:\n  {path}")
    else:
        for opt_tag, bucket in experiments.items():
            output_path = output_paths[opt_tag]
            with open(output_path, "w") as f:
                for axis, group, name, sweep_conf in bucket:
                    run = wandb.init(
                        project=conf.project,
                        entity=conf.entity,
                        name=name,
                        group=group,
                        config=_to_run_config(sweep_conf),
                        job_type="config-seed",
                        tags=["config-seed", axis, opt_tag],
                    )
                    assert run is not None
                    f.write(run.id + "\n")
                    f.flush()
                    run.finish()
                    print(f"  [{opt_tag}] {run.id}  {name}")
            print(f"\nRun IDs ({opt_tag}) → {output_path}")

        print("\nSubmit to SLURM:")
        for opt_tag, output_path in output_paths.items():
            print(
                f"  cat {output_path} | parallel -q uv run cc/slurm/run-starter.py"
                f" --run_id={{}} --jobname='\"{safe_name}-{opt_tag}\"'"
            )
