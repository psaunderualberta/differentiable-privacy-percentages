import dataclasses
from dataclasses import dataclass
from pprint import pprint
from typing import Annotated, Any, Literal, Union

import tyro

from conf.config_util import (
    DistributionConfig,
    dist_config_helper,
    merge_wandb_sweep_union,
    to_wandb_conf,
    to_wandb_sweep_params,
)
from networks.auto.config import AutoNetworkConfig
from networks.cnn.config import CNNConfig
from networks.mlp.config import MLPConfig
from policy.schedules.config import (
    AlternatingSigmaAndClipScheduleConfig,
    DynamicDPSGDScheduleConfig,
    ParallelSigmaAndClipScheduleConfig,
    PolicyAndClipScheduleConfig,
    SigmaAndClipScheduleConfig,
    WarmupAlternatingSigmaAndClipScheduleConfig,
    WarmupParallelSigmaAndClipScheduleConfig,
    WarmupSigmaAndClipScheduleConfig,
)
from policy.stateful_schedules.config import StatefulMedianGradientNoiseAndClipConfig

# ---------------------------------------------------------------------------
# Union type aliases — tyro treats a Union of dataclasses as subcommands,
# so only one variant is ever instantiated.  ConsolidateSubcommandArgs in the
# tyro.cli call folds the subcommand selection into the flat --flag style.
# ---------------------------------------------------------------------------

ScheduleConfig = Union[
    Annotated[
        AlternatingSigmaAndClipScheduleConfig,
        tyro.conf.subcommand("alternating-sigma-and-clip"),
    ],
    Annotated[SigmaAndClipScheduleConfig, tyro.conf.subcommand("sigma-and-clip")],
    Annotated[PolicyAndClipScheduleConfig, tyro.conf.subcommand("policy-and-clip")],
    Annotated[DynamicDPSGDScheduleConfig, tyro.conf.subcommand("dynamic-dp-sgd")],
    Annotated[
        StatefulMedianGradientNoiseAndClipConfig,
        tyro.conf.subcommand("median-gradient"),
    ],
    Annotated[
        WarmupAlternatingSigmaAndClipScheduleConfig,
        tyro.conf.subcommand("warmup-alternating"),
    ],
    Annotated[
        WarmupSigmaAndClipScheduleConfig,
        tyro.conf.subcommand("warmup-sigma-and-clip"),
    ],
    Annotated[
        ParallelSigmaAndClipScheduleConfig,
        tyro.conf.subcommand("parallel-sigma-and-clip"),
    ],
    Annotated[
        WarmupParallelSigmaAndClipScheduleConfig,
        tyro.conf.subcommand("warmup-parallel-sigma-and-clip"),
    ],
]

NetworkConfig = Union[
    Annotated[AutoNetworkConfig, tyro.conf.subcommand("auto")],
    Annotated[MLPConfig, tyro.conf.subcommand("mlp")],
    Annotated[CNNConfig, tyro.conf.subcommand("cnn")],
]

# ---
# Config for the schedule optimizer
# ---


@dataclass
class ScheduleOptimizerConfig:
    schedule: ScheduleConfig = dataclasses.field(
        default_factory=WarmupParallelSigmaAndClipScheduleConfig,
    )
    batch_size: int = 1
    lr: DistributionConfig = dist_config_helper(value=1.0, distribution="constant")
    momentum: DistributionConfig = dist_config_helper(
        values=(0.0, 0.1, 0.3),
        distribution="values",
    )
    max_sigma: float = 10.0
    # When non-empty, sweep over these schedule type names rather than fixing
    # _type to the current schedule's class.  Set programmatically before
    # calling to_wandb_sweep(); not exposed as a CLI flag.
    sweep_schedule_conf_types: Annotated[tuple[str, ...], tyro.conf.Fixed] = (
        AlternatingSigmaAndClipScheduleConfig.__name__,
        WarmupAlternatingSigmaAndClipScheduleConfig.__name__,
        WarmupSigmaAndClipScheduleConfig.__name__,
        SigmaAndClipScheduleConfig.__name__,
    )

    def to_wandb_sweep(self) -> dict[str, Any]:
        result = to_wandb_sweep_params(self)
        if self.sweep_schedule_conf_types:
            # Lazy import avoids circular: singleton_conf imports config at top-level.
            from conf.singleton_conf import _get_config_classes

            config_classes = _get_config_classes()
            instances = []
            for name in self.sweep_schedule_conf_types:
                if name not in config_classes:
                    raise ValueError(
                        f"Unknown schedule type '{name}'. Known: {sorted(config_classes)}",
                    )
                instances.append(config_classes[name]())
            result["parameters"]["schedule"] = merge_wandb_sweep_union(instances)
        return result

    def to_wandb_conf(self) -> dict[str, Any]:
        return to_wandb_conf(self)  # ---


# Config for the private environment
# ---


@dataclass
class EnvConfig:
    "Configuration for the private training environment"

    network: NetworkConfig = dataclasses.field(default_factory=AutoNetworkConfig)

    lr: DistributionConfig = dist_config_helper(value=1e-3, distribution="constant")
    optimizer: Literal["sgd", "adam", "adamw"] = "sgd"
    loss_type: Literal["mse", "cce"] = "cce"

    # Privacy Parameters
    eps: float = 0.5
    delta: float = 1e-6
    batch_size: int = 250
    num_training_steps: int = 100
    scan_segments: int = -1
    """Number of segments for the scan-of-scans. Must divide num_training_steps.
    K=1 is equivalent to the current single scan with no behaviour change.
    K>1 reduces peak gradient tape memory by a factor of K at negligible runtime cost."""

    @property
    def scan_segments_derived(self) -> int:
        if self.scan_segments < 0:
            return self.num_training_steps
        return self.scan_segments

    def to_wandb_sweep(self) -> dict[str, Any]:
        return to_wandb_sweep_params(self)

    def to_wandb_conf(self) -> dict[str, Any]:
        return to_wandb_conf(self)


@dataclass
class SweepConfig:
    env: EnvConfig
    schedule_optimizer: ScheduleOptimizerConfig
    method: str = "grid"
    metric_name: str = "val-accuracy"
    metric_goal: str = "maximize"
    plotting_interval: int = 1
    name: str | None = None
    description: str | None = None
    with_baselines: bool = False
    # Only used when with_baselines=True. Evaluate and log baselines every this
    # many outer steps (0 = only log at the end of training, i.e. old behaviour).
    baseline_log_interval: int = 0
    dataset: Literal["mnist", "cifar-10", "fashion-mnist", "eyepacs"] = "mnist"
    dataset_poly_d: int | None = None
    num_outer_steps: int = 100
    prng_seed: DistributionConfig = dist_config_helper(
        values=(447831761, 159020393, 435372193),
        distribution="values",
    )
    train_on_single_network: bool = False
    shutdown_buffer_secs: int = 180
    """Seconds before SLURM wall-time expiry to trigger graceful checkpoint-and-resubmit."""

    @property
    def plotting_steps(self) -> int:
        if self.plotting_interval >= self.num_outer_steps:
            return 1
        return self.num_outer_steps // self.plotting_interval

    def to_wandb_sweep(self) -> dict[str, Any]:
        config = {
            "method": self.method,
            "metric": {
                "name": self.metric_name,
                "goal": self.metric_goal,
            },
            **to_wandb_sweep_params(self),
        }
        if self.description is not None:
            config["description"] = self.description
        if self.name is not None:
            config["name"] = self.name
        return config

    def to_wandb_conf(self) -> dict[str, Any]:
        return to_wandb_conf(self)


@dataclass
class WandbConfig:
    """Wandb Configuration"""

    project: str | None = None
    entity: str | None = None
    mode: Literal["disabled", "online", "offline"] = "disabled"
    restart_run_id: str | None = None

    # --- Checkpointing ---
    # Run ID whose checkpoint artifact to restore.  Set to the same value as
    # restart_run_id to resume a crashed job; set to a different run's ID to
    # branch from another run's state.
    checkpoint_run_id: str | None = None
    # Specific outer-loop step to restore.  None means the latest checkpoint.
    # If set to a value other than the latest step, a new W&B run is created
    # in ``{project}-branched`` with a note referencing the original run.
    checkpoint_step: int | None = None
    # Save a checkpoint (locally + W&B artifact) every this many outer steps.
    checkpoint_every: int = 25


@dataclass
class Config:
    wandb_conf: WandbConfig
    sweep: SweepConfig
    data_dir: str | None = None  # Used in log_wandb.py, ignored otherwise


if __name__ == "__main__":
    args = tyro.cli(
        Config,
        config=(tyro.conf.ConsolidateSubcommandArgs,),
    )
    pprint(args)
