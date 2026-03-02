import dataclasses
from dataclasses import dataclass
from pprint import pprint
from typing import Annotated, Literal, Union

import tyro

from conf.config_util import (
    DistributionConfig,
    dist_config_helper,
    to_wandb_sweep_params,
)
from networks.cnn.config import CNNConfig
from networks.mlp.config import MLPConfig
from policy.schedules.config import (
    AlternatingSigmaAndClipScheduleConfig,
    DynamicDPSGDScheduleConfig,
    PolicyAndClipScheduleConfig,
    SigmaAndClipScheduleConfig,
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
]

NetworkConfig = Union[
    Annotated[MLPConfig, tyro.conf.subcommand("mlp")],
    Annotated[CNNConfig, tyro.conf.subcommand("cnn")],
]

# ---
# Config for the policy
# ---


@dataclass
class PolicyConfig:
    schedule: ScheduleConfig = dataclasses.field(
        default_factory=AlternatingSigmaAndClipScheduleConfig
    )
    batch_size: int = 1
    lr: DistributionConfig = dist_config_helper(value=1.0, distribution="constant")
    momentum: DistributionConfig = dist_config_helper(
        value=0.1, distribution="constant"
    )
    max_sigma: float = 10.0

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)


# ---
# Config for the private environment
# ---


@dataclass
class EnvConfig:
    "Configuration for the Reinforcement Learning Environment"

    network: NetworkConfig = dataclasses.field(default_factory=MLPConfig)

    lr: DistributionConfig = dist_config_helper(value=0.1, distribution="constant")
    optimizer: Literal["sgd", "adam", "adamw"] = "sgd"
    loss_type: Literal["mse", "cce"] = "cce"

    # Privacy Parameters
    eps: float = 0.5
    delta: float = 1e-7
    batch_size: int = 250
    max_steps_in_episode: int = 100

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)


@dataclass
class SweepConfig:
    env: EnvConfig
    policy: PolicyConfig
    method: str = "random"
    metric_name: str = "accuracy"
    metric_goal: str = "maximize"
    plotting_interval: int = 1
    name: str | None = None
    description: str | None = None
    with_baselines: bool = False
    dataset: Literal["mnist", "california", "cifar-10", "fashion-mnist"] = "mnist"
    dataset_poly_d: int | None = None
    total_timesteps: int = 100
    prng_seed: DistributionConfig = dist_config_helper(
        min=0,
        max=10**9,
        distribution="int_uniform",
    )
    train_on_single_network: bool = False

    @property
    def plotting_steps(self) -> int:
        if self.plotting_interval >= self.total_timesteps:
            return 1
        return self.total_timesteps // self.plotting_interval

    def to_wandb_sweep(self) -> dict[str, object]:
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


@dataclass
class WandbConfig:
    """Wandb Configuration"""

    project: str | None = None
    entity: str | None = None
    mode: Literal["disabled", "online", "offline"] = "disabled"
    restart_run_id: str | None = None


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
