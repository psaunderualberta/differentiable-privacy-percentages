from dataclasses import dataclass
from pprint import pprint
from typing import Annotated, Literal, Union

import tyro
from tyro.conf import Fixed

from conf.config_util import (
    DistributionConfig,
    dist_config_helper,
    to_wandb_sweep_params,
)
from networks.cnn.config import CNNConfig
from networks.mlp.config import MLPConfig

# ---
# Config for the policy
# ---


@dataclass
class PolicyConfig:
    network: Union[
        Annotated[MLPConfig, tyro.conf.subcommand("mlp")],
        Annotated[CNNConfig, tyro.conf.subcommand("cnn")],
    ]
    batch_size: int = 1  # Batch size for policy training
    lr: DistributionConfig = dist_config_helper(
        value=1.0,
        distribution="constant",
    )  # Learning rate configuration of policy network
    max_sigma: float = 10.0

    attrs: Fixed[tuple[str, ...]] = (
        "network",
        "lr",
        "batch_size",
        "max_sigma",
    )

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)


# ---
# Config for the private environemnt
# ---


@dataclass
class EnvConfig:
    "Configuration for the Reinforcement Learning Environment"

    network: Union[
        Annotated[MLPConfig, tyro.conf.subcommand("mlp")],
        Annotated[CNNConfig, tyro.conf.subcommand("cnn")],
    ]

    lr: DistributionConfig = dist_config_helper(
        value=0.1,
        distribution="constant",
    )  # Learning rate of private network
    optimizer: Literal["sgd", "adam", "adamw"] = "sgd"
    loss_type: Literal["mse", "cce"] = "cce"  # The type of loss function to use

    # Privacy Parameters
    eps: float = 0.5  # Epsilon privacy parameter
    delta: float = 1e-7  # Delta privacy parameter
    batch_size: int = 250  # Batch size for NN training
    max_steps_in_episode: int = 100  # Maximum # of steps within an episode
    C: float = 1.0  # Ignored

    attrs: Fixed[tuple[str, ...]] = (
        "network",
        "lr",
        "optimizer",
        "loss_type",
        "eps",
        "delta",
        "batch_size",
        "max_steps_in_episode",
        "C",
    )

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)


@dataclass
class SweepConfig:
    env: EnvConfig
    policy: PolicyConfig
    method: str = "random"  # The wandb search method
    metric_name: str = "accuracy"  # The metric for wandb to optimize
    metric_goal: str = "maximize"  # The wandb optimization goal
    plotting_steps: int = 30
    name: str | None = None  # The (optional) name of the wandb sweep
    description: str | None = None  # The (optional) description of the wandb sweep
    with_baselines: bool = False  # Flag to compute plots comparing against baseline (Expensive, default is False)
    dataset: Literal["mnist", "california", "cifar-10", "fashion-mnist"] = (
        "mnist"  # Dataset on which to privatise
    )
    dataset_poly_d: int | None = None  # Degree of polynomial features to be generated
    total_timesteps: int = 100  # Training steps of RL algorithm
    cfg_prng_seed: int = 42  # RL Agent configuration seed
    env_prng_seed: int = 42  # Environment configuration seed
    train_on_single_network: bool = False  # Train the policy on only a single network (same initialization & minibatches)

    attrs: Fixed[tuple[str, ...]] = (
        "policy",
        "env",
        "plotting_steps",
        "with_baselines",
        "dataset",
        "dataset_poly_d",
        "total_timesteps",
        "cfg_prng_seed",
        "env_prng_seed",
        "train_on_single_network",
    )

    @property
    def plotting_interval(self) -> int:
        if self.plotting_steps >= self.total_timesteps:
            return 1
        return self.total_timesteps // self.plotting_steps

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

    project: str | None = None  # The wandb project
    entity: str | None = None  # The wandb entity
    mode: Literal["disabled", "online", "offline"] = "disabled"  # The wandb mode
    restart_run_id: str | None = (
        None  # Wandb Run ID from which to download config, populate for script
    )


@dataclass
class Config:
    wandb_conf: WandbConfig
    sweep: SweepConfig
    data_dir: str | None = None  # Used in log_wandb.py, ignored otherwise


if __name__ == "__main__":
    args = tyro.cli(Config, config=(tyro.conf.ConsolidateSubcommandArgs,))
    pprint(args)
