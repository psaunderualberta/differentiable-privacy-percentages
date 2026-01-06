from dataclasses import dataclass
from pprint import pprint
from typing import Literal

import numpy as np
import tyro

from networks.cnn.config import CNNConfig
from networks.mlp.config import MLPConfig


@dataclass(frozen=True)
class DistributionConfig:
    min: float  # minimum of distribiution
    max: float  # maximum of distribution
    value: float  # constant value if distribution == 'constant'
    distribution: str  # type of distribution, in wandb-format (i.e. uniform, log_uniform_values, etc.)

    def sample(self) -> float:
        if self.distribution == "constant":
            return self.value

        elif self.distribution == "log_uniform_values":
            return np.exp(
                np.random.uniform(low=np.log(self.min), high=np.log(self.max))
            )

        return np.random.uniform(low=self.min, high=self.min)

    def to_wandb_sweep(self):
        if self.distribution == "constant":
            return {"distribution": self.distribution, "value": self.value}

        return {
            "min": self.min,
            "max": self.max,
            "distribution": self.distribution,
        }


# wandb cannot create sweeps if any distribution has min >= max
# >:(
def dist_config_helper(
    min: float = 0.0,
    max: float = 0.0,
    value: float = 0.0,
    distribution: str = "constant",
) -> DistributionConfig:
    if min >= max:
        max += 1e-10
    return DistributionConfig(min=min, max=max, value=value, distribution=distribution)


# ---
# Config for the policy
# ---


@dataclass
class PolicyConfig:
    cnn: CNNConfig  # Configuration for the CNN policy
    mlp: MLPConfig  # Configuration for the MLP policy
    network_type: Literal["mlp", "cnn"] = "mlp"  # The type of network to use as policy
    batch_size: int = 1  # Batch size for policy training
    lr: DistributionConfig = dist_config_helper(
        value=1.0,
        distribution="constant",
    )  # Learning rate configuration of policy network
    max_sigma: float = 10.0

    @property
    def network(self) -> MLPConfig | CNNConfig:
        """Get the actual network configuration."""
        return getattr(self, self.network_type)

    def to_wandb_sweep(self) -> dict[str, object]:
        assert isinstance(self.lr, DistributionConfig)
        return {
            "parameters": {
                "network_type": {"value": self.network_type},
                "mlp": self.mlp.to_wandb_sweep(),
                "cnn": self.cnn.to_wandb_sweep(),
                "lr": self.lr.to_wandb_sweep(),
                "batch_size": {"value": self.batch_size},
                "max_sigma": {"value": self.max_sigma},
            }
        }


# ---
# Config for the private environemnt
# ---


@dataclass
class EnvConfig:
    "Configuration for the Reinforcement Learning Environment"

    mlp: MLPConfig  # Configuration for the MLP to privatize. Ignored if 'network_type' = cnn
    cnn: CNNConfig  # Configuration for the CNN to privatize. Ignored if 'network_type' = mlp

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
    network_type: Literal["mlp", "cnn"] = "mlp"  # The type of network to privatize.

    # Derived object, getting the actual network config
    @property
    def network(self) -> MLPConfig | CNNConfig:
        return getattr(self, self.network_type)

    def to_wandb_sweep(self) -> dict[str, object]:
        assert isinstance(self.lr, DistributionConfig)
        return {
            "parameters": {
                "mlp": self.mlp.to_wandb_sweep(),
                "cnn": self.cnn.to_wandb_sweep(),
                "lr": self.lr.to_wandb_sweep(),
                "optimizer": {"value": self.optimizer},
                "loss_type": {"value": self.loss_type},
                "eps": {"value": self.eps},
                "delta": {"value": self.delta},
                "batch_size": {"value": self.batch_size},
                "max_steps_in_episode": {"value": self.max_steps_in_episode},
                "C": {"value": self.C},
                "network_type": {"value": self.network_type},
            }
        }


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
            "parameters": {
                "env": self.env.to_wandb_sweep(),
                "policy": self.policy.to_wandb_sweep(),
                "plotting_steps": {"value": self.plotting_steps},
                "with_baselines": {"value": self.with_baselines},
                "dataset": {"value": self.dataset},
                "dataset_poly_d": {"value": self.dataset_poly_d},
                "total_timesteps": {"value": self.total_timesteps},
                "cfg_prng_seed": {"value": self.cfg_prng_seed},
                "env_prng_seed": {"value": self.env_prng_seed},
                "train_on_single_network": {"value": self.train_on_single_network},
            },
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
