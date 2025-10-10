from dataclasses import dataclass
from pprint import pprint
import numpy as np
from typing import Literal

import tyro
import jax.numpy as jnp


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

    def to_wandb(self):
        if self.distribution == "constant":
            return {"distribution": self.distribution, "value": self.value}

        return (
            {
                "min": self.min,
                "max": self.max,
                "distribution": self.distribution,
            },
        )


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
# Configs for different private networks
# ---
@dataclass
class MLPConfig:
    """Configuration for Multi-Layer Perceptron"""

    din: int = -1  # Value is derived from data
    dhidden: int = 32  # Size of hidden layers
    nhidden: int = 1  # Number of hidden layers
    nclasses: int = -1  # Value is derived from data
    initialization: Literal["glorot", "zeros"] = "glorot"
    key: int = 0  # Overridden as derivative from experiment.env_prng_key

    def to_wandb(self) -> dict[str, object]:
        attrs = [
            "din",
            "dhidden",
            "nclasses",
        ]
        return {"parameters": {attr: {"value": getattr(self, attr)} for attr in attrs}}


@dataclass
class CNNConfig:
    """Configuration for Convolutional Neural Network"""

    # linear config params
    mlp: MLPConfig

    # conv config params
    nchannels: int = -1  # Number of input channels, derived from data
    kernel_size: int = 3  # Edge Length of kernel
    pool_kernel_size: int = 2  # Edge length of pooling kernel
    hidden_channels: int = 3  # Number of hidden channels
    nhidden_conv: int = 1  # Number of hidden convolution layers
    key: int = 0  # Overridden as derivative from experiment.env_prng_key

    # dummy item, used to determine MLP input shape
    dummy_data: jnp.ndarray | None = None

    def to_wandb(self) -> dict[str, object]:
        attrs = [
            "nchannels",
            "kernel_size",
            "pool_kernel_size",
            "hidden_channels",
            "nhidden_conv",
            "key",
        ]
        return {
            "parameters": {attr: {"value": getattr(self, attr)} for attr in attrs}
            | {"mlp": self.mlp.to_wandb()}
        }


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
        value=0.001,
        distribution="constant",
    )  # Learning rate of policy network
    max_sigma: float = 10.0

    @property
    def network(self) -> MLPConfig | CNNConfig:
        """Get the actual network configuration."""
        return getattr(self, self.network_type)

    def to_wandb(self) -> dict[str, object]:
        return {
            "parameters": {
                "network_type": {"value": self.network_type},
                "network": self.network.to_wandb(),
                "lr": self.lr.to_wandb(),
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
        value=0.01,
        distribution="constant",
    )  # Learning rate of private network
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

    def to_wandb(self) -> dict[str, object]:
        return {
            "parameters": {
                "lr": self.lr.to_wandb(),
                "eps": {"value": self.eps},
                "delta": {"value": self.delta},
                "batch_size": {"value": self.batch_size},
                "max_steps_in_episode": {"value": self.max_steps_in_episode},
                "C": {"value": self.C},
                "network_type": {"value": self.network_type},
                "network": self.network.to_wandb(),
            }
        }


@dataclass
class SweepConfig:
    env: EnvConfig
    policy: PolicyConfig
    method: str = "random"  # The wandb search method
    metric_name: str = "Mean Accuracy"  # The metric for wandb to optimize
    metric_goal: str = "maximize"  # The wandb optimization goal
    plotting_steps: int = 80
    name: str | None = None  # The (optional) name of the wandb sweep
    description: str | None = None  # The (optional) description of the wandb sweep
    with_baselines: bool = False  # Flag to compute plots comparing against baseline (Expensive, default is False)

    def to_wandb(self) -> dict[str, object]:
        config = {
            "method": self.method,
            "metric": {
                "name": self.metric_name,
                "goal": self.metric_goal,
            },
            "name": self.name,
            "parameters": {
                "env": self.env.to_wandb(),
                "policy": self.policy.to_wandb(),
            },
        }

        if self.description is not None:
            config["description"] = self.description
        if self.name is not None:
            config["name"] = self.name

        return config


@dataclass
class ExperimentConfig:
    sweep: SweepConfig
    num_configs: int = 1  # Number of random agent configurations to run
    dataset: Literal["mnist", "california"] = "mnist"  # Dataset on which to privatise
    dataset_poly_d: int | None = None  # Degree of polynomial features to be generated
    total_timesteps: int = 100  # Training steps of RL algorithm
    cfg_prng_seed: int = 42  # RL Agent configuration seed
    env_prng_seed: int = 42  # Environment configuration seed
    log_dir: str = "logs"  # Relative directory in which to log results


@dataclass
class WandbConfig:
    """Wandb Configuration"""

    project: str | None = None  # The wandb project
    entity: str | None = None  # The wandb entity
    mode: Literal["disabled", "online", "offline"] = "disabled"  # The wandb mode


@dataclass
class Config:
    wandb_conf: WandbConfig
    experiment: ExperimentConfig
    data_dir: str | None = None  # Used in log_wandb.py, ignored otherwise


if __name__ == "__main__":
    args = tyro.cli(Config)
    pprint(args)
