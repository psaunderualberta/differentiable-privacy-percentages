from dataclasses import dataclass
from pprint import pprint
import chex
import numpy as np
from typing import Dict, Literal

import tyro


@dataclass(frozen=True)
class DistributionConfig:
    min: float  # minimum of distribiution
    max: float  # maximum of distribution
    distribution: str  # type of distribution, in wandb-format (i.e. uniform, log_uniform_values, etc.)

    def sample(self) -> float:
        if self.distribution == "log_uniform_values":
            return np.exp(
                np.random.uniform(low=np.log(self.min), high=np.log(self.max))
            )
        
        return np.random.uniform(low=self.min, high=self.min)



# wandb cannot create sweeps if any distribution has min >= max
# >:(
def dist_config_helper(min: float, max: float, distribution: str) -> DistributionConfig:
    if min >= max:
        max += 1e-10
    return DistributionConfig(min=min, max=max, distribution=distribution)


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
    key: int = 0  # Overridden as derivative from experiment.env_prng_key

    def to_wandb(self) -> Dict:
        attrs = [
            "din",
            "dhidden",
            "nclasses",
        ]
        return {"parameters": {attr: {"value": getattr(self, attr)} for attr in attrs}}


@dataclass
class CNNConfig:
    """Configuration for Convolutional Neural Network"""

    # conv config params
    nchannels: int = -1  # Number of input channels, derived from data
    kernel_size: int = 5  # Size of kernel
    pool_dim: int = 2  # Which dimension to pool over
    conv_dim_out: int = 6272  # Dimension of the output after convolution and flattening
    hidden_channels: int = 32  # Number of hidden channels
    nhidden_conv: int = 1  # Number of hidden convolution layers

    # linear config params
    dhidden: int = 32  # See MLP.dhidden
    nhidden: int = 2  # See MLP.nhidden
    nclasses: int = 10  # Value is derived from data
    key: int = 0  # Overridden as derivative from experiment.env_prng_key

    def to_wandb(self) -> Dict:
        attrs = [
            "nchannels",
            "kernel_size",
            "pool_dim",
            "conv_dim_out",
            "hidden_channels",
            "nhidden_conv",
            "dhidden",
            "nhidden",
            "nclasses",
        ]
        return {"parameters": {attr: {"value": getattr(self, attr)} for attr in attrs}}

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
        min=1e-3, max=1e-3, distribution="log_uniform_values"
    )  # Learning rate of policy network


    @property
    def network(self) -> MLPConfig | CNNConfig:
        """Get the actual network configuration."""
        return getattr(self, self.network_type)
    
    def to_wandb(self) -> Dict:
        return {
            "parameters": {
                "network_type": {"value": self.network_type},
                "network": self.network.to_wandb(),
                "lr": {
                    "min": self.lr.min,
                    "max": self.lr.max,
                    "distribution": self.lr.distribution,
                },
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
        min=0.01, max=0.01, distribution="log_uniform_values"
    )  # Learning rate of private network

    # Privacy Parameters
    eps: float = 0.5 # Epsilon privacy parameter
    delta: float = 1e-7  # Delta privacy parameter
    batch_size: int = 512  # Batch size for NN training
    max_steps_in_episode: int = 30 # Maximum # of steps within an episode
    C: float = 1.0  # Ignored
    network_type: Literal["mlp", "cnn"] = "mlp"  # The type of network to privatize.

    # Derived object, getting the actual network config
    @property
    def network(self) -> MLPConfig | CNNConfig:
        return getattr(self, self.network_type)

    def to_wandb(self) -> Dict:
        return {
            "parameters": {
                "lr": {
                    "min": self.lr.min,
                    "max": self.lr.max,
                    "distribution": self.lr.distribution,
                },
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
    name: str | None = None  # The (optional) name of the wandb sweep
    description: str | None = None  # The (optional) description of the wandb sweep
    with_baselines: bool = False  # Flag to compute plots comparing against baseline (Expensive, default is False)

    def to_wandb(self) -> Dict:
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
    total_timesteps: int = 1000  # Training steps of RL algorithm
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
