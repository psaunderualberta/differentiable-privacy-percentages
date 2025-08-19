from dataclasses import dataclass
from pprint import pprint
from typing import Dict, Literal

import tyro


@dataclass(frozen=True)
class DistributionConfig:
    min: float  # minimum of distribiution
    max: float  # maximum of distribution
    distribution: str  # type of distribution, in wandb-format (i.e. uniform, log_uniform_values, etc.)


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
# Configs for different RL algorithms
# ---
@dataclass
class StreamQConfig:
    """Configuration for the 'Stream-Q' RL algorithm"""

    lambda_: DistributionConfig = dist_config_helper(
        min=0.8, max=1.0, distribution="uniform"
    )
    alpha: DistributionConfig = dist_config_helper(
        min=1.0, max=1.0, distribution="uniform"
    )
    kappa: DistributionConfig = dist_config_helper(
        min=2.0, max=2.0, distribution="uniform"
    )
    start_e: DistributionConfig = dist_config_helper(
        min=1.0, max=1.0, distribution="uniform"
    )
    end_e: DistributionConfig = dist_config_helper(
        min=0.01, max=0.1, distribution="uniform"
    )
    stop_exploring_fraction: DistributionConfig = dist_config_helper(
        min=0.5, max=0.7, distribution="uniform"
    )

    def to_wandb(self) -> Dict:
        attrs = [
            "lambda_",
            "alpha",
            "kappa",
            "start_e",
            "end_e",
            "stop_exploring_fraction",
        ]
        return {
            "parameters": {
                attr: {
                    "min": getattr(self, attr).min,
                    "max": getattr(self, attr).max,
                    "distribution": getattr(self, attr).distribution,
                }
                for attr in attrs
            }
        }


@dataclass
class StreamACConfig:
    """Configuration for the 'Stream-AC' RL algorithm"""

    lambda_: DistributionConfig = dist_config_helper(
        min=0.8, max=1.0, distribution="uniform"
    )
    alpha: DistributionConfig = dist_config_helper(
        min=1.0, max=1.0, distribution="uniform"
    )
    policy_kappa: DistributionConfig = dist_config_helper(
        min=3.0, max=3.0, distribution="uniform"
    )
    value_kappa: DistributionConfig = dist_config_helper(
        min=2.0, max=2.0, distribution="uniform"
    )
    tau: DistributionConfig = dist_config_helper(
        min=1e-3, max=1e-1, distribution="log_uniform_values"
    )

    def to_wandb(self) -> Dict:
        attrs = [
            "lambda_",
            "alpha",
            "policy_kappa",
            "value_kappa",
            "tau",
        ]
        return {
            "parameters": {
                attr: {
                    "min": getattr(self, attr).min,
                    "max": getattr(self, attr).max,
                    "distribution": getattr(self, attr).distribution,
                }
                for attr in attrs
            }
        }


@dataclass
class StreamSarsaConfig:
    """Configuration for the 'Stream-Sarsa' RL algorithm"""

    lambda_: DistributionConfig = dist_config_helper(
        min=0.8, max=1.0, distribution="uniform"
    )
    alpha: DistributionConfig = dist_config_helper(
        min=1.0, max=1.0, distribution="uniform"
    )
    kappa: DistributionConfig = dist_config_helper(
        min=2.0, max=2.0, distribution="uniform"
    )
    start_e: DistributionConfig = dist_config_helper(
        min=1.0, max=1.0, distribution="uniform"
    )
    end_e: DistributionConfig = dist_config_helper(
        min=0.01, max=0.1, distribution="uniform"
    )
    stop_exploring_fraction: DistributionConfig = dist_config_helper(
        min=0.5, max=0.7, distribution="uniform"
    )

    def to_wandb(self) -> Dict:
        attrs = [
            "lambda_",
            "alpha",
            "kappa",
            "start_e",
            "end_e",
            "stop_exploring_fraction",
        ]
        return {
            "parameters": {
                attr: {
                    "min": getattr(self, attr).min,
                    "max": getattr(self, attr).max,
                    "distribution": getattr(self, attr).distribution,
                }
                for attr in attrs
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

    var_low: float = 0.0  # Lower bound of variance
    var_high: float = 20.0  # Upper bound of variance

    # Privacy Parameters
    eps: float = 0.5 # Epsilon privacy parameter
    delta: float = 1e-7  # Delta privacy parameter
    batch_size: int = 512  # Batch size for NN training
    moments: int = 50  # The # of moments to use within the moments accountant
    max_steps_in_episode: int = 500 # Maximum # of steps within an episode
    C: float = 1.0  # Ignored
    action: float = (
        0.0  # Initial action, ignored for algorithms which don't use past actions as input
    )
    network_type: Literal["mlp", "cnn"] = "mlp"  # The type of network to privatize.

    # The type of steps to take.
    step_taker: Literal["private", "non-private", "averaged-reward", "sticky-actions", "privacy-percentage"] = "private"

    # The type of actions produced by the RL algorithm
    action_taker: Literal["continuous", "discrete", "squashed", "change", "privacy-percentage"] = "continuous"

    # The type of observation provided to the RL algorithm.
    obs_maker: Literal["accuracy", "iteration", "hidden-node-grads"] = "accuracy"

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
                "var_low": {"value": self.var_low},
                "var_high": {"value": self.var_high},
                "eps": {"value": self.eps},
                "delta": {"value": self.delta},
                "batch_size": {"value": self.batch_size},
                "moments": {"value": self.moments},
                "max_steps_in_episode": {"value": self.max_steps_in_episode},
                "C": {"value": self.C},
                "action": {"value": self.action},
                "network_type": {"value": self.network_type},
                "network": self.network.to_wandb(),
                "step_taker": {"value": self.step_taker},
                "obs_maker": {"value": self.obs_maker},
                "action_taker": {"value": self.action_taker},
            }
        }


@dataclass
class SweepConfig:
    env: EnvConfig
    stream_sarsa: StreamSarsaConfig
    stream_q: StreamQConfig
    stream_ac: StreamACConfig
    method: str = "random"  # The wandb search method
    metric_name: str = "Mean Accuracy"  # The metric for wandb to optimize
    metric_goal: str = "maximize"  # The wandb optimization goal
    name: str | None = None  # The (optional) name of the wandb sweep
    description: str | None = None  # The (optional) description of the wandb sweep
    algorithm: Literal["stream_sarsa", "stream_q", "stream_ac", "ppo"] = (
        "stream_q"  # The type of RL algorithm to run.
    )
    with_baselines: bool = False  # Flag to compute plots comparing against baseline (Expensive, default is False)
    eval_freq: int = 2_500  # Frequency in # training steps with which to call the evaluation function

    # Derived object, getting the actual RL algorithm's config
    @property
    def algorithm_config(self) -> StreamSarsaConfig | StreamQConfig | StreamACConfig:
        return getattr(self, self.algorithm)

    def to_wandb(self) -> Dict:
        config = {
            "method": self.method,
            "metric": {
                "name": self.metric_name,
                "goal": self.metric_goal,
            },
            "name": self.name,
            "parameters": {
                "algorithm": {"value": self.algorithm},
                "env": self.env.to_wandb(),
                "agent": self.algorithm_config.to_wandb(),
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
    num_envs_per_eval: int = 8  # Number of environment initializations per evaluation
    num_configs: int = 1  # Number of random agent configurations to run
    dataset: Literal["mnist", "california"] = "mnist"  # Dataset on which to privatise
    dataset_poly_d: int | None = None  # Degree of polynomial features to be generated
    total_timesteps: int = 2_000_000  # Training steps of RL algorithm
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
