from dataclasses import dataclass

import jax.numpy as jnp
from tyro.conf import Fixed

from conf.config_util import to_wandb_sweep_params
from networks.mlp.config import MLPConfig


@dataclass
class CNNConfig:
    """Configuration for Convolutional Neural Network"""

    # linear config params
    mlp: MLPConfig

    # conv config params
    channels: tuple[int, ...] = (16, 32)  # Number of channels in each conv layer
    nchannels: int = -1  # Number of input channels, derived from data
    kernel_sizes: tuple[int, ...] = (8, 4)  # Kernel sizes for each conv layer
    paddings: tuple[int, ...] = (2, 0)  # Padding for each conv layer
    strides: tuple[int, ...] = (2, 2)  # Stride
    pool_kernel_size: int = 2  # Edge length of pooling kernel
    key: Fixed[int] = 0  # Overridden as derivative from experiment.env_prng_key

    # dummy item, used to determine MLP input shape
    dummy_data: jnp.ndarray | None = None

    attrs: Fixed[tuple[str, ...]] = (
        "nchannels",
        "kernel_size",
        "pool_kernel_size",
        "hidden_channels",
        "nhidden_conv",
    )

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)
