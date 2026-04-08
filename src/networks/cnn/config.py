import dataclasses
from dataclasses import dataclass

from conf.config_util import to_wandb_conf, to_wandb_sweep_params
from networks.mlp.config import MLPConfig


@dataclass
class CNNConfig:
    """Configuration for Convolutional Neural Network.

    Fields that depend on the dataset (``nchannels``, ``dummy_data``) and the
    experiment seed (``key``) are passed explicitly to ``CNN.from_config``
    rather than embedded as sentinel values here.
    """

    mlp: MLPConfig = dataclasses.field(default_factory=MLPConfig)
    channels: tuple[int, ...] = (16, 32)
    kernel_sizes: tuple[int, ...] = (8, 4)
    paddings: tuple[int, ...] = (2, 0)
    strides: tuple[int, ...] = (2, 2)
    pool_kernel_size: int = 2

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)

    def to_wandb_conf(self) -> dict[str, object]:
        return to_wandb_conf(self)
