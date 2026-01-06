from dataclasses import dataclass
from typing import Literal

from tyro.conf import Fixed

from conf.config_util import to_wandb_sweep_params


@dataclass
class MLPConfig:
    """Configuration for Multi-Layer Perceptron"""

    din: int = -1  # Value is derived from data
    hidden_sizes: tuple[int, ...] = (32,)  # Size of hidden layers
    nclasses: int = -1  # Value is derived from data
    initialization: Literal["glorot", "zeros"] = "glorot"
    key: Fixed[int] = 0  # Overridden as derivative from experiment.env_prng_key
    attrs: Fixed[tuple[str, ...]] = (
        "din",
        "dhidden",
        "nhidden",
        "initialization",
        "nclasses",
    )

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)
