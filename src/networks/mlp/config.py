from dataclasses import dataclass
from typing import Literal

from conf.config_util import to_wandb_sweep_params


@dataclass
class MLPConfig:
    """Configuration for Multi-Layer Perceptron.

    Fields that depend on the dataset (``din``, ``nclasses``) and the
    experiment seed (``key``) are passed explicitly to ``MLP.from_config``
    rather than embedded as sentinel values here.
    """

    hidden_sizes: tuple[int, ...] = (32,)
    initialization: Literal["glorot", "zeros"] = "glorot"

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)
