from dataclasses import dataclass
from typing import Literal


# ---
# Configs for different private networks
# ---
@dataclass
class MLPConfig:
    """Configuration for Multi-Layer Perceptron"""

    din: int = -1  # Value is derived from data
    hidden_sizes: tuple[int, ...] = (32,)  # Size of hidden layers
    nclasses: int = -1  # Value is derived from data
    initialization: Literal["glorot", "zeros"] = "glorot"
    key: int = 0  # Overridden as derivative from experiment.env_prng_key

    def to_wandb_sweep(self) -> dict[str, object]:
        attrs = [
            "din",
            "dhidden",
            "nhidden",
            "initialization",
            "nclasses",
        ]
        return {"parameters": {attr: {"value": getattr(self, attr)} for attr in attrs}}
