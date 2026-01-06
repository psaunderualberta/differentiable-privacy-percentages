from dataclasses import dataclass


@dataclass
class MedianGradientScheduleConfig:
    """Configuration for Median Gradient Schedule"""

    c_0: float = 0.1  # Initial clipping value
    eta_c: float = 0.2  # Learning rate for median clip value

    def to_wandb_sweep(self) -> dict[str, object]:
        attrs = [
            "c_0",
            "eta_c",
        ]
        return {"parameters": {attr: {"value": getattr(self, attr)} for attr in attrs}}
