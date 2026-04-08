from dataclasses import dataclass

from conf.config_util import to_wandb_conf, to_wandb_sweep_params


@dataclass
class AutoNetworkConfig:
    """Automatically selects network architecture based on the configured dataset."""

    def to_wandb_sweep(self) -> dict:
        return to_wandb_sweep_params(self)

    def to_wandb_conf(self) -> dict[str, object]:
        return to_wandb_conf(self)
