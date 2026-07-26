from dataclasses import dataclass

from conf.config_util import to_wandb_conf, to_wandb_sweep_params


class AbstractStatefulScheduleConfig:
    def to_wandb_conf(self):
        return to_wandb_conf(self)


@dataclass
class StatefulMedianGradientNoiseAndClipConfig(AbstractStatefulScheduleConfig):
    c_0: float = 0.1
    eta_c: float = 0.2
    r: float = 20.0
    """Count noise ratio: the count release's Gaussian noise is L/r for the expected
    batch size L, fixing the within-clip fraction's standard error at 1/r in every
    privacy regime. Andrew et al. (NeurIPS 2021) use 20. Not swept — the derived
    median budget fraction ρ follows from it and the regime (ADR 0013)."""

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)
