import dataclasses
from dataclasses import dataclass

from conf.config_util import to_wandb_conf, to_wandb_sweep_params
from policy.base_schedules.config import (
    BaseScheduleConfig,
    BSplineScheduleConfig,
    InterpolatedExponentialScheduleConfig,
)


class AbstractNoiseAndClipScheduleConfig:
    pass

    def to_wandb_conf(self) -> dict[str, object]:
        return to_wandb_conf(self)


@dataclass
class SigmaAndClipScheduleConfig(AbstractNoiseAndClipScheduleConfig):
    noise: BaseScheduleConfig = dataclasses.field(
        default_factory=InterpolatedExponentialScheduleConfig,
    )
    clip: BaseScheduleConfig = dataclasses.field(
        default_factory=InterpolatedExponentialScheduleConfig,
    )

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)


@dataclass
class DecoupledSigmaAndClipScheduleConfig(AbstractNoiseAndClipScheduleConfig):
    """Noise σ = C · (1/w) with C fully decoupled from the privacy budget.

    The w-side base schedule's ``get_valid_schedule()`` returns w directly
    (length T); the C-side base schedule provides the per-step clip threshold.
    """

    noise: BaseScheduleConfig = dataclasses.field(
        default_factory=BSplineScheduleConfig,
    )
    clip: BaseScheduleConfig = dataclasses.field(
        default_factory=BSplineScheduleConfig,
    )

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)


@dataclass
class DynamicDPSGDScheduleConfig(AbstractNoiseAndClipScheduleConfig):
    rho_mu: float = 0.5
    rho_c: float = 0.5
    c_0: float = 1.5

    def to_wandb_sweep(self):
        return to_wandb_sweep_params(self)
