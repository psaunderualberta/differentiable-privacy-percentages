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
class AlternatingSigmaAndClipScheduleConfig(AbstractNoiseAndClipScheduleConfig):
    noise: BaseScheduleConfig = dataclasses.field(
        default_factory=InterpolatedExponentialScheduleConfig,
    )
    clip: BaseScheduleConfig = dataclasses.field(
        default_factory=InterpolatedExponentialScheduleConfig,
    )
    diff_clips_first: bool = False

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)


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
class PolicyAndClipScheduleConfig(AbstractNoiseAndClipScheduleConfig):
    policy: BaseScheduleConfig = dataclasses.field(
        default_factory=InterpolatedExponentialScheduleConfig,
    )
    clip: BaseScheduleConfig = dataclasses.field(
        default_factory=InterpolatedExponentialScheduleConfig,
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


@dataclass
class WarmupAlternatingSigmaAndClipScheduleConfig(AbstractNoiseAndClipScheduleConfig):
    noise_tail: BaseScheduleConfig = dataclasses.field(
        default_factory=InterpolatedExponentialScheduleConfig,
    )
    clip_tail: BaseScheduleConfig = dataclasses.field(
        default_factory=InterpolatedExponentialScheduleConfig,
    )
    warmup_noise_init: float = 1.0
    warmup_clip_init: float = 1.0
    warmup_pct: float = 0.3
    diff_clips_first: bool = False

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)


@dataclass
class ParallelSigmaAndClipScheduleConfig(AbstractNoiseAndClipScheduleConfig):
    noise: BaseScheduleConfig = dataclasses.field(
        default_factory=InterpolatedExponentialScheduleConfig,
    )
    clip: BaseScheduleConfig = dataclasses.field(
        default_factory=InterpolatedExponentialScheduleConfig,
    )
    use_fista: bool = False

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)


@dataclass
class WarmupSigmaAndClipScheduleConfig(AbstractNoiseAndClipScheduleConfig):
    noise_tail: BaseScheduleConfig = dataclasses.field(
        default_factory=InterpolatedExponentialScheduleConfig,
    )
    clip_tail: BaseScheduleConfig = dataclasses.field(
        default_factory=InterpolatedExponentialScheduleConfig,
    )
    warmup_noise_init: float = 1.0
    warmup_clip_init: float = 1.0
    warmup_pct: float = 0.3

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)


@dataclass
class WarmupParallelSigmaAndClipScheduleConfig(AbstractNoiseAndClipScheduleConfig):
    noise_tail: BaseScheduleConfig = dataclasses.field(
        default_factory=BSplineScheduleConfig,
    )
    clip_tail: BaseScheduleConfig = dataclasses.field(
        default_factory=BSplineScheduleConfig,
    )
    warmup_noise_init: float = 1.0
    warmup_clip_init: float = 1.0
    warmup_pct: float = 0.1
    use_fista: bool = False

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)
