from dataclasses import dataclass
from typing import Annotated, Literal, Union

import tyro

from conf.config_util import to_wandb_sweep_params
from policy.base_schedules.config import (
    ConstantScheduleConfig,
    InterpolatedClippedScheduleConfig,
    InterpolatedExponentialScheduleConfig,
)


class AbstractNoiseAndClipScheduleConfig:
    pass


@dataclass
class SigmaAndClipScheduleConfig(AbstractNoiseAndClipScheduleConfig):
    noise_constant: ConstantScheduleConfig
    noise_clipped: InterpolatedClippedScheduleConfig
    noise_exponential: InterpolatedExponentialScheduleConfig
    clip_constant: ConstantScheduleConfig
    clip_clipped: InterpolatedClippedScheduleConfig
    clip_exponential: InterpolatedExponentialScheduleConfig
    noise_type: Literal["noise_constant", "noise_clipped", "noise_exponential"] = (
        "noise_exponential"
    )
    clip_type: Literal["clip_constant", "clip_clipped", "clip_exponential"] = (
        "clip_exponential"
    )
    attrs: tyro.conf.Fixed[tuple[str, ...]] = (
        "noise_constant",
        "noise_clipped",
        "noise_exponential",
        "clip_constant",
        "clip_clipped",
        "clip_exponential",
        "noise_type",
        "clip_type",
    )

    @property
    def noise(self):
        return getattr(self, self.noise_type)

    @property
    def clip(self):
        return getattr(self, self.clip_type)

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)


@dataclass
class PolicyAndClipScheduleConfig(AbstractNoiseAndClipScheduleConfig):
    policy_constant: ConstantScheduleConfig
    policy_clipped: InterpolatedClippedScheduleConfig
    policy_exponential: InterpolatedExponentialScheduleConfig
    clip_constant: ConstantScheduleConfig
    clip_clipped: InterpolatedClippedScheduleConfig
    clip_exponential: InterpolatedExponentialScheduleConfig
    policy_type: Literal["policy_constant", "policy_clipped", "policy_exponential"] = (
        "policy_exponential"
    )
    clip_type: Literal["clip_constant", "clip_clipped", "clip_exponential"] = (
        "clip_exponential"
    )
    attrs: tyro.conf.Fixed[tuple[str, ...]] = (
        "policy_constant",
        "policy_clipped",
        "policy_exponential",
        "clip_constant",
        "clip_clipped",
        "clip_exponential",
        "policy_type",
        "clip_type",
    )

    @property
    def policy(self):
        return getattr(self, self.policy_type)

    @property
    def clip(self):
        return getattr(self, self.clip_type)

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)


@dataclass
class AlternatingSigmaAndClipScheduleConfig(AbstractNoiseAndClipScheduleConfig):
    noise_constant: ConstantScheduleConfig
    noise_clipped: InterpolatedClippedScheduleConfig
    noise_exponential: InterpolatedExponentialScheduleConfig
    clip_constant: ConstantScheduleConfig
    clip_clipped: InterpolatedClippedScheduleConfig
    clip_exponential: InterpolatedExponentialScheduleConfig
    noise_type: Literal["noise_constant", "noise_clipped", "noise_exponential"] = (
        "noise_exponential"
    )
    clip_type: Literal["clip_constant", "clip_clipped", "clip_exponential"] = (
        "clip_exponential"
    )
    attrs: tyro.conf.Fixed[tuple[str, ...]] = (
        "noise_constant",
        "noise_clipped",
        "noise_exponential",
        "clip_constant",
        "clip_clipped",
        "clip_exponential",
        "noise_type",
        "clip_type",
    )

    diff_clips_first: bool = False

    @property
    def noise(self):
        return getattr(self, self.noise_type)

    @property
    def clip(self):
        return getattr(self, self.clip_type)

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)


@dataclass
class DynamicDPSGDScheduleConfig(AbstractNoiseAndClipScheduleConfig):
    rho_mu: float = 0.5
    rho_c: float = 0.5
    c_0: float = 1.5
    attrs: tyro.conf.Fixed[tuple[str, ...]] = ("rho_mu", "rho_c", "c_0")

    def to_wandb_sweep(self):
        return to_wandb_sweep_params(self)
