from typing import Self

import equinox as eqx
from jax import lax as jlax
from jaxtyping import Array

from policy.base_schedules.abstract import AbstractSchedule
from policy.base_schedules.factory import base_schedule_factory
from policy.schedules._registry import register
from policy.schedules.abstract import AbstractNoiseAndClipSchedule
from policy.schedules.config import DecoupledSigmaAndClipScheduleConfig
from privacy.gdp_privacy import GDPPrivacyParameters


@register(DecoupledSigmaAndClipScheduleConfig)
class DecoupledSigmaAndClipSchedule(AbstractNoiseAndClipSchedule):
    """Noise std σ_noise = C · (1/w), with C fully decoupled from the privacy budget.

    The noise-side base schedule's ``get_valid_schedule()`` returns w directly
    (the privacy-constraint variable); ``get_private_noise_scales()`` returns
    C · σ_mult = C · (1/w) at the dp.py boundary. ``project()`` projects only
    the w-side onto the GDP constraint Σ exp(w²) ≤ (μ/p)² + T; C is untouched.
    """

    noise_schedule: AbstractSchedule
    clip_schedule: AbstractSchedule
    privacy_params: GDPPrivacyParameters

    def __init__(
        self,
        noise_schedule: AbstractSchedule,
        clip_schedule: AbstractSchedule,
        privacy_params: GDPPrivacyParameters,
    ):
        self.noise_schedule = noise_schedule
        self.clip_schedule = clip_schedule
        self.privacy_params = privacy_params

    @classmethod
    def from_config(
        cls,
        conf: DecoupledSigmaAndClipScheduleConfig,
        privacy_params: GDPPrivacyParameters,
    ) -> "DecoupledSigmaAndClipSchedule":
        T = privacy_params.T
        noise_schedule = base_schedule_factory(conf.noise, T)
        clip_schedule = base_schedule_factory(conf.clip, T)
        return cls(noise_schedule, clip_schedule, privacy_params)

    def get_private_noise_scales(self) -> Array:
        return self.get_private_clips() / self.get_private_weights()

    def get_private_clips(self) -> Array:
        return self.clip_schedule.get_valid_schedule().squeeze()

    def _get_private_sigmas(self):
        return self.noise_schedule.get_valid_schedule().squeeze()

    def get_private_weights(self) -> Array:
        return 1 / self._get_private_sigmas()

    def apply_updates(self, updates) -> Self:
        return eqx.apply_updates(self, updates)

    @eqx.filter_jit
    def project(self) -> Self:
        s_proj = self.privacy_params.project_inverse_sigmas(self._get_private_sigmas())
        new_noise_schedule = self.noise_schedule.__class__.from_projection(
            self.noise_schedule,
            s_proj,
        )
        return self.__class__(
            noise_schedule=new_noise_schedule,
            clip_schedule=self.clip_schedule,
            privacy_params=self.privacy_params,
        )

    def _get_log_arrays(self) -> dict[str, Array]:
        return {
            "sigmas": self.get_private_noise_scales(),
            "clips": self.get_private_clips(),
            "mus": self.get_private_weights(),
            "multipliers": self._get_private_sigmas(),
        }
