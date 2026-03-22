from typing import Self

import equinox as eqx
from jaxtyping import Array

from policy.base_schedules.abstract import AbstractSchedule
from policy.base_schedules.factory import base_schedule_factory
from policy.schedules._registry import register
from policy.schedules.abstract import AbstractNoiseAndClipSchedule
from policy.schedules.config import ParallelSigmaAndClipScheduleConfig
from privacy.gdp_privacy import GDPPrivacyParameters


@register(ParallelSigmaAndClipScheduleConfig)
class ParallelSigmaAndClipSchedule(AbstractNoiseAndClipSchedule):
    """Jointly optimise sigma and clip with Euclidean projection onto the GDP constraint.

    Unlike ``SigmaAndClipSchedule``, which projects only the noise schedule while
    keeping clips fixed, this schedule projects both sigma and clip simultaneously
    via the nearest-point (L2) projection onto
        sum_i (exp((clip_i / sigma_i)^2) - 1) <= (mu/p)^2.
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
        conf: ParallelSigmaAndClipScheduleConfig,
        privacy_params: GDPPrivacyParameters,
    ) -> "ParallelSigmaAndClipSchedule":
        T = privacy_params.T
        noise_schedule = base_schedule_factory(conf.noise, T)
        clip_schedule = base_schedule_factory(conf.clip, T)
        return cls(noise_schedule, clip_schedule, privacy_params)

    def get_private_sigmas(self) -> Array:
        return self.noise_schedule.get_valid_schedule().squeeze()

    def get_private_clips(self) -> Array:
        return self.clip_schedule.get_valid_schedule().squeeze()

    def get_private_weights(self) -> Array:
        private_sigmas = self.get_private_sigmas()
        clips = self.get_private_clips()
        return self.privacy_params.project_weights(clips / private_sigmas).squeeze()

    def apply_updates(self, updates) -> Self:
        return eqx.apply_updates(self, updates)

    @eqx.filter_jit
    def project(self) -> Self:
        sigmas = self.get_private_sigmas()
        clips = self.get_private_clips()

        proj_sigmas, proj_clips = self.privacy_params.project_sigma_and_clip(sigmas, clips)

        new_noise_schedule = self.noise_schedule.__class__.from_projection(
            self.noise_schedule, proj_sigmas
        )
        new_clip_schedule = self.clip_schedule.__class__.from_projection(
            self.clip_schedule, proj_clips
        )

        return self.__class__(
            noise_schedule=new_noise_schedule,
            clip_schedule=new_clip_schedule,
            privacy_params=self.privacy_params,
        )

    def _get_log_arrays(self) -> dict[str, Array]:
        return {
            "sigmas": self.get_private_sigmas(),
            "clips": self.get_private_clips(),
            "mus": self.get_private_weights(),
        }
