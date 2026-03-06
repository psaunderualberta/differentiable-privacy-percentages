from typing import Self

import equinox as eqx
from jaxtyping import Array

from policy.base_schedules.abstract import AbstractSchedule
from policy.base_schedules.factory import base_schedule_factory
from policy.schedules._registry import register
from policy.schedules.abstract import AbstractNoiseAndClipSchedule
from policy.schedules.config import PolicyAndClipScheduleConfig
from privacy.gdp_privacy import GDPPrivacyParameters


@register(PolicyAndClipScheduleConfig)
class PolicyAndClipSchedule(AbstractNoiseAndClipSchedule):
    policy_schedule: AbstractSchedule
    clip_schedule: AbstractSchedule
    privacy_params: GDPPrivacyParameters

    def __init__(
        self,
        policy_schedule: AbstractSchedule,
        clip_schedule: AbstractSchedule,
        privacy_params: GDPPrivacyParameters,
    ):
        """Initialise the schedule with a policy (weight) sub-schedule and a clip sub-schedule.

        Args:
            policy_schedule: Parametric schedule whose outputs are GDP weight values.
            clip_schedule: Parametric schedule that produces per-step clip thresholds.
            privacy_params: GDP privacy budget and subsampling parameters.
        """
        self.policy_schedule = policy_schedule
        self.clip_schedule = clip_schedule
        self.privacy_params = privacy_params

    @classmethod
    def from_config(
        cls,
        conf: PolicyAndClipScheduleConfig,
        privacy_params: GDPPrivacyParameters,
    ) -> "PolicyAndClipSchedule":
        T = privacy_params.T
        policy_schedule = base_schedule_factory(conf.policy, T)
        clip_schedule = base_schedule_factory(conf.clip, T)
        return cls(policy_schedule, clip_schedule, privacy_params)

    def get_private_sigmas(self) -> Array:
        clips = self.clip_schedule.get_valid_schedule()
        policies = self.policy_schedule.get_valid_schedule()
        return self.privacy_params.weights_to_sigma_schedule(clips, policies).squeeze()

    def get_private_clips(self) -> Array:
        return self.clip_schedule.get_valid_schedule().squeeze()

    def get_private_weights(self) -> Array:
        weights = self.policy_schedule.get_valid_schedule()
        return self.privacy_params.project_weights(weights).squeeze()

    def apply_updates(self, updates) -> Self:
        return eqx.apply_updates(self, updates)

    @eqx.filter_jit
    def project(self) -> Self:
        private_weights = self.get_private_weights()
        new_policy_schedule = self.policy_schedule.__class__.from_projection(
            self.policy_schedule,
            private_weights,
        )
        return self.__class__(
            policy_schedule=new_policy_schedule,
            clip_schedule=self.clip_schedule,
            privacy_params=self.privacy_params,
        )

    def _get_log_arrays(self) -> dict[str, Array]:
        weights = self.get_private_weights()
        return {
            "sigmas": self.get_private_sigmas(),
            "clips": self.get_private_clips(),
            "weights": weights,
            "mus": self.privacy_params.weights_to_mu_schedule(weights),
        }
