import equinox as eqx
from jaxtyping import Array

from policy.base_schedules.abstract import AbstractSchedule
from policy.base_schedules.factory import base_schedule_factory
from policy.schedules.abstract import AbstractNoiseAndClipSchedule
from policy.schedules.config import PolicyAndClipScheduleConfig
from privacy.gdp_privacy import GDPPrivacyParameters
from util.logger import Loggable, LoggableArray, LoggingSchema


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
        self.policy_schedule = policy_schedule
        self.clip_schedule = clip_schedule
        self.privacy_params = privacy_params

    @classmethod
    def from_config(
        cls, conf: PolicyAndClipScheduleConfig, privacy_params: GDPPrivacyParameters
    ) -> "PolicyAndClipSchedule":
        noise_schedule = base_schedule_factory(conf.noise)
        clip_schedule = base_schedule_factory(conf.clip)

        return cls(noise_schedule, clip_schedule, privacy_params)

    def get_private_sigmas(self) -> Array:
        clips = self.clip_schedule.get_valid_schedule()
        policies = self.policy_schedule.get_valid_schedule()

        private_sigmas = self.privacy_params.weights_to_sigma_schedule(clips, policies)
        return private_sigmas.squeeze()

    def get_private_clips(self) -> Array:
        return self.clip_schedule.get_valid_schedule().squeeze()

    def get_private_weights(self) -> Array:
        weights = self.policy_schedule.get_valid_schedule()
        proj_weights = self.privacy_params.project_weights(weights)
        return proj_weights.squeeze()

    @eqx.filter_jit
    def project(self) -> "PolicyAndClipSchedule":
        private_weights = self.get_private_weights()
        new_policy_schedule = self.policy_schedule.__class__.from_projection(
            self.policy_schedule, private_weights
        )

        return PolicyAndClipSchedule(
            policy_schedule=new_policy_schedule,
            clip_schedule=self.clip_schedule,
            privacy_params=self.privacy_params,
        )

    def get_logging_schemas(self) -> list[LoggingSchema]:
        col_names = [str(step) for step in range(len(self.get_private_sigmas()))]
        return [
            LoggingSchema(table_name="sigmas", cols=col_names),
            LoggingSchema(table_name="clips", cols=col_names),
            LoggingSchema(table_name="weights", cols=col_names),
            LoggingSchema(table_name="mus", cols=col_names),
        ]

    def get_loggables(self, force=False) -> list[Loggable | LoggableArray]:
        return [
            LoggableArray(
                table_name="sigmas",
                array=self.get_private_sigmas(),
                plot=True,
                force=force,
            ),
            LoggableArray(
                table_name="clips",
                array=self.get_private_clips(),
                plot=True,
                force=force,
            ),
            LoggableArray(
                table_name="weights",
                array=self.get_private_weights(),
                plot=True,
                force=force,
            ),
            LoggableArray(
                table_name="mus",
                array=self.privacy_params.weights_to_mu_schedule(
                    self.get_private_weights()
                ),
                plot=True,
                force=force,
            ),
        ]
