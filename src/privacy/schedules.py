from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array

from conf.singleton_conf import SingletonConfig
from privacy.base_schedules import AbstractSchedule
from privacy.gdp_privacy import GDPPrivacyParameters
from util.logger import Loggable, LoggableArray, LoggingSchema
from util.util import pytree_has_inf


class AbstractNoiseAndClipSchedule(eqx.Module):
    @abstractmethod
    def get_private_sigmas(self) -> Array:
        raise NotImplementedError(
            "Subclasses must implement get_private_sigmas method."
        )

    @abstractmethod
    def get_private_clips(self) -> Array:
        raise NotImplementedError("Subclasses must implement get_private_clips method.")

    @abstractmethod
    def get_private_weights(self) -> Array:
        raise NotImplementedError(
            "Subclasses must implement get_private_weights method."
        )

    @abstractmethod
    def project(self) -> "AbstractNoiseAndClipSchedule":
        raise NotImplementedError("Subclasses must implement 'project' class method.")

    @abstractmethod
    def get_logging_schemas(self) -> list[LoggingSchema]:
        raise NotImplementedError(
            "Subclasses must implement get_logging_schemas method."
        )

    @abstractmethod
    def get_loggables(self, force=False) -> list[Loggable | LoggableArray]:
        raise NotImplementedError("Subclasses must implement get_loggables method.")


class SigmaAndClipSchedule(AbstractNoiseAndClipSchedule):
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

    def get_private_sigmas(self) -> Array:
        return self.noise_schedule.get_valid_schedule().squeeze()

    def get_private_clips(self) -> Array:
        return self.clip_schedule.get_valid_schedule().squeeze()

    def get_private_weights(self) -> Array:
        private_sigmas = self.get_private_sigmas()
        clips = self.get_private_clips()

        private_sigmas = eqx.error_if(
            private_sigmas, pytree_has_inf(private_sigmas), "private_sigmas have Inf!"
        )
        private_sigmas = eqx.error_if(
            private_sigmas, (private_sigmas == 0).any(), "private_sigmas has 0!"
        )
        weights = self.privacy_params.sigma_schedule_to_weights(clips, private_sigmas)
        weights = eqx.error_if(weights, pytree_has_inf(weights), "weights1 have Inf!")
        proj_weights = self.privacy_params.project_weights(weights)
        proj_weights = eqx.error_if(
            proj_weights, pytree_has_inf(proj_weights), "weights2 have Inf!"
        )
        return proj_weights.squeeze()

    @eqx.filter_jit
    def project(self) -> "SigmaAndClipSchedule":
        private_weights = self.get_private_weights()
        private_clips = self.get_private_clips()

        new_noises = self.privacy_params.weights_to_sigma_schedule(
            private_clips, private_weights
        )

        new_noise_schedule = self.noise_schedule.__class__.from_projection(
            self.noise_schedule, new_noises
        )

        return SigmaAndClipSchedule(
            noise_schedule=new_noise_schedule,
            clip_schedule=self.clip_schedule,
            privacy_params=self.privacy_params,
        )

    def get_logging_schemas(self) -> list[LoggingSchema]:
        plot_interval = SingletonConfig.get_sweep_config_instance().plotting_interval
        col_names = [str(step) for step in range(len(self.get_private_sigmas()))]
        return [
            LoggingSchema(table_name="sigmas", cols=col_names, freq=plot_interval),
            LoggingSchema(table_name="clips", cols=col_names, freq=plot_interval),
            LoggingSchema(table_name="weights", cols=col_names, freq=plot_interval),
            LoggingSchema(table_name="mus", cols=col_names, freq=plot_interval),
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
