from abc import abstractmethod
import equinox as eqx
from jaxtyping import Array
from privacy.gdp_privacy import GDPPrivacyParameters
from privacy.base_schedules import AbstractSchedule
from util.logger import Loggable, LoggableArray, LoggingSchema
from conf.singleton_conf import SingletonConfig


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

    @classmethod
    @abstractmethod
    def project(
        cls, schedule: "AbstractNoiseAndClipSchedule"
    ) -> "AbstractNoiseAndClipSchedule":
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

        weights = self.privacy_params.sigma_schedule_to_weights(clips, private_sigmas)
        proj_weights = self.privacy_params.project_weights(weights)
        return proj_weights.squeeze()

    @classmethod
    def project(cls, schedule: "SigmaAndClipSchedule") -> "SigmaAndClipSchedule":
        if not isinstance(schedule, SigmaAndClipSchedule):
            raise ValueError(
                "Input schedule must be an instance of SigmaAndClipSchedule."
            )
        private_weights = schedule.get_private_weights()
        private_clips = schedule.get_private_clips()

        new_noises = schedule.privacy_params.weights_to_sigma_schedule(
            private_clips, private_weights
        )

        new_noise_schedule = schedule.noise_schedule.__class__.from_projection(
            schedule.noise_schedule, new_noises
        )

        return SigmaAndClipSchedule(
            noise_schedule=new_noise_schedule,
            clip_schedule=schedule.clip_schedule,
            privacy_params=schedule.privacy_params,
        )

    def get_logging_schemas(self) -> list[LoggingSchema]:
        plot_interval = SingletonConfig.get_sweep_config_instance().plotting_interval
        return [
            LoggingSchema(
                table_name="sigmas",
                cols=[str(step) for step in range(len(self.get_private_sigmas()))],
                freq=plot_interval,
            ),
            LoggingSchema(
                table_name="clips",
                cols=[str(step) for step in range(len(self.get_private_clips()))],
                freq=plot_interval,
            ),
            LoggingSchema(
                table_name="weights",
                cols=[str(step) for step in range(len(self.get_private_weights()))],
                freq=plot_interval,
            ),
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

    @classmethod
    def project(cls, schedule: "PolicyAndClipSchedule") -> "PolicyAndClipSchedule":
        if not isinstance(schedule, PolicyAndClipSchedule):
            raise ValueError(
                "Input schedule must be an instance of PolicyAndClipSchedule."
            )

        private_weights = schedule.get_private_weights()
        new_policy_schedule = schedule.policy_schedule.__class__.from_projection(
            schedule.policy_schedule, private_weights
        )

        return PolicyAndClipSchedule(
            policy_schedule=new_policy_schedule,
            clip_schedule=schedule.clip_schedule,
            privacy_params=schedule.privacy_params,
        )
    
    def get_logging_schemas(self) -> list[LoggingSchema]:
        return [
            LoggingSchema(
                table_name="sigmas",
                cols=[str(step) for step in range(len(self.get_private_sigmas()))],
            ),
            LoggingSchema(
                table_name="clips",
                cols=[str(step) for step in range(len(self.get_private_clips()))],
            ),
            LoggingSchema(
                table_name="weights",
                cols=[str(step) for step in range(len(self.get_private_weights()))],
             ),
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
        ]
