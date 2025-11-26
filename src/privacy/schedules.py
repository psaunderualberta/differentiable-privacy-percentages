from abc import abstractmethod
import equinox as eqx
from jaxtyping import Array
from privacy.gdp_privacy import GDPPrivacyParameters
from privacy.base_schedules import AbstractSchedule


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
        clips = self.clip_schedule.get_valid_schedule()
        noises = self.noise_schedule.get_valid_schedule()

        weights = self.privacy_params.sigma_schedule_to_weights(clips, noises)
        proj_weights = self.privacy_params.project_weights(weights)
        private_sigmas = self.privacy_params.weights_to_sigma_schedule(
            clips, proj_weights
        )
        return private_sigmas.squeeze()

    def get_private_clips(self) -> Array:
        return self.clip_schedule.get_valid_schedule().squeeze()

    def get_private_weights(self) -> Array:
        private_sigmas = self.get_private_sigmas()
        clips = self.get_private_clips()

        weights = self.privacy_params.sigma_schedule_to_weights(clips, private_sigmas)
        return weights.squeeze()


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

        proj_weights = self.privacy_params.project_weights(policies)
        private_sigmas = self.privacy_params.weights_to_sigma_schedule(
            clips, proj_weights
        )
        return private_sigmas.squeeze()

    def get_private_clips(self) -> Array:
        return self.clip_schedule.get_valid_schedule().squeeze()

    def get_private_weights(self) -> Array:
        weights = self.policy_schedule.get_valid_schedule()
        proj_weights = self.privacy_params.project_weights(weights)
        return proj_weights.squeeze()
