from typing import Self

import equinox as eqx
from jaxtyping import Array

from conf.singleton_conf import SingletonConfig
from policy.base_schedules.abstract import AbstractSchedule
from policy.base_schedules.factory import base_schedule_factory
from policy.schedules.abstract import AbstractNoiseAndClipSchedule
from policy.schedules.config import SigmaAndClipScheduleConfig
from privacy.gdp_privacy import GDPPrivacyParameters
from util.logger import Loggable, LoggableArray, LoggingSchema
from util.util import pytree_has_inf


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

    @classmethod
    def from_config(
        cls, conf: SigmaAndClipScheduleConfig, privacy_params: GDPPrivacyParameters
    ) -> "SigmaAndClipSchedule":
        noise_schedule = base_schedule_factory(conf.noise)
        clip_schedule = base_schedule_factory(conf.clip)

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
        private_weights = self.get_private_weights()
        private_clips = self.get_private_clips()

        new_noises = private_clips / private_weights
        new_noise_schedule = self.noise_schedule.__class__.from_projection(
            self.noise_schedule, new_noises
        )

        return self.__class__(
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
                table_name="mus",
                array=self.get_private_weights(),
                plot=True,
                force=force,
            ),
        ]
