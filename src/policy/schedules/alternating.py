from typing import Self

import equinox as eqx
import jax.lax as jlax
import jax.numpy as jnp
import jax.tree as jtree
from jaxtyping import Array

from conf.singleton_conf import SingletonConfig
from policy.base_schedules.abstract import AbstractSchedule
from policy.base_schedules.factory import base_schedule_factory
from policy.schedules._registry import register
from policy.schedules.abstract import AbstractNoiseAndClipSchedule
from policy.schedules.config import AlternatingSigmaAndClipScheduleConfig
from privacy.gdp_privacy import GDPPrivacyParameters
from util.logger import Loggable, LoggableArray, LoggingSchema


@register(AlternatingSigmaAndClipScheduleConfig)
class AlternatingSigmaAndClipSchedule(AbstractNoiseAndClipSchedule):
    noise_schedule: AbstractSchedule
    clip_schedule: AbstractSchedule
    privacy_params: GDPPrivacyParameters
    diff_clips: Array

    def __init__(
        self,
        noise_schedule: AbstractSchedule,
        clip_schedule: AbstractSchedule,
        privacy_params: GDPPrivacyParameters,
        diff_clips: bool | Array = False,
    ):
        self.noise_schedule = noise_schedule
        self.clip_schedule = clip_schedule
        self.privacy_params = privacy_params
        self.diff_clips = jnp.asarray(diff_clips)

    @classmethod
    def from_config(
        cls,
        conf: AlternatingSigmaAndClipScheduleConfig,
        privacy_params: GDPPrivacyParameters,
    ) -> "AlternatingSigmaAndClipSchedule":
        T = privacy_params.T
        noise_schedule = base_schedule_factory(conf.noise, T)
        clip_schedule = base_schedule_factory(conf.clip, T)
        return cls(noise_schedule, clip_schedule, privacy_params, conf.diff_clips_first)

    def __diff_clips_select(self, a, b):
        def tree_select(a, b):
            if a is None:
                return a
            return jlax.select(self.diff_clips, a, b)

        return jtree.map(tree_select, a, b)

    def get_private_sigmas(self) -> Array:
        sigmas = self.noise_schedule.get_valid_schedule().squeeze()
        return jlax.select(self.diff_clips, jlax.stop_gradient(sigmas), sigmas)

    def get_private_clips(self) -> Array:
        clips = self.clip_schedule.get_valid_schedule().squeeze()
        return jlax.select(self.diff_clips, clips, jlax.stop_gradient(clips))

    def get_private_weights(self) -> Array:
        private_sigmas = self.get_private_sigmas()
        clips = self.get_private_clips()
        return self.privacy_params.project_weights(clips / private_sigmas).squeeze()

    def apply_updates(self, updates) -> Self:
        updated_noise = eqx.apply_updates(self.noise_schedule, updates.noise_schedule)
        updated_clips = eqx.apply_updates(self.clip_schedule, updates.clip_schedule)

        new_noise = self.__diff_clips_select(self.noise_schedule, updated_noise)
        new_clips = self.__diff_clips_select(updated_clips, self.clip_schedule)

        return self.__class__(
            noise_schedule=new_noise,
            clip_schedule=new_clips,
            privacy_params=self.privacy_params,
            diff_clips=self.diff_clips,
        )

    @eqx.filter_jit
    def project(self) -> Self:
        private_weights = self.get_private_weights()
        private_clips = self.get_private_clips()
        private_sigmas = self.get_private_sigmas()

        new_noises = private_clips / private_weights
        new_clips = private_weights * private_sigmas

        new_noise_schedule = self.noise_schedule.__class__.from_projection(
            self.noise_schedule, new_noises
        )
        new_clip_schedule = self.clip_schedule.__class__.from_projection(
            self.clip_schedule, new_clips
        )

        clip_schedule = self.__diff_clips_select(new_clip_schedule, self.clip_schedule)
        noise_schedule = self.__diff_clips_select(
            self.noise_schedule, new_noise_schedule
        )

        return self.__class__(
            noise_schedule=noise_schedule,
            clip_schedule=clip_schedule,
            privacy_params=self.privacy_params,
            diff_clips=~self.diff_clips,
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
