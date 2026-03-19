from typing import Self

import equinox as eqx
import jax.lax as jlax
import jax.numpy as jnp
import jax.tree as jtree
from jaxtyping import Array

from conf.singleton_conf import SingletonConfig
from policy.base_schedules.abstract import AbstractSchedule
from policy.base_schedules.constant import ConstantSchedule
from policy.base_schedules.factory import base_schedule_factory
from policy.schedules._registry import register
from policy.schedules.abstract import AbstractNoiseAndClipSchedule
from policy.schedules.config import WarmupSigmaAndClipScheduleConfig
from privacy.gdp_privacy import GDPPrivacyParameters


@register(WarmupSigmaAndClipScheduleConfig)
class WarmupSigmaAndClipSchedule(AbstractNoiseAndClipSchedule):
    noise_warmup: ConstantSchedule
    clip_warmup: ConstantSchedule
    noise_tail: AbstractSchedule
    clip_tail: AbstractSchedule
    privacy_params: GDPPrivacyParameters
    step_count: Array
    warmup_steps: int

    def __init__(
        self,
        noise_warmup: ConstantSchedule,
        clip_warmup: ConstantSchedule,
        noise_tail: AbstractSchedule,
        clip_tail: AbstractSchedule,
        privacy_params: GDPPrivacyParameters,
        step_count: int | Array = 0,
        warmup_steps: int = 30,
    ):
        """Initialise the schedule with constant warmup schedules and learnable tail schedules.

        During warmup, constant σ and clip schedules are used. After warmup, the tail schedules
        take over. Projection only updates clip (leaving σ unchanged), the inverse of SigmaAndClipSchedule.

        Args:
            noise_warmup: Constant σ schedule used during warmup.
            clip_warmup: Constant clip schedule used during warmup.
            noise_tail: Learnable σ schedule activated after warmup.
            clip_tail: Learnable clip schedule activated after warmup.
            privacy_params: GDP privacy budget and subsampling parameters.
            step_count: Current outer-loop step counter (used to track warmup/tail phase).
            warmup_steps: Number of outer-loop steps to spend in the warmup phase.
        """
        self.noise_warmup = noise_warmup
        self.clip_warmup = clip_warmup
        self.noise_tail = noise_tail
        self.clip_tail = clip_tail
        self.privacy_params = privacy_params
        self.step_count = jnp.asarray(step_count, dtype=jnp.int32)
        self.warmup_steps = warmup_steps

    @classmethod
    def from_config(
        cls,
        conf: WarmupSigmaAndClipScheduleConfig,
        privacy_params: GDPPrivacyParameters,
    ) -> "WarmupSigmaAndClipSchedule":
        T = privacy_params.T
        total_timesteps = SingletonConfig.get_sweep_config_instance().total_timesteps
        warmup_steps = max(1, int(conf.warmup_pct * total_timesteps))
        noise_warmup = ConstantSchedule(conf.warmup_noise_init, T)
        clip_warmup = ConstantSchedule(conf.warmup_clip_init, T)
        noise_tail = base_schedule_factory(conf.noise_tail, T)
        clip_tail = base_schedule_factory(conf.clip_tail, T)
        return cls(
            noise_warmup=noise_warmup,
            clip_warmup=clip_warmup,
            noise_tail=noise_tail,
            clip_tail=clip_tail,
            privacy_params=privacy_params,
            step_count=0,
            warmup_steps=warmup_steps,
        )

    def _is_warmup(self) -> Array:
        """Return True while the outer-loop step count is within the warmup phase."""
        return jnp.all(self.step_count < self.warmup_steps)

    def get_private_sigmas(self) -> Array:
        is_warmup = self._is_warmup()
        return jlax.cond(
            is_warmup,
            lambda: self.noise_warmup.get_valid_schedule().squeeze(),
            lambda: self.noise_tail.get_valid_schedule().squeeze(),
        )

    def get_private_clips(self) -> Array:
        is_warmup = self._is_warmup()
        return jlax.cond(
            is_warmup,
            lambda: self.clip_warmup.get_valid_schedule().squeeze(),
            lambda: self.clip_tail.get_valid_schedule().squeeze(),
        )

    def get_private_weights(self) -> Array:
        private_sigmas = self.get_private_sigmas()
        clips = self.get_private_clips()
        return self.privacy_params.project_weights(clips / private_sigmas).squeeze()

    def apply_updates(self, updates) -> Self:
        return eqx.apply_updates(self, updates)

    def project(self) -> Self:
        is_warmup = self._is_warmup()
        is_last_warmup = self.step_count == self.warmup_steps - 1

        private_sigmas = self.get_private_sigmas()
        private_weights = self.privacy_params.project_weights(
            self.get_private_clips() / private_sigmas,
        ).squeeze()

        new_clips = private_weights * private_sigmas

        # Project clip schedules; σ schedules are left unchanged (inverse of SigmaAndClipSchedule).
        proj_clip_warmup = ConstantSchedule.from_projection(
            self.clip_warmup,
            new_clips,
        )
        proj_clip_tail = self.clip_tail.__class__.from_projection(
            self.clip_tail,
            new_clips,
        )

        # Phase selection for warmup clip: update during warmup, freeze post-warmup.
        final_clip_warmup = jtree.map(
            lambda new, old: jlax.select(is_warmup, new, old),
            proj_clip_warmup,
            self.clip_warmup,
        )

        # Phase selection for tail clip:
        #   warmup (not last): keep current tail unchanged
        #   last warmup step:  initialise tail from projected constant values
        #   post-warmup:       apply projected tail
        final_clip_tail = jtree.map(
            lambda proj, current: jlax.select(
                is_warmup,
                jlax.select(is_last_warmup, proj, current),
                proj,
            ),
            proj_clip_tail,
            self.clip_tail,
        )

        # Sigma warmup: project during warmup (for consistency), freeze post-warmup.
        proj_noise_warmup = ConstantSchedule.from_projection(
            self.noise_warmup,
            private_sigmas,
        )
        final_noise_warmup = jtree.map(
            lambda new, old: jlax.select(is_warmup, new, old),
            proj_noise_warmup,
            self.noise_warmup,
        )

        return self.__class__(
            noise_warmup=final_noise_warmup,
            clip_warmup=final_clip_warmup,
            noise_tail=self.noise_tail,
            clip_tail=final_clip_tail,
            privacy_params=self.privacy_params,
            step_count=self.step_count + 1,
            warmup_steps=self.warmup_steps,
        )

    def _get_log_arrays(self) -> dict[str, Array]:
        return {
            "sigmas": self.get_private_sigmas(),
            "clips": self.get_private_clips(),
            "mus": self.get_private_weights(),
        }
