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
from policy.schedules.config import WarmupAlternatingSigmaAndClipScheduleConfig
from privacy.gdp_privacy import GDPPrivacyParameters


@register(WarmupAlternatingSigmaAndClipScheduleConfig)
class WarmupAlternatingSigmaAndClipSchedule(AbstractNoiseAndClipSchedule):
    noise_warmup: ConstantSchedule
    clip_warmup: ConstantSchedule
    noise_tail: AbstractSchedule
    clip_tail: AbstractSchedule
    privacy_params: GDPPrivacyParameters
    diff_clips: Array
    step_count: Array
    warmup_steps: int

    def __init__(
        self,
        noise_warmup: ConstantSchedule,
        clip_warmup: ConstantSchedule,
        noise_tail: AbstractSchedule,
        clip_tail: AbstractSchedule,
        privacy_params: GDPPrivacyParameters,
        diff_clips: bool | Array = False,
        step_count: int | Array = 0,
        warmup_steps: int = 30,
    ):
        """Initialise the schedule with constant warmup schedules and learnable tail schedules.

        Args:
            noise_warmup: Constant σ schedule used during warmup.
            clip_warmup: Constant clip schedule used during warmup.
            noise_tail: Learnable σ schedule activated after warmup.
            clip_tail: Learnable clip schedule activated after warmup.
            privacy_params: GDP privacy budget and subsampling parameters.
            diff_clips: If True, differentiate through clips and stop-gradient noise; alternated each step.
            step_count: Current outer-loop step counter (used to track warmup/tail phase).
            warmup_steps: Number of outer-loop steps to spend in the warmup phase.
        """
        self.noise_warmup = noise_warmup
        self.clip_warmup = clip_warmup
        self.noise_tail = noise_tail
        self.clip_tail = clip_tail
        self.privacy_params = privacy_params
        self.diff_clips = jnp.asarray(diff_clips)
        self.step_count = jnp.asarray(step_count, dtype=jnp.int32)
        self.warmup_steps = warmup_steps

    @classmethod
    def from_config(
        cls,
        conf: WarmupAlternatingSigmaAndClipScheduleConfig,
        privacy_params: GDPPrivacyParameters,
    ) -> "WarmupAlternatingSigmaAndClipSchedule":
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
            diff_clips=conf.diff_clips_first,
            step_count=0,
            warmup_steps=warmup_steps,
        )

    def _is_warmup(self) -> Array:
        """Return True while the outer-loop step count is within the warmup phase."""
        return jnp.all(self.step_count < self.warmup_steps)

    def __select(self, condition: Array, a, b):
        """Select elementwise between pytrees `a` and `b` based on a boolean JAX array condition."""
        def tree_select(x, y):
            if x is None:
                return x
            return jlax.select(condition, x, y)

        return jtree.map(tree_select, a, b)

    def get_private_sigmas(self) -> Array:
        is_warmup = self._is_warmup()
        sigmas = jlax.cond(
            is_warmup,
            lambda: self.noise_warmup.get_valid_schedule().squeeze(),
            lambda: self.noise_tail.get_valid_schedule().squeeze(),
        )
        return jlax.select(self.diff_clips, jlax.stop_gradient(sigmas), sigmas)

    def get_private_clips(self) -> Array:
        is_warmup = self._is_warmup()
        clips = jlax.cond(
            is_warmup,
            lambda: self.clip_warmup.get_valid_schedule().squeeze(),
            lambda: self.clip_tail.get_valid_schedule().squeeze(),
        )
        return jlax.select(self.diff_clips, clips, jlax.stop_gradient(clips))

    def get_private_weights(self) -> Array:
        private_sigmas = self.get_private_sigmas()
        clips = self.get_private_clips()
        return self.privacy_params.project_weights(clips / private_sigmas).squeeze()

    def apply_updates(self, updates) -> Self:
        updated_noise_warmup = eqx.apply_updates(
            self.noise_warmup, updates.noise_warmup
        )
        updated_clip_warmup = eqx.apply_updates(self.clip_warmup, updates.clip_warmup)
        updated_noise_tail = eqx.apply_updates(self.noise_tail, updates.noise_tail)
        updated_clip_tail = eqx.apply_updates(self.clip_tail, updates.clip_tail)

        # diff_clips=True: update clip, freeze noise; diff_clips=False: update noise, freeze clip
        new_noise_warmup = self.__select(
            self.diff_clips, self.noise_warmup, updated_noise_warmup
        )
        new_clip_warmup = self.__select(
            self.diff_clips, updated_clip_warmup, self.clip_warmup
        )
        new_noise_tail = self.__select(
            self.diff_clips, self.noise_tail, updated_noise_tail
        )
        new_clip_tail = self.__select(
            self.diff_clips, updated_clip_tail, self.clip_tail
        )

        return self.__class__(
            noise_warmup=new_noise_warmup,
            clip_warmup=new_clip_warmup,
            noise_tail=new_noise_tail,
            clip_tail=new_clip_tail,
            privacy_params=self.privacy_params,
            diff_clips=self.diff_clips,
            step_count=self.step_count,
            warmup_steps=self.warmup_steps,
        )

    @eqx.filter_jit
    def project(self) -> Self:
        is_warmup = self._is_warmup()
        is_last_warmup = self.step_count == self.warmup_steps - 1

        private_sigmas = self.get_private_sigmas()
        private_clips = self.get_private_clips()
        private_weights = self.privacy_params.project_weights(
            private_clips / private_sigmas
        ).squeeze()

        new_noises = private_clips / private_weights
        new_clips = private_weights * private_sigmas

        # Compute projected representations for all four schedules.
        # During warmup, new_noises/new_clips come from the constant schedules,
        # so proj_noise_tail initialises the tail from the learned constant values.
        # Post-warmup, new_noises/new_clips come from the tail schedules.
        proj_noise_warmup = ConstantSchedule.from_projection(
            self.noise_warmup, new_noises
        )
        proj_clip_warmup = ConstantSchedule.from_projection(self.clip_warmup, new_clips)

        proj_noise_tail = self.noise_tail.__class__.from_projection(
            self.noise_tail, new_noises
        )
        proj_clip_tail = self.clip_tail.__class__.from_projection(
            self.clip_tail, new_clips
        )

        # Apply diff_clips alternation: diff_clips=True → update clip, freeze noise.
        new_noise_warmup = self.__select(
            self.diff_clips, proj_noise_warmup, self.noise_warmup
        )
        new_clip_warmup = self.__select(
            self.diff_clips, self.clip_warmup, proj_clip_warmup
        )
        new_noise_tail_alt = self.__select(
            self.diff_clips, proj_noise_tail, self.noise_tail
        )
        new_clip_tail_alt = self.__select(
            self.diff_clips, self.clip_tail, proj_clip_tail
        )

        # Phase selection for warmup schedules: update during warmup, freeze post-warmup.
        final_noise_warmup = jtree.map(
            lambda new, old: jlax.select(is_warmup, new, old),
            new_noise_warmup,
            self.noise_warmup,
        )
        final_clip_warmup = jtree.map(
            lambda new, old: jlax.select(is_warmup, new, old),
            new_clip_warmup,
            self.clip_warmup,
        )

        # Phase selection for tail schedules:
        #   warmup (not last): keep current tail unchanged
        #   last warmup step:  initialise tail from projected constant values
        #   post-warmup:       apply alternating projected tail
        final_noise_tail = jtree.map(
            lambda proj, alt, current: jlax.select(
                is_warmup,
                jlax.select(is_last_warmup, proj, current),
                alt,
            ),
            proj_noise_tail,
            new_noise_tail_alt,
            self.noise_tail,
        )
        final_clip_tail = jtree.map(
            lambda proj, alt, current: jlax.select(
                is_warmup,
                jlax.select(is_last_warmup, proj, current),
                alt,
            ),
            proj_clip_tail,
            new_clip_tail_alt,
            self.clip_tail,
        )

        return self.__class__(
            noise_warmup=final_noise_warmup,
            clip_warmup=final_clip_warmup,
            noise_tail=final_noise_tail,
            clip_tail=final_clip_tail,
            privacy_params=self.privacy_params,
            diff_clips=~self.diff_clips,
            step_count=self.step_count + 1,
            warmup_steps=self.warmup_steps,
        )

    def _get_log_arrays(self) -> dict[str, Array]:
        return {
            "sigmas": self.get_private_sigmas(),
            "clips": self.get_private_clips(),
            "mus": self.get_private_weights(),
        }

