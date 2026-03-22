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
from policy.schedules.config import WarmupParallelSigmaAndClipScheduleConfig
from privacy.gdp_privacy import GDPPrivacyParameters


@register(WarmupParallelSigmaAndClipScheduleConfig)
class WarmupParallelSigmaAndClipSchedule(AbstractNoiseAndClipSchedule):
    """Warmup variant of ParallelSigmaAndClipSchedule.

    During warmup, constant σ and clip schedules are tuned jointly via
    ``project_sigma_and_clip`` (Euclidean projection onto the GDP constraint).
    After warmup, learnable tail schedules take over, still projected with
    ``project_sigma_and_clip`` so both σ and clip remain co-optimised.

    On the final warmup step the tail schedules are initialised from the
    projected constant values, giving a warm start for the tail phase.
    """

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
        conf: WarmupParallelSigmaAndClipScheduleConfig,
        privacy_params: GDPPrivacyParameters,
    ) -> "WarmupParallelSigmaAndClipSchedule":
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

    @eqx.filter_jit
    def project(self) -> Self:
        is_warmup = self._is_warmup()
        is_last_warmup = self.step_count == self.warmup_steps - 1

        proj_sigmas, proj_clips = self.privacy_params.project_sigma_and_clip(
            self.get_private_sigmas(),
            self.get_private_clips(),
        )

        # --- warmup schedules ---
        proj_noise_warmup = ConstantSchedule.from_projection(self.noise_warmup, proj_sigmas)
        proj_clip_warmup = ConstantSchedule.from_projection(self.clip_warmup, proj_clips)

        final_noise_warmup = jtree.map(
            lambda new, old: jlax.select(is_warmup, new, old),
            proj_noise_warmup,
            self.noise_warmup,
        )
        final_clip_warmup = jtree.map(
            lambda new, old: jlax.select(is_warmup, new, old),
            proj_clip_warmup,
            self.clip_warmup,
        )

        # --- tail schedules ---
        # During warmup (not last step): keep current tail unchanged.
        # On last warmup step: seed tail from the projected constant values.
        # Post-warmup: project the tail directly.
        proj_noise_tail = self.noise_tail.__class__.from_projection(self.noise_tail, proj_sigmas)
        proj_clip_tail = self.clip_tail.__class__.from_projection(self.clip_tail, proj_clips)

        final_noise_tail = jtree.map(
            lambda proj, current: jlax.select(
                is_warmup,
                jlax.select(is_last_warmup, proj, current),
                proj,
            ),
            proj_noise_tail,
            self.noise_tail,
        )
        final_clip_tail = jtree.map(
            lambda proj, current: jlax.select(
                is_warmup,
                jlax.select(is_last_warmup, proj, current),
                proj,
            ),
            proj_clip_tail,
            self.clip_tail,
        )

        return self.__class__(
            noise_warmup=final_noise_warmup,
            clip_warmup=final_clip_warmup,
            noise_tail=final_noise_tail,
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
