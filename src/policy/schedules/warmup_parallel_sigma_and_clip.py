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

    FISTA acceleration
    ------------------
    When ``use_fista=True``, the outer loop in ``main.py`` should call
    ``fista_extrapolate()`` once after the initial ``project()``, then at the
    end of each step:

        x_new    = schedule.project()
        schedule = schedule.fista_advance(x_new)
        schedule = schedule.fista_extrapolate()

    FISTA momentum is automatically reset to zero at the warmup-to-tail
    transition to avoid the large accumulated momentum coefficient destabilising
    the freshly-seeded tail schedule.
    """

    noise_warmup: ConstantSchedule
    clip_warmup: ConstantSchedule
    noise_tail: AbstractSchedule
    clip_tail: AbstractSchedule
    privacy_params: GDPPrivacyParameters
    step_count: Array
    warmup_steps: int
    use_fista: bool
    # FISTA state — never updated by the optax optimizer (see apply_updates).
    _x_curr_sigmas: Array  # projected x_k sigmas,     shape (T,)
    _x_curr_clips: Array  # projected x_k clips,      shape (T,)
    _x_prev_sigmas: Array  # projected x_{k-1} sigmas, shape (T,)
    _x_prev_clips: Array  # projected x_{k-1} clips,  shape (T,)
    _fista_t: Array  # FISTA t parameter, scalar float

    def __init__(
        self,
        noise_warmup: ConstantSchedule,
        clip_warmup: ConstantSchedule,
        noise_tail: AbstractSchedule,
        clip_tail: AbstractSchedule,
        privacy_params: GDPPrivacyParameters,
        step_count: int | Array = 0,
        warmup_steps: int = 30,
        use_fista: bool = False,
        _x_curr_sigmas: Array | None = None,
        _x_curr_clips: Array | None = None,
        _x_prev_sigmas: Array | None = None,
        _x_prev_clips: Array | None = None,
        _fista_t: Array | None = None,
    ):
        self.noise_warmup = noise_warmup
        self.clip_warmup = clip_warmup
        self.noise_tail = noise_tail
        self.clip_tail = clip_tail
        self.privacy_params = privacy_params
        self.step_count = jnp.asarray(step_count, dtype=jnp.int32)
        self.warmup_steps = warmup_steps
        self.use_fista = use_fista
        T = privacy_params.T
        placeholder = jnp.ones(T)
        self._x_curr_sigmas = placeholder if _x_curr_sigmas is None else _x_curr_sigmas
        self._x_curr_clips = placeholder if _x_curr_clips is None else _x_curr_clips
        self._x_prev_sigmas = placeholder if _x_prev_sigmas is None else _x_prev_sigmas
        self._x_prev_clips = placeholder if _x_prev_clips is None else _x_prev_clips
        self._fista_t = jnp.ones(()) if _fista_t is None else _fista_t

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
            use_fista=conf.use_fista,
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
        # Only apply optax updates to the learnable schedules; FISTA state is
        # managed explicitly by fista_advance/fista_extrapolate.
        new_noise_warmup = eqx.apply_updates(self.noise_warmup, updates.noise_warmup)
        new_clip_warmup = eqx.apply_updates(self.clip_warmup, updates.clip_warmup)
        new_noise_tail = eqx.apply_updates(self.noise_tail, updates.noise_tail)
        new_clip_tail = eqx.apply_updates(self.clip_tail, updates.clip_tail)
        return self.__class__(
            noise_warmup=new_noise_warmup,
            clip_warmup=new_clip_warmup,
            noise_tail=new_noise_tail,
            clip_tail=new_clip_tail,
            privacy_params=self.privacy_params,
            step_count=self.step_count,
            warmup_steps=self.warmup_steps,
            use_fista=self.use_fista,
            _x_curr_sigmas=self._x_curr_sigmas,
            _x_curr_clips=self._x_curr_clips,
            _x_prev_sigmas=self._x_prev_sigmas,
            _x_prev_clips=self._x_prev_clips,
            _fista_t=self._fista_t,
        )

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
            use_fista=self.use_fista,
            _x_curr_sigmas=self._x_curr_sigmas,
            _x_curr_clips=self._x_curr_clips,
            _x_prev_sigmas=self._x_prev_sigmas,
            _x_prev_clips=self._x_prev_clips,
            _fista_t=self._fista_t,
        )

    def fista_extrapolate(self) -> Self:
        """Return the FISTA lookahead y_k = x_k + mom * (x_k - x_{k-1}).

        On the first call (_fista_t == 1 => mom == 0) the placeholder _x_prev
        values are invalid; this is handled by falling back to the current
        schedule values for both curr and prev, giving y = x_k (no extrapolation)
        and correctly seeding _x_prev for subsequent steps.
        """
        t_next = (1 + jnp.sqrt(1 + 4 * self._fista_t**2)) / 2
        mom = (self._fista_t - 1) / t_next

        x_sig = self.get_private_sigmas()
        x_clip = self.get_private_clips()

        # When _fista_t == 1 (first call), treat x_prev == x_curr so mom * 0 == 0.
        x_prev_sig = jnp.where(self._fista_t == 1.0, x_sig, self._x_prev_sigmas)
        x_prev_clip = jnp.where(self._fista_t == 1.0, x_clip, self._x_prev_clips)

        y_sig = x_sig + mom * (x_sig - x_prev_sig)
        y_clip = x_clip + mom * (x_clip - x_prev_clip)

        y_noise_warmup = ConstantSchedule.from_projection(self.noise_warmup, y_sig)
        y_clip_warmup = ConstantSchedule.from_projection(self.clip_warmup, y_clip)
        y_noise_tail = self.noise_tail.__class__.from_projection(self.noise_tail, y_sig)
        y_clip_tail = self.clip_tail.__class__.from_projection(self.clip_tail, y_clip)

        return self.__class__(
            noise_warmup=y_noise_warmup,
            clip_warmup=y_clip_warmup,
            noise_tail=y_noise_tail,
            clip_tail=y_clip_tail,
            privacy_params=self.privacy_params,
            step_count=self.step_count,
            warmup_steps=self.warmup_steps,
            use_fista=self.use_fista,
            _x_curr_sigmas=x_sig,  # carry x_k so fista_advance can make it x_prev
            _x_curr_clips=x_clip,
            _x_prev_sigmas=x_prev_sig,
            _x_prev_clips=x_prev_clip,
            _fista_t=self._fista_t,  # t advances only in fista_advance
        )

    def fista_advance(self, x_new: Self) -> Self:
        """Advance FISTA state after projection: x_{k-1} <- x_k, x_k <- x_{k+1}, t <- t_{k+1}.

        Resets FISTA momentum to zero at the warmup-to-tail transition
        (x_new.step_count == warmup_steps) to prevent the large accumulated
        momentum coefficient from destabilising the freshly-seeded tail schedule.

        Must be called on the gradient-updated schedule (self), not on x_new,
        because self._x_curr_sigmas holds x_k (set by fista_extrapolate and
        preserved by apply_updates), which becomes the new x_prev.
        """
        t_next = (1 + jnp.sqrt(1 + 4 * self._fista_t**2)) / 2

        # Reset at the warmup-to-tail transition: zero momentum for the first tail step.
        at_transition = int(x_new.step_count) == x_new.warmup_steps
        fista_t_new = jnp.ones(()) if at_transition else t_next
        x_prev_sig = x_new.get_private_sigmas() if at_transition else self._x_curr_sigmas
        x_prev_clip = x_new.get_private_clips() if at_transition else self._x_curr_clips

        return self.__class__(
            noise_warmup=x_new.noise_warmup,
            clip_warmup=x_new.clip_warmup,
            noise_tail=x_new.noise_tail,
            clip_tail=x_new.clip_tail,
            privacy_params=x_new.privacy_params,
            step_count=x_new.step_count,
            warmup_steps=x_new.warmup_steps,
            use_fista=self.use_fista,
            _x_curr_sigmas=x_new.get_private_sigmas(),
            _x_curr_clips=x_new.get_private_clips(),
            _x_prev_sigmas=x_prev_sig,
            _x_prev_clips=x_prev_clip,
            _fista_t=fista_t_new,
        )

    def _get_log_arrays(self) -> dict[str, Array]:
        return {
            "sigmas": self.get_private_sigmas(),
            "clips": self.get_private_clips(),
            "mus": self.get_private_weights(),
        }
