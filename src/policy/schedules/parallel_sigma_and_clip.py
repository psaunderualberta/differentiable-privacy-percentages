from typing import Self

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from policy.base_schedules.abstract import AbstractSchedule
from policy.base_schedules.factory import base_schedule_factory
from policy.schedules._registry import register
from policy.schedules.abstract import AbstractNoiseAndClipSchedule
from policy.schedules.config import ParallelSigmaAndClipScheduleConfig
from privacy.gdp_privacy import GDPPrivacyParameters


@register(ParallelSigmaAndClipScheduleConfig)
class ParallelSigmaAndClipSchedule(AbstractNoiseAndClipSchedule):
    """Jointly optimise sigma and clip with Euclidean projection onto the GDP constraint.

    Unlike ``SigmaAndClipSchedule``, which projects only the noise schedule while
    keeping clips fixed, this schedule projects both sigma and clip simultaneously
    via the nearest-point (L2) projection onto
        sum_i (exp((clip_i / sigma_i)^2) - 1) <= (mu/p)^2.

    FISTA acceleration
    ------------------
    Call ``fista_extrapolate()`` once right after the initial ``project()`` to seed
    the lookahead, then at the end of each outer-loop step:

        x_new  = schedule.project()
        schedule = schedule.fista_advance(x_new)   # x_{k-1} <- x_k, t advances
        schedule = schedule.fista_extrapolate()    # schedule now holds y_{k+1}

    ``_x_curr_sigmas/_clips`` carry the projected x_k through ``apply_updates``
    so that ``fista_advance`` can promote it to x_prev without needing to call
    ``get_private_sigmas`` on the already-gradient-updated schedule.
    """

    noise_schedule: AbstractSchedule
    clip_schedule: AbstractSchedule
    privacy_params: GDPPrivacyParameters
    use_fista: bool
    # FISTA state — never updated by the optax optimizer (see apply_updates).
    _x_curr_sigmas: Array  # projected x_k sigmas,   shape (T,)
    _x_curr_clips: Array  # projected x_k clips,    shape (T,)
    _x_prev_sigmas: Array  # projected x_{k-1} sigmas, shape (T,)
    _x_prev_clips: Array  # projected x_{k-1} clips,  shape (T,)
    _fista_t: Array  # FISTA t parameter, scalar float

    def __init__(
        self,
        noise_schedule: AbstractSchedule,
        clip_schedule: AbstractSchedule,
        privacy_params: GDPPrivacyParameters,
        use_fista: bool,
        _x_curr_sigmas: Array,
        _x_curr_clips: Array,
        _x_prev_sigmas: Array,
        _x_prev_clips: Array,
        _fista_t: Array,
    ):
        self.noise_schedule = noise_schedule
        self.clip_schedule = clip_schedule
        self.privacy_params = privacy_params
        self.use_fista = use_fista
        self._x_curr_sigmas = _x_curr_sigmas
        self._x_curr_clips = _x_curr_clips
        self._x_prev_sigmas = _x_prev_sigmas
        self._x_prev_clips = _x_prev_clips
        self._fista_t = _fista_t

    @classmethod
    def from_config(
        cls,
        conf: ParallelSigmaAndClipScheduleConfig,
        privacy_params: GDPPrivacyParameters,
    ) -> "ParallelSigmaAndClipSchedule":
        T = privacy_params.T
        noise_schedule = base_schedule_factory(conf.noise, T)
        clip_schedule = base_schedule_factory(conf.clip, T)
        placeholder = jnp.ones(T)
        return cls(
            noise_schedule=noise_schedule,
            clip_schedule=clip_schedule,
            privacy_params=privacy_params,
            use_fista=conf.use_fista,
            _x_curr_sigmas=placeholder,
            _x_curr_clips=placeholder,
            _x_prev_sigmas=placeholder,
            _x_prev_clips=placeholder,
            _fista_t=jnp.ones(()),  # t_1 = 1 => zero momentum on first extrapolation
        )

    def get_private_sigmas(self) -> Array:
        return self.noise_schedule.get_valid_schedule().squeeze()

    def get_private_clips(self) -> Array:
        return self.clip_schedule.get_valid_schedule().squeeze()

    def get_private_weights(self) -> Array:
        private_sigmas = self.get_private_sigmas()
        clips = self.get_private_clips()
        return self.privacy_params.project_weights(clips / private_sigmas).squeeze()

    def apply_updates(self, updates) -> Self:
        # Only apply optax updates to the learnable schedules; FISTA state is
        # managed explicitly by fista_advance/fista_extrapolate.
        new_noise = eqx.apply_updates(self.noise_schedule, updates.noise_schedule)
        new_clip = eqx.apply_updates(self.clip_schedule, updates.clip_schedule)
        return self.__class__(
            noise_schedule=new_noise,
            clip_schedule=new_clip,
            privacy_params=self.privacy_params,
            use_fista=self.use_fista,
            _x_curr_sigmas=self._x_curr_sigmas,
            _x_curr_clips=self._x_curr_clips,
            _x_prev_sigmas=self._x_prev_sigmas,
            _x_prev_clips=self._x_prev_clips,
            _fista_t=self._fista_t,
        )

    @eqx.filter_jit
    def project(self) -> Self:
        sigmas = self.get_private_sigmas()
        clips = self.get_private_clips()

        proj_sigmas, proj_clips = self.privacy_params.project_sigma_and_clip(sigmas, clips)

        new_noise_schedule = self.noise_schedule.__class__.from_projection(
            self.noise_schedule, proj_sigmas
        )
        new_clip_schedule = self.clip_schedule.__class__.from_projection(
            self.clip_schedule, proj_clips
        )

        return self.__class__(
            noise_schedule=new_noise_schedule,
            clip_schedule=new_clip_schedule,
            privacy_params=self.privacy_params,
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

        y_noise = self.noise_schedule.__class__.from_projection(self.noise_schedule, y_sig)
        y_clip_sched = self.clip_schedule.__class__.from_projection(self.clip_schedule, y_clip)

        return self.__class__(
            noise_schedule=y_noise,
            clip_schedule=y_clip_sched,
            privacy_params=self.privacy_params,
            use_fista=self.use_fista,
            _x_curr_sigmas=x_sig,  # carry x_k so fista_advance can make it x_prev
            _x_curr_clips=x_clip,
            _x_prev_sigmas=x_prev_sig,
            _x_prev_clips=x_prev_clip,
            _fista_t=self._fista_t,  # t advances only in fista_advance
        )

    def fista_advance(self, x_new: Self) -> Self:
        """Advance FISTA state after projection: x_{k-1} <- x_k, x_k <- x_{k+1}, t <- t_{k+1}.

        Must be called on the gradient-updated schedule (self), not on x_new,
        because self._x_curr_sigmas holds x_k (set by fista_extrapolate and
        preserved by apply_updates), which becomes the new x_prev.
        """
        t_next = (1 + jnp.sqrt(1 + 4 * self._fista_t**2)) / 2
        return self.__class__(
            noise_schedule=x_new.noise_schedule,
            clip_schedule=x_new.clip_schedule,
            privacy_params=x_new.privacy_params,
            use_fista=self.use_fista,
            _x_curr_sigmas=x_new.get_private_sigmas(),
            _x_curr_clips=x_new.get_private_clips(),
            _x_prev_sigmas=self._x_curr_sigmas,  # x_k carried from fista_extrapolate
            _x_prev_clips=self._x_curr_clips,
            _fista_t=t_next,
        )

    def _get_log_arrays(self) -> dict[str, Array]:
        return {
            "sigmas": self.get_private_sigmas(),
            "clips": self.get_private_clips(),
            "mus": self.get_private_weights(),
        }
