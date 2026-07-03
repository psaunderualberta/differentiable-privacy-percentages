from typing import Self

import equinox as eqx
from jaxtyping import Array

from policy.base_schedules.abstract import AbstractSchedule
from policy.base_schedules.factory import base_schedule_factory
from policy.schedules._registry import register
from policy.schedules.abstract import AbstractNoiseAndClipSchedule
from policy.schedules.config import DecoupledSigmaAndClipScheduleConfig
from privacy.gdp_privacy import GDPPrivacyParameters
from privacy.rdp_privacy import RDPPrivacyParameters


@register(DecoupledSigmaAndClipScheduleConfig)
class DecoupledSigmaAndClipSchedule(AbstractNoiseAndClipSchedule):
    """Noise std σ_noise = C · σ_mult, with C fully decoupled from the privacy budget.

    The noise-side base schedule's ``get_valid_schedule()`` returns σ_mult;
    ``get_private_weights()`` returns the accountant's noise weight w = 1/σ_mult
    (the privacy-constraint variable) and ``get_private_noise_scales()`` returns
    C · σ_mult at the dp.py boundary. Under RDP accounting (ADR-0007/0008) the
    per-step cost depends on w only, so ``project()`` is the equality **scaling
    retraction** onto the budget manifold ρ_total(α*; w) = c(α*) — it scales the
    σ_mult side by a common factor and leaves C untouched.
    """

    noise_schedule: AbstractSchedule
    clip_schedule: AbstractSchedule
    privacy_params: RDPPrivacyParameters

    def __init__(
        self,
        noise_schedule: AbstractSchedule,
        clip_schedule: AbstractSchedule,
        privacy_params: RDPPrivacyParameters,
    ):
        self.noise_schedule = noise_schedule
        self.clip_schedule = clip_schedule
        self.privacy_params = privacy_params

    @classmethod
    def from_config(
        cls,
        conf: DecoupledSigmaAndClipScheduleConfig,
        privacy_params: GDPPrivacyParameters,
    ) -> "DecoupledSigmaAndClipSchedule":
        # The decoupled (DP-PSAC) schedule accounts under RDP (ADR-0007/0008);
        # build RDP params from the (ε, δ, p, T) carried by the shared GDP object.
        T = privacy_params.T
        rdp_params = RDPPrivacyParameters(
            privacy_params.eps,
            privacy_params.delta,
            privacy_params.p,
            T,
        )
        noise_schedule = base_schedule_factory(conf.noise, T)
        clip_schedule = base_schedule_factory(conf.clip, T)
        return cls(noise_schedule, clip_schedule, rdp_params)

    def get_private_noise_scales(self) -> Array:
        return self.get_private_clips() * self._get_private_sigmas()

    def get_private_clips(self) -> Array:
        return self.clip_schedule.get_valid_schedule().squeeze()

    def _get_private_sigmas(self):
        return self.noise_schedule.get_valid_schedule().squeeze()

    def get_private_weights(self) -> Array:
        return 1 / self._get_private_sigmas()

    def apply_updates(self, updates) -> Self:
        return eqx.apply_updates(self, updates)

    def constraint_value(self) -> Array:
        """Signed budget residual ``g = ρ_total(α*; w) − c(α*)`` at the binding order.

        Zero on the budget manifold; differentiable w.r.t. the noise control
        points (the manifold normal is its gradient). The Riemannian tangent
        projection differentiates *this*, never the retraction.
        """
        return self.privacy_params.constraint(self.get_private_weights())

    def refresh_alpha_star(self) -> Self:
        """Re-select the binding order α* for the current schedule (non-JIT).

        Called between outer steps on the (on-budget) schedule so the following
        tangent projection uses a single fixed order (ADR-0007). ``select_optimal_alpha``
        returns a Python int, so this must run outside JIT.
        """
        new_params = self.privacy_params.with_alpha_star(self.get_private_weights())
        return self.__class__(
            noise_schedule=self.noise_schedule,
            clip_schedule=self.clip_schedule,
            privacy_params=new_params,
        )

    @eqx.filter_jit
    def project(self) -> Self:
        """RDP equality scaling retraction onto the budget manifold.

        Scales the σ_mult side by a common factor until the schedule realises the
        target ε at δ (exact in the BSpline family); C is left untouched.
        """
        scaled = self.privacy_params.project_scale(self._get_private_sigmas())
        new_noise_schedule = self.noise_schedule.__class__.from_projection(
            self.noise_schedule,
            scaled,
        )
        return self.__class__(
            noise_schedule=new_noise_schedule,
            clip_schedule=self.clip_schedule,
            privacy_params=self.privacy_params,
        )

    def _get_log_arrays(self) -> dict[str, Array]:
        return {
            "sigmas": self.get_private_noise_scales(),
            "clips": self.get_private_clips(),
            "mus": self.get_private_weights(),
            "multipliers": self._get_private_sigmas(),
        }
