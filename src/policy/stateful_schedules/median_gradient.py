import math

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import vmap
from jaxtyping import Array, ArrayLike, PRNGKeyArray, PyTree

from conf.singleton_conf import SingletonConfig
from policy.stateful_schedules._registry import register
from policy.stateful_schedules.abstract import (
    AbstractScheduleState,
    AbstractStatefulNoiseAndClipSchedule,
)
from policy.stateful_schedules.config import StatefulMedianGradientNoiseAndClipConfig
from privacy.gdp_privacy import GDPPrivacyParameters
from util.logger import Loggable, LoggableArray, LoggingSchema


@register(StatefulMedianGradientNoiseAndClipConfig)
class StatefulMedianGradientNoiseAndClipSchedule(AbstractStatefulNoiseAndClipSchedule):
    """Adaptive clipping (Andrew et al., NeurIPS 2021).

    Steers the *within-clip fraction* toward the quantile target γ; the median gradient
    norm is the fixed point of that process, never a quantity estimated directly. The
    fraction is released privately as a second vector group of the same per-step
    Gaussian mechanism as the gradient, so μ_count² + μ_grad² = μ₀² and the total ε is
    unchanged (ADR 0013).

    https://proceedings.neurips.cc/paper_files/paper/2021/file/91cff01af640a24e7f9f7a5ab407889f-Paper.pdf
    """

    MAX_RHO = 0.25
    """Largest share of the per-step budget the count release may consume."""

    class MedianGradientScheduleState(AbstractScheduleState):
        C: Array
        sigma: Array

        def __init__(self, C: Array, sigma: Array):
            """Store the current clip threshold and noise scale.

            Args:
                C: Current gradient clipping threshold.
                sigma: Current Gaussian noise standard deviation.
            """
            self.C = C
            self.sigma = sigma

        def get_clip(self) -> Array:
            return self.C

        def get_noise(self) -> Array:
            return self.sigma

    c_0: Array
    eta_c: Array
    r: float
    iteration_array: Array
    privacy_params: GDPPrivacyParameters
    gamma: float = 0.5
    c_min: float = 1e-12

    def __init__(
        self,
        c_0: ArrayLike,
        eta_c: ArrayLike,
        privacy_params: GDPPrivacyParameters,
        r: float = 20.0,
    ):
        """Initialise the adaptive clipping schedule.

        Args:
            c_0: Initial gradient clipping threshold.
            eta_c: Learning rate for the exponential clip update rule.
            privacy_params: GDP privacy parameters supplying the per-step μ₀.
            r: Count noise ratio — the count release's noise is ``L / r`` for the
                expected batch size ``L``, so the within-clip fraction has standard
                error ``1 / r`` in every privacy regime (Andrew et al.'s default is 20).

        The noised update is normalised by ``C_t`` before the optimiser step (see
        :meth:`postprocess_update`), decoupling the effective step size from the
        absolute clip magnitude so the model no longer freezes as ``C_t`` tracks the
        shrinking median gradient norm toward zero — the Andrew-et-al.-faithful fix
        (their server learning rate plays the same decoupling role). This is
        privacy-neutral: it post-processes the already-privatised gradient.
        """
        self.c_0 = jnp.asarray(c_0)
        self.eta_c = jnp.asarray(eta_c)
        self.r = float(r)
        self.privacy_params = privacy_params
        self.iteration_array = jnp.arange(self.privacy_params.T)
        self._check_budget_absorbs_count()

    @classmethod
    def from_config(
        cls,
        conf: StatefulMedianGradientNoiseAndClipConfig,
        privacy_params: GDPPrivacyParameters,
    ) -> "StatefulMedianGradientNoiseAndClipSchedule":
        return cls(conf.c_0, conf.eta_c, privacy_params, conf.r)

    @property
    def expected_batch_size(self) -> int:
        """The public expected batch size ``L`` the count release is divided by.

        Never the truncated-Poisson buffer size nor the realised occupancy (ADR 0009).
        """
        return SingletonConfig.get_environment_config_instance().batch_size

    @property
    def sigma_count(self) -> float:
        """Gaussian noise std added to the ±½-encoded within-clip count."""
        return self.expected_batch_size / self.r

    @property
    def mu_count(self) -> float:
        """Per-step GDP μ spent on the count release (±½ encoding ⇒ sensitivity ½)."""
        return 0.5 / self.sigma_count

    @property
    def mu_grad(self) -> Array:
        """Per-step GDP μ left for the gradient release after the count is paid for.

        The split happens at the *un-amplified* μ₀ because both groups are released
        under the same Poisson draw, so amplification applies once to the joint
        release (ADR 0013). The accountant therefore needs no change.
        """
        return jnp.sqrt(self.privacy_params.mu_0**2 - self.mu_count**2)

    @property
    def rho(self) -> Array:
        """Median budget fraction — the share of μ₀² spent on the count release."""
        return self.mu_count**2 / self.privacy_params.mu_0**2

    def _check_budget_absorbs_count(self) -> None:
        """Reject regimes where the count release would eat the per-step budget.

        μ_count = r/(2L) grows as L shrinks, and once it reaches μ₀ the gradient noise
        is NaN — thousands of steps into training, far from the cause. Transfer targets
        choose their own batch size, so this is a live constraint, not a theoretical one.
        """
        L = self.expected_batch_size
        mu_0 = float(self.privacy_params.mu_0)
        rho = float(self.rho)
        print(
            f"Adaptive clip: batch_size L={L}, r={self.r:g}, mu_0={mu_0:.4g}, "
            f"rho={rho:.4g} of the per-step budget spent on the count release."
        )
        if rho > self.MAX_RHO:
            min_L = math.ceil(self.r / (math.sqrt(self.MAX_RHO) * 2 * mu_0))
            raise ValueError(
                f"Count release would consume rho={rho:.4g} of the per-step privacy "
                f"budget (limit {self.MAX_RHO}): batch_size={L}, r={self.r:g}, "
                f"mu_0={mu_0:.4g}. Raise batch_size to at least {min_L}, or lower r "
                f"(coarser within-clip fraction) to fit this regime."
            )

    def get_initial_state(self) -> MedianGradientScheduleState:
        sigma = self.c_0 / self.mu_grad
        return self.MedianGradientScheduleState(C=self.c_0, sigma=sigma)

    def update_state(
        self,
        state: AbstractScheduleState,
        grads: Array,
        iter: Array,
        batch_x: Array,
        batch_y: Array,
        valid: Array,
        key: PRNGKeyArray,
    ) -> MedianGradientScheduleState:
        """Release the within-clip fraction privately and step the clip threshold.

        The count is the second vector group of the same per-step Gaussian mechanism
        as the gradient release (ADR 0013): indicators are ±½-encoded so the
        replace-one sensitivity is ½, noised at ``L/r``, and divided by the *expected*
        batch size ``L`` — public, and never the realised buffer occupancy.
        """
        current_C = state.get_clip()
        norms = vmap(optax.tree.norm)(grads)
        # ±½ encoding (Andrew et al.): halves the sensitivity relative to {0,1} and
        # works against the public expected divisor L. Invalid buffer rows (ADR 0009)
        # contribute nothing.
        within = (norms <= current_C) & valid
        centred_count = jnp.sum(jnp.where(valid, jnp.where(within, 0.5, -0.5), 0.0))

        L = self.expected_batch_size
        noised_count = centred_count + self.sigma_count * jr.normal(key, ())
        # Clamping is post-processing of an already-private release, so it is free.
        b_bar = jnp.clip(noised_count / L + 0.5, 0.0, 1.0)

        # Numerical backstop only: `postprocess_update` divides by C_t. At r = 20 the
        # count noise is far too small for the clip random walk to approach this.
        new_C = jnp.maximum(
            current_C * jnp.exp(-self.eta_c * (b_bar - self.gamma)),
            self.c_min,
        )
        new_sigma = new_C / self.mu_grad

        return self.MedianGradientScheduleState(C=new_C, sigma=new_sigma)

    def postprocess_update(
        self,
        noised_grads: PyTree,
        state: AbstractScheduleState,
    ) -> PyTree:
        """Divide the noised clipped gradient by ``C_t``.

        The optimiser sees ``(ḡ_clip + noise) / C_t``.
        The signal ``ḡ_clip / C_t`` is an O(1) direction and the noise term becomes
        ``z / (μ₀·L)`` — constant in ``C_t`` — so the effective step no longer
        collapses as ``C_t`` decays toward zero. Pure post-processing of the already
        privatised gradient, so the privacy calibration is unchanged.
        """
        C = state.get_clip()
        return jax.tree.map(lambda g: g / C, noised_grads)

    def get_logging_schemas(self) -> list[LoggingSchema]:
        """Nothing to log: σ and C are chosen at runtime, not held as a schedule."""
        return []

    def get_loggables(self, force=False) -> list[Loggable | LoggableArray]:
        return []
