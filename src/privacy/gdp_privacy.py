import equinox as eqx
import jax
import jax.lax as jlax
import jax.numpy as jnp
import jax.scipy.stats as jstats
from jaxtyping import Array
from scipy import optimize

from conf.singleton_conf import SingletonConfig
from util.util import pytree_has_inf

# ---------------------------------------------------------------------------
# Helpers for project_sigma_and_clip — Euclidean projection onto
#   sum_i (exp(X_i^2 / Y_i^2) - 1) <= B   (X = clips, Y = sigmas)
# via outer bisection on the dual variable lambda and inner 2D Newton solve.
# ---------------------------------------------------------------------------


def _sc_residual(a, b, x, y, lam):
    mu2 = (a / b) ** 2
    e = jnp.exp(mu2)
    f0 = a - x + lam * (2.0 * a / b**2) * e
    f1 = b - y - lam * (2.0 * a**2 / b**3) * e
    return f0, f1


def _sc_jacobian(a, b, lam):
    mu2 = (a / b) ** 2
    e = jnp.exp(mu2)
    c = lam * e
    j00 = 1.0 + (2.0 * c / b**2) * (1.0 + 2.0 * mu2)
    j01 = -4.0 * c * a / b**3 * (1.0 + mu2)
    j10 = j01
    j11 = 1.0 + 2.0 * c * a**2 / b**4 * (3.0 + 2.0 * mu2)
    return j00, j01, j10, j11


def _sc_newton_step(a, b, x, y, lam):
    f0, f1 = _sc_residual(a, b, x, y, lam)
    j00, j01, j10, j11 = _sc_jacobian(a, b, lam)
    det = j00 * j11 - j01 * j10
    det = jnp.where(jnp.abs(det) < 1e-30, 1e-30, det)
    da = -(j11 * f0 - j01 * f1) / det
    db = -(-j10 * f0 + j00 * f1) / det
    return da, db


def _sc_inner_solve_scalar(a, b, x, y, lam):
    """Newton solve for a single (a, b) component. Vectorize with vmap."""

    def body(_, state):
        a, b = state
        da, db = _sc_newton_step(a, b, x, y, lam)
        max_step_a = jnp.where(da < 0, -0.9 * a / da, 1.0)
        max_step_b = jnp.where(db < 0, -0.9 * b / db, 1.0)
        alpha = jnp.minimum(1.0, jnp.minimum(max_step_a, max_step_b))
        return (
            jnp.maximum(a + alpha * da, 1e-12),
            jnp.maximum(b + alpha * db, 1e-12),
        )

    return jlax.fori_loop(0, 15, body, (a, b))


# Vectorized inner solve over all T components simultaneously.
_sc_inner_solve_vec = jax.vmap(_sc_inner_solve_scalar, in_axes=(0, 0, 0, 0, None))


def approx_to_gdp(eps, delta, tol=1e-12) -> float:
    """Convert (eps, delta)-DP to GDP.

    Args:
        eps: The epsilon parameter of (epsilon, delta)-DP.
        delta: The delta parameter of (epsilon, delta)-DP.

    Returns:
        The mu parameter of GDP.
    """

    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")
    if eps < 0:
        raise ValueError("epsilon must be non-negative")

    def f(current_mu):
        current_delta = jstats.norm.cdf(-eps / current_mu + current_mu / 2) - jnp.exp(
            eps,
        ) * jstats.norm.cdf(-eps / current_mu - current_mu / 2)
        return current_delta - delta

    return optimize.root_scalar(f, bracket=[tol, 100], method="brentq").root


class GDPPrivacyParameters(eqx.Module):
    eps: float
    delta: float
    __mu: float
    __p: float
    __T: int
    __mu_0: Array
    __w_min: Array
    __w_max: Array

    def __init__(self, eps: float, delta: float, p: float, T: int):
        """Compute GDP parameters from (ε, δ)-DP and subsampling settings.

        Args:
            eps: The ε parameter of (ε, δ)-DP.
            delta: The δ parameter of (ε, δ)-DP.
            p: Poisson subsampling probability (batch_size / dataset_size).
            T: Total number of DP-SGD steps.
        """
        self.eps = eps
        self.delta = delta
        self.__mu = approx_to_gdp(eps, delta)
        self.__p = p
        self.__T = T
        self.__mu_0 = self.compute_mu_0()
        self.__w_min = jnp.asarray(0.1)
        self.__w_max = jnp.asarray(10.0)

    @property
    def mu(self) -> float:
        return jlax.stop_gradient(self.__mu)

    @property
    def p(self) -> float:
        return jlax.stop_gradient(self.__p)

    @property
    def T(self) -> int:
        return jlax.stop_gradient(self.__T)

    @property
    def mu_0(self) -> Array:
        return jlax.stop_gradient(self.__mu_0)

    @property
    def w_min(self) -> Array:
        return jlax.stop_gradient(self.__w_min)

    @property
    def w_max(self) -> Array:
        return jlax.stop_gradient(self.__w_max)

    def compute_mu_0(self) -> Array:
        """Compute the per-step GDP μ₀ such that T uniform steps compose to the total μ."""
        return jnp.sqrt(jnp.log(self.mu**2 / (self.p**2 * self.T) + 1))

    def compute_eps(self, max_sigma: float | None = None) -> Array:
        """Compute the effective epsilon ratio between the maximum sigma and the per-step μ₀.

        Args:
            max_sigma: Upper bound on σ. Defaults to `max_sigma` from the policy config.

        Returns:
            The ratio (exp(1/max_sigma²) - 1) / (exp(μ₀²) - 1).
        """
        if max_sigma is None:
            max_sigma = SingletonConfig.get_policy_config_instance().max_sigma

        return (jnp.exp(1 / max_sigma**2) - 1) / (jnp.exp(self.mu_0**2) - 1)

    def compute_expenditure(self, sigmas: Array, clips: Array) -> Array:
        """Compute the total GDP μ expenditure for a given σ/clip schedule."""
        return jnp.sum(self.p * jnp.sqrt(jnp.exp((clips / sigmas) ** 2) - 1))

    def weights_to_mu_schedule(self, schedule: Array) -> Array:
        """Convert a GDP mu parameter to a Poisson subsampling schedule.

        Args:
            schedule: A 1D array representing the initial schedule (e.g., learning rates). Assumed to be non-negative and sum to T.

        Returns:
            A 1D array representing the adjusted schedule.
        """

        schedule = schedule**2
        return jnp.sqrt(jnp.log(schedule * (jnp.exp(self.mu_0**2) - 1) + 1))

    def mu_schedule_to_weights(self, schedule: Array) -> Array:
        """Convert to a Poisson subsampling schedule to vector of non-negative weights summing to T.
        Inverse of `mu_to_poisson_subsampling_shedule`

        Args:
            mu: The mu parameter of GDP. Assumed to be a non-negative scalar.
            schedule: A 1D array representing the initial schedule (e.g., learning rates). Assumed to be non-negative and sum to T.
            p: The subsampling probability. Assumed to be in (0, 1).
            T: The total number of steps. Assumed to be a positive integer.

        Returns:
            A 1D array representing the adjusted schedule.
        """
        sqrd_weights = (jnp.exp(schedule**2) - 1) / (jnp.exp(self.mu_0**2) - 1)
        return jnp.sqrt(sqrd_weights)

    def gdp_to_sigma(self, C: Array, mu: Array) -> Array:
        """Convert GDP mu parameter to Gaussian noise scale sigma.

        Args:
            mu: The mu parameter of GDP. Assumed to be a non-negative array.

        Returns:
            The Gaussian noise scale sigma.
        """

        return C / mu

    def _validated_mu_schedule(self, weights: Array) -> Array:
        """Convert weights to a μ schedule with runtime Inf/zero guards."""
        weights = eqx.error_if(weights, pytree_has_inf(weights), "weights have Inf!")
        weights = eqx.error_if(weights, (weights == 0).any(), "weights have 0!")
        mu_schedule = self.weights_to_mu_schedule(weights)
        mu_schedule = eqx.error_if(
            mu_schedule,
            pytree_has_inf(mu_schedule),
            "mu schedule has Inf!",
        )
        return eqx.error_if(
            mu_schedule,
            (mu_schedule == 0).any(),
            "Some mus are 0!",
        )

    def weights_to_sigma_schedule(self, C: Array, weights: Array) -> Array:
        """Convert weights to a per-step sigma schedule for G-DP."""
        return self.gdp_to_sigma(C, self._validated_mu_schedule(weights))

    def weights_to_clip_schedule(self, sigmas: Array, weights: Array) -> Array:
        """Convert weights to a per-step clip schedule for G-DP."""
        return sigmas * self._validated_mu_schedule(weights)

    def sigma_schedule_to_weights(self, C: Array, schedule: Array):
        """Convert a sigma schedule vector to a vector of non-negative weights summing to T for G-DP.
        Inverse of `weights_to_sigma_schedule`

        Args:
            schedule: sigma schedule vector. Assumed to be non-negative.
            mu: The mu parameter of GDP. Assumed to be a non-negative float or array.
            p: The sampling probability for poisson sampling
            T: The number of training iterations

        Returns:
            The Gaussian noise scale sigma.
        """
        max_sigma = SingletonConfig.get_policy_config_instance().max_sigma
        schedule = jnp.where(schedule > max_sigma, max_sigma, schedule)
        schedule = eqx.error_if(schedule, (schedule == 0).any(), "schedule has 0!")
        schedule = eqx.error_if(
            schedule,
            jnp.isinf(C / schedule).any(),
            "schedule has Inf!",
        )
        return self.mu_schedule_to_weights(C / schedule)

    # TODO: Move projection into the gradient computation
    @eqx.filter_jit
    def project_weights(self, weights: Array, tol: float | Array = 1e-6) -> Array:
        """
        post-GD weights (i.e. after updating sigma, clip, policy, or any three)
        returns: projected weights W s.t. 1^T @ W = self.T && W in [self.w_min, self.w_max]
        """

        # Ensure jnp array
        tol = jnp.asarray(tol)

        bound = (self.mu / self.p) ** 2 + self.T  # == sum_{i=1}^{T} e^(w_i^2)

        def safe_avg(lo_hi: tuple[Array, Array]) -> Array:
            lo, hi = lo_hi
            return lo + (hi - lo) / 2

        def c_i_tildes_cond(lo_hi: tuple[Array, Array]) -> Array:
            c_i_tildes, mu = lo_hi
            f_ = c_i_tildes * (1 + 2 * mu * jnp.exp(c_i_tildes**2)) - weights
            return jnp.any(f_ > tol)

        def c_i_tildes_body(
            lo_hi: tuple[Array, Array],
        ) -> tuple[Array, Array]:
            """
            The goal is to find c_i_tilde s.t.
            """
            mu = lo_hi[1]
            c_i_tildes = lo_hi[0]

            f_ = c_i_tildes * (1 + 2 * mu * jnp.exp(c_i_tildes**2)) - weights
            f_prime = 1 + 2 * mu * jnp.exp(c_i_tildes**2) * (1 + 2 * c_i_tildes**2)

            return (c_i_tildes - f_ / f_prime, mu)

        def get_c_i_tildes(mu: Array) -> Array:
            c_i_tildes, _ = jlax.while_loop(
                c_i_tildes_cond,
                c_i_tildes_body,
                (weights, mu),
            )

            return c_i_tildes

        def h(mu: Array) -> Array:
            c_i_tildes = get_c_i_tildes(mu)

            return jnp.sum(jnp.exp(c_i_tildes**2)) - bound

        def cond(lo_hi: tuple[Array, Array]) -> Array:
            lo, hi = lo_hi
            return jnp.any(hi - lo > tol)

        def body(lo_hi: tuple[Array, Array]) -> tuple[Array, Array]:
            mid = safe_avg(lo_hi)
            obj_derivative = h(mid)

            lo, hi = lo_hi
            _cond = jnp.all(obj_derivative < 0)
            new_lo = jlax.select(_cond, lo, mid)
            new_hi = jlax.select(_cond, mid, hi)

            return (new_lo, new_hi)

        min_val = jnp.asarray(0.0)  # mu is constrained to be >= 0
        max_val = jnp.asarray(bound)

        # run bisection
        lo_hi = jlax.while_loop(cond, body, (min_val, max_val))

        # final result
        mu = safe_avg(lo_hi)

        return get_c_i_tildes(mu)

    @eqx.filter_jit
    def project_sigma_and_clip(
        self,
        sigmas: Array,
        clips: Array,
    ) -> tuple[Array, Array]:
        """Project (sigmas, clips) onto the GDP privacy constraint in Euclidean space.

        Finds the nearest point (in L2) satisfying
            sum_i (exp((clip_i / sigma_i)^2) - 1) <= (mu/p)^2.

        Uses outer bisection on the dual variable lambda and a per-component
        2D Newton solve for the proximal subproblem, fully JIT-compatible.

        Args:
            sigmas: Per-step noise multipliers, shape (T,).
            clips: Per-step clipping thresholds, shape (T,).

        Returns:
            Tuple of (projected_sigmas, projected_clips), each shape (T,).
        """
        # Constraint: sum_i (exp(X_i^2/Y_i^2) - 1) <= B  with X=clips, Y=sigmas
        X, Y = clips, sigmas
        B = (self.mu / self.p) ** 2

        S = jnp.sum(jnp.exp((X / Y) ** 2) - 1.0)
        feasible = S <= B

        def _solve_for_lam(lam):
            X_p, Y_p = _sc_inner_solve_vec(X, Y, X, Y, lam)
            h = jnp.sum(jnp.exp((X_p / Y_p) ** 2) - 1.0) - B
            return h, X_p, Y_p

        # Find lambda_max by repeated doubling until h(lambda_max) <= 0.
        init_h, _, _ = _solve_for_lam(jnp.asarray(1.0))

        def _double_cond(state):
            _, h = state
            return h > 0

        def _double_body(state):
            lam_max, _ = state
            lam_max = lam_max * 2.0
            h, _, _ = _solve_for_lam(lam_max)
            return (lam_max, h)

        lam_max, _ = jlax.while_loop(_double_cond, _double_body, (jnp.asarray(1.0), init_h))

        # Bisect over [0, lam_max] for 60 iterations (~1e-18 relative precision).
        def bisect_body(_, state):
            lam_lo, lam_hi = state
            lam_mid = (lam_lo + lam_hi) / 2.0
            h, _, _ = _solve_for_lam(lam_mid)
            lam_lo = jnp.where(h > 0, lam_mid, lam_lo)
            lam_hi = jnp.where(h > 0, lam_hi, lam_mid)
            return (lam_lo, lam_hi)

        lam_lo, lam_hi = jlax.fori_loop(0, 60, bisect_body, (jnp.asarray(0.0), lam_max))

        lam_final = (lam_lo + lam_hi) / 2.0
        _, X_proj, Y_proj = _solve_for_lam(lam_final)

        # If already feasible, return originals unchanged.
        X_out = jnp.where(feasible, X, X_proj)
        Y_out = jnp.where(feasible, Y, Y_proj)

        # X=clips, Y=sigmas — return (sigmas, clips)
        return Y_out, X_out


def get_privacy_params(dataset_length: int) -> GDPPrivacyParameters:
    """Build a GDPPrivacyParameters instance from the current singleton config.

    Args:
        dataset_length: Number of training examples; used to compute the subsampling probability p.

    Returns:
        GDPPrivacyParameters for the configured (ε, δ, batch_size, T).
    """
    sweep_config = SingletonConfig.get_sweep_config_instance()
    epsilon = sweep_config.env.eps
    delta = sweep_config.env.delta
    p = sweep_config.env.batch_size / dataset_length
    T = SingletonConfig.get_environment_config_instance().max_steps_in_episode

    return GDPPrivacyParameters(epsilon, delta, p, T)
