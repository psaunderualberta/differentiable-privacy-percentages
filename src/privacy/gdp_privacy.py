import equinox as eqx
import jax.lax as jlax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import optax
from jaxtyping import Array
from scipy import optimize

from conf.singleton_conf import SingletonConfig
from util.util import pytree_has_inf


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
            eps
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
        return jnp.sqrt(jnp.log(self.mu**2 / (self.p**2 * self.T) + 1))

    def compute_eps(self, max_sigma: float | None = None) -> Array:
        if max_sigma is None:
            max_sigma = SingletonConfig.get_policy_config_instance().max_sigma

        return (jnp.exp(1 / max_sigma**2) - 1) / (jnp.exp(self.mu_0**2) - 1)

    def compute_expenditure(self, sigmas: Array, clips: Array) -> Array:
        return jnp.sum(self.p * jnp.sqrt(jnp.exp((clips / sigmas) ** 2) - 1))

    def weights_to_mu_schedule(self, schedule: Array) -> Array:
        """Convert a GDP mu parameter to a Poisson subsampling schedule.

        Args:
            schedule: A 1D array representing the initial schedule (e.g., learning rates). Assumed to be non-negative and sum to T.

        Returns:
            A 1D array representing the adjusted schedule.
        """

        eps = self.compute_eps()

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

    def weights_to_sigma_schedule(self, C: Array, weights: Array) -> Array:
        """Convert a vector of non-negative weights summing to T to a sigma schedule for G-DP.

        Args:
            schedule: A 1D array representing the weights. Assumed to be non-negative and sum to T.
            mu: The mu parameter of GDP. Assumed to be a non-negative float or array.
            p: The sampling probability for poisson sampling
            T: The number of training iterations

        Returns:
            The Gaussian noise scale sigma.
        """
        weights = eqx.error_if(weights, pytree_has_inf(weights), "weights have Inf!")
        weights = eqx.error_if(weights, (weights == 0).any(), "weights have 0!")
        mu_schedule = self.weights_to_mu_schedule(weights)
        mu_schedule = eqx.error_if(
            mu_schedule, pytree_has_inf(mu_schedule), "mu schedule has Inf!"
        )
        mu_schedule = eqx.error_if(
            mu_schedule, (mu_schedule == 0).any(), "Some mus are 0!"
        )
        return self.gdp_to_sigma(C, mu_schedule)

    def weights_to_clip_schedule(self, sigmas: Array, weights: Array) -> Array:
        """Convert a vector of non-negative weights summing to T to a sigma schedule for G-DP.

        Args:
            schedule: A 1D array representing the weights. Assumed to be non-negative and sum to T.
            mu: The mu parameter of GDP. Assumed to be a non-negative float or array.
            p: The sampling probability for poisson sampling
            T: The number of training iterations

        Returns:
            The Gaussian noise scale sigma.
        """
        weights = eqx.error_if(weights, pytree_has_inf(weights), "weights have Inf!")
        weights = eqx.error_if(weights, (weights == 0).any(), "weights have 0!")
        mu_schedule = self.weights_to_mu_schedule(weights)
        mu_schedule = eqx.error_if(
            mu_schedule, pytree_has_inf(mu_schedule), "mu schedule has Inf!"
        )
        mu_schedule = eqx.error_if(
            mu_schedule, (mu_schedule == 0).any(), "Some mus are 0!"
        )
        return sigmas * mu_schedule

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
            schedule, jnp.isinf(C / schedule).any(), "schedule has Inf!"
        )
        return self.mu_schedule_to_weights(C / schedule)

    # TODO: Move projection into the gradient computation
    def project_weights(self, weights: Array, tol: float | Array = 1e-6) -> Array:
        """
        post-GD weights (i.e. after updating sigma, clip, policy, or any three)
        returns: projected weights W s.t. 1^T @ W = self.T && W \in [self.w_min, self.w_max]
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
            return jnp.all(f_ > tol)

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
                c_i_tildes_cond, c_i_tildes_body, (weights, mu)
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
        max_val = jnp.asarray(5.0)

        # run bisection
        lo_hi = jlax.while_loop(cond, body, (min_val, max_val))

        # final result
        mu = safe_avg(lo_hi)

        return get_c_i_tildes(mu)


def get_privacy_params(dataset_length: int) -> GDPPrivacyParameters:
    sweep_config = SingletonConfig.get_sweep_config_instance()
    epsilon = sweep_config.env.eps
    delta = sweep_config.env.delta
    p = sweep_config.env.batch_size / dataset_length
    T = SingletonConfig.get_environment_config_instance().max_steps_in_episode

    return GDPPrivacyParameters(epsilon, delta, p, T)

    # def h(mu: Array) -> Array:
    #     # 'mu' as in KKT literature, not privacy literature
    #     clipped = jnp.clip(weights - mu, self.w_min, self.w_max)
    #     return jnp.sum(clipped) - weights.size

    # def cond(lo_hi: tuple[Array, Array]) -> Array:
    #     lo, hi = lo_hi
    #     return jnp.any(hi - lo > tol)

    # def body(lo_hi: tuple[Array, Array]) -> tuple[Array, Array]:
    #     mid = safe_avg(lo_hi)
    #     obj_derivative = h(mid)

    #     lo, hi = lo_hi
    #     _cond = jnp.all(obj_derivative < 0)
    #     new_lo = jlax.select(_cond, lo, mid)
    #     new_hi = jlax.select(_cond, mid, hi)

    #     return (new_lo, new_hi)

    # # initial boundary values for bisection
    # # dh/dmu outside this interval is 0, i.e. everything is clipped
    # min_val = jnp.min(weights) - self.w_max
    # max_val = jnp.max(weights) - self.w_min
