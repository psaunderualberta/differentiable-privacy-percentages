import jax.scipy.stats as jstats
from jaxtyping import Array, PRNGKeyArray
import chex
import jax.numpy as jnp
from conf.singleton_conf import SingletonConfig
from scipy import optimize
import equinox as eqx
from util.util import pytree_has_inf
import jax.lax as jlax
from jax.nn import softmax
import optax


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

    def __init__(self, eps: float, delta: float, p: float, T: int):
        self.eps = eps
        self.delta = delta
        self.__mu = approx_to_gdp(eps, delta)
        self.__p = p
        self.__T = T
        self.__mu_0 = self.compute_mu_0()

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
    
    def compute_mu_0(self) -> Array:
        return jnp.sqrt(jnp.log(self.mu**2 / (self.p**2 * self.T) + 1))

    def compute_eps(self, max_sigma: float | None = None) -> Array:

        if max_sigma is None:
            max_sigma = SingletonConfig.get_policy_config_instance().max_sigma

        return (jnp.exp(1 / max_sigma**2) - 1) / (jnp.exp(self.mu_0**2) - 1)
    
    def weights_to_mu_schedule(self, schedule: Array) -> Array:
        """Convert a GDP mu parameter to a Poisson subsampling schedule.

        Args:
            schedule: A 1D array representing the initial schedule (e.g., learning rates). Assumed to be non-negative and sum to T.

        Returns:
            A 1D array representing the adjusted schedule.
        """

        eps = self.compute_eps()

        schedule = schedule**2 + eps
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
        eps = self.compute_eps()
        return jnp.sqrt((jnp.exp(schedule**2) - 1) / (jnp.exp(self.mu_0**2) - 1) - eps)


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
        mu_schedule = self.weights_to_mu_schedule(weights)
        mu_schedule = eqx.error_if(
            mu_schedule, pytree_has_inf(mu_schedule), "New Sigmas has Inf!"
        )
        mu_schedule = eqx.error_if(mu_schedule, (mu_schedule == 0).any(), "Some mus are 0!")
        return self.gdp_to_sigma(C, mu_schedule)


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
        return self.mu_schedule_to_weights(C / schedule)


    # TODO: Move projection into the gradient computation
    def project_weights(self, weights: Array) -> Array:
        """
        W: in the form w**2 + eps
        """
        eps = self.compute_eps()

        # project to l2 sphere
        l2_sphere_radius = jnp.sqrt(self.mu**2 / (self.p**2 * (jnp.exp(self.mu_0**2) - 1)) - self.T * eps)
        projected_weights = optax.projections.projection_l2_sphere(
            weights, scale=l2_sphere_radius
        )

        return projected_weights
