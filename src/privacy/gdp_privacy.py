import jax.scipy.stats as jstats
from jaxtyping import Array, PRNGKeyArray
import chex
import jax.numpy as jnp
from conf.singleton_conf import SingletonConfig
from scipy import optimize
import equinox as eqx
from util.util import pytree_has_inf
from jax.nn import softmax
import optax


def approx_to_gdp(eps: float, delta: float, tol: float = 1e-6) -> float:
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


def compute_mu_0(mu: float, p: float, T: int) -> Array:
    return jnp.sqrt(jnp.log(mu**2 / (p**2 * T) + 1))


def compute_eps(mu: float, p: float, T: int, max_sigma: float | None = None) -> Array:
    mu_0 = compute_mu_0(mu, p, T)

    if max_sigma is None:
        max_sigma = SingletonConfig.get_policy_config_instance().max_sigma

    return (jnp.exp(1 / max_sigma**2) - 1) / (jnp.exp(mu_0**2) - 1)


def weights_to_mu_schedule(mu: float, schedule: Array, p: float, T: int) -> Array:
    """Convert a GDP mu parameter to a Poisson subsampling schedule.

    Args:
        mu: The mu parameter of GDP. Assumed to be a non-negative scalar.
        schedule: A 1D array representing the initial schedule (e.g., learning rates). Assumed to be non-negative and sum to T.
        p: The subsampling probability. Assumed to be in (0, 1).
        T: The total number of steps. Assumed to be a positive integer.

    Returns:
        A 1D array representing the adjusted schedule.
    """

    eps = compute_eps(mu, p, T)
    schedule = schedule**2 + eps

    mu_0 = compute_mu_0(mu, p, T)
    schedule = eqx.error_if(
        schedule, (schedule == 0).any(axis=None), "Schedule has zeroes"
    )
    return jnp.sqrt(jnp.log(schedule * (jnp.exp(mu_0**2) - 1) + 1))


def mu_schedule_to_weights(mu: float, schedule: Array, p: float, T: int) -> Array:
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

    mu_0 = compute_mu_0(mu, p, T)
    return (jnp.exp(schedule**2) - 1) / (jnp.exp(mu_0**2) - 1)


def gdp_to_sigma(mu: Array) -> Array:
    """Convert GDP mu parameter to Gaussian noise scale sigma.

    Args:
        mu: The mu parameter of GDP. Assumed to be a non-negative array.

    Returns:
        The Gaussian noise scale sigma.
    """

    C = SingletonConfig.get_environment_config_instance().C
    return C / mu


def dsigma_dweight(sigmas: Array, mu, p, T) -> Array:
    """
    Compute ds / dp, where 's' is sigma and 'p' is the policy
    """

    mu_0 = jnp.sqrt(jnp.log(mu**2 / (p**2 * T) + 1))
    numerator = jnp.exp(mu_0**2) - 1
    denominator_pt1 = numerator * sigmas + 1
    denominator_pt2 = jnp.log(denominator_pt1) ** (3 / 2)

    return numerator / (2 * denominator_pt1 * denominator_pt2)


def weights_to_sigma_schedule(weights: Array, mu: float, p: float, T: int) -> Array:
    """Convert a vector of non-negative weights summing to T to a sigma schedule for G-DP.

    Args:
        schedule: A 1D array representing the weights. Assumed to be non-negative and sum to T.
        mu: The mu parameter of GDP. Assumed to be a non-negative float or array.
        p: The sampling probability for poisson sampling
        T: The number of training iterations

    Returns:
        The Gaussian noise scale sigma.
    """
    mu_schedule = weights_to_mu_schedule(mu, weights, p, T)
    mu_schedule = eqx.error_if(
        mu_schedule, pytree_has_inf(mu_schedule), "New Sigmas has Inf!"
    )
    mu_schedule = eqx.error_if(mu_schedule, (mu_schedule == 0).any(), "Some mus are 0!")
    return gdp_to_sigma(mu_schedule)


def sigma_schedule_to_weights(schedule: Array, mu, p, T):
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
    C = SingletonConfig.get_environment_config_instance().C
    mus = C / schedule
    return mu_schedule_to_weights(mu, mus, p, T)


# TODO: Move projection into the gradient computation
def project_weights(weights: Array, mu: float, p: float, T: int) -> Array:
    """
    W: in the form w**2 + eps
    """
    eps = compute_eps(mu, p, T)

    # project to l2 ball
    mu_0 = compute_mu_0(mu, p, T)
    l2_ball_radius = jnp.sqrt(mu**2 / (p**2 * (jnp.exp(mu_0**2) - 1)) - T * eps)
    projected_weights = optax.projections.projection_l2_ball(
        weights, scale=l2_ball_radius
    )

    # reshift to ensure no mu is 0
    return projected_weights
