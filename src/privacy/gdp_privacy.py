import jax.scipy.stats as jstats
import chex
import jax.numpy as jnp
from conf.singleton_conf import SingletonConfig
from scipy import optimize
import equinox as eqx
from util.util import pytree_has_inf
from jax.nn import softmax

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
        current_delta = (
            jstats.norm.cdf(-eps/current_mu + current_mu/2)
            - jnp.exp(eps) * jstats.norm.cdf(-eps/current_mu - current_mu/2)
        )
        return current_delta-delta    
    return optimize.root_scalar(f, bracket=[tol, 100], method='brentq').root


def weights_to_mu_schedule(mu: float, schedule: chex.Array, p: float, T: int) -> chex.Array:
    """Convert a GDP mu parameter to a Poisson subsampling schedule.

    Args:
        mu: The mu parameter of GDP. Assumed to be a non-negative scalar.
        schedule: A 1D array representing the initial schedule (e.g., learning rates). Assumed to be non-negative and sum to T.
        p: The subsampling probability. Assumed to be in (0, 1).
        T: The total number of steps. Assumed to be a positive integer.

    Returns:
        A 1D array representing the adjusted schedule.
    """

    sigma_s = SingletonConfig.get_policy_config_instance().sigma_s
    
    mu_0 = jnp.sqrt(jnp.log(mu**2 / (p**2 * T) + 1))
    schedule = eqx.error_if(schedule, (schedule == 0).any(axis=None), "Schedule has zeroes")
    return jnp.sqrt(jnp.log(schedule * (jnp.exp(mu_0 ** 2) - 1) + 1) / sigma_s ** 2)


def mu_schedule_to_weights(mu: float, schedule: chex.Array, p: float, T: int) -> chex.Array:
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

    sigma_s = SingletonConfig.get_policy_config_instance().sigma_s
    
    mu_0 = jnp.sqrt(jnp.log(mu**2 / (p**2 * T) + 1))
    return (jnp.exp((schedule * sigma_s) ** 2) - 1) / (jnp.exp(mu_0 ** 2) - 1)

def gdp_to_sigma(mu: chex.Array) -> chex.Array:
    """Convert GDP mu parameter to Gaussian noise scale sigma.

    Args:
        mu: The mu parameter of GDP. Assumed to be a non-negative array.

    Returns:
        The Gaussian noise scale sigma.
    """

    C = SingletonConfig.get_environment_config_instance().C
    return C / mu

def dsigma_dweight(sigmas: jnp.ndarray, mu, p, T) -> jnp.ndarray:
    """
    Compute ds / dp, where 's' is sigma and 'p' is the policy
    """

    mu_0 = jnp.sqrt(jnp.log(mu**2 / (p**2 * T) + 1))
    numerator = jnp.exp(mu_0**2) - 1
    denominator_pt1 = numerator * sigmas + 1
    denominator_pt2 = jnp.log(denominator_pt1) ** (3/2)
    
    return numerator / (2 * denominator_pt1 * denominator_pt2)


def weights_to_sigma_schedule(weights: chex.Array, mu, p, T):
    """Convert a vector of non-negative weights summing to T to a sigma schedule for G-DP. 

    Args:
        schedule: A 1D array representing the weights. Assumed to be non-negative and sum to T.
        mu: The mu parameter of GDP. Assumed to be a non-negative float or array.
        p: The sampling probability for poisson sampling
        T: The number of training iterations

    Returns:
        The Gaussian noise scale sigma.
    """
    weights = (T * jnp.exp(weights)) / jnp.sum(jnp.exp(weights), axis=-1, keepdims=True)
    mu_schedule = weights_to_mu_schedule(mu, weights, p, T)
    mu_schedule = eqx.error_if(mu_schedule, pytree_has_inf(mu_schedule), "New Sigmas has Inf!")
    mu_schedule = eqx.error_if(mu_schedule, (mu_schedule == 0).any(), "Some mus are 0!")
    return gdp_to_sigma(mu_schedule)


def sigma_schedule_to_weights(schedule: chex.Array, mu, p, T):
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


def test_approx_to_gdp():
    epsilon = 3.0
    delta = 0.566737999092
    mu = approx_to_gdp(epsilon, delta)
    assert jnp.isclose(mu, 3.0, atol=1e-3), f"Expected mu to be close to 3.0, but got {mu}"

    epsilon = 0.5
    delta = 0.0524403232877
    mu = approx_to_gdp(epsilon, delta)
    assert jnp.isclose(mu, 0.5, atol=1e-3), f"Expected mu to be close to 0.5, but got {mu}"

    epsilon = 1.0
    delta = 0.126936737507
    mu = approx_to_gdp(epsilon, delta)
    assert jnp.isclose(mu, 1.0, atol=1e-3), f"Expected mu to be close to 1.0, but got {mu}"

    epsilon = 7
    delta = 0.811589893405
    mu = approx_to_gdp(epsilon, delta)
    assert jnp.isclose(mu, 5.0, atol=1e-3), f"Expected mu to be close to 5.0, but got {mu}"


def test_weights_to_sigmas_and_back():
    def test(weights, mu, p, T):
        sigmas = weights_to_sigma_schedule(weights, mu, p, T)
        new_weights = sigma_schedule_to_weights(sigmas, mu, p, T)
        assert jnp.isclose(weights, new_weights).all(), f"{weights} =/= {new_weights}"

    test(jnp.asarray([1, 1, 1, 1]), mu=0.5, p=250/60_000, T=1000)
    test(jnp.asarray([5, 3, 0.05, 10.0]), mu=0.5, p=250/60_000, T=1000)
    test(jnp.asarray([5, 3, 0.05, 10.0]), mu=0.5, p=250/60_000, T=3000)
    test(jnp.asarray([5, 3, 0.05, 10.0]), mu=0.5, p=1/60_000, T=1000)
    test(jnp.asarray([5, 3, 0.05, 10.0]), mu=0.1, p=250/60_000, T=1000)


if __name__ == "__main__":
    test_approx_to_gdp()
    test_weights_to_sigmas_and_back()
    print("All tests passed.")