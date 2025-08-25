import jax.scipy.stats as jstats
import chex
from typing import Union
import jax.numpy as jnp

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

    low = 0.0
    high = 100.0  # Arbitrary upper bound for mu

    while high - low > tol:
        mid = (low + high) / 2
        computed_delta = jstats.norm.cdf(-eps/mid + mid/2) - jnp.exp(eps) * jstats.norm.cdf(-eps/mid - mid/2)
        if computed_delta < delta:
            low = mid
        else:
            high = mid
    
    return mid


def mu_to_poisson_subsampling_shedule(mu: float, schedule: chex.Array, p: float, T: int) -> chex.Array:
    """Convert a GDP mu parameter to a Poisson subsampling schedule.

    Args:
        mu: The mu parameter of GDP. Assumed to be a non-negative scalar.
        schedule: A 1D array representing the initial schedule (e.g., learning rates). Assumed to be non-negative and sum to 1.
        p: The subsampling probability. Assumed to be in (0, 1).
        T: The total number of steps. Assumed to be a positive integer.

    Returns:
        A 1D array representing the adjusted schedule.
    """
    
    return jnp.sqrt(jnp.log(schedule * mu ** 2 / (p ** 2 * T) + 1))


def gdp_to_sigma(mu: chex.Array) -> chex.Array:
    """Convert GDP mu parameter to Gaussian noise scale sigma.

    Args:
        mu: The mu parameter of GDP. Assumed to be a non-negative array.

    Returns:
        The Gaussian noise scale sigma.
    """
    
    return 1 / mu


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


if __name__ == "__main__":
    test_approx_to_gdp()
    print("All tests passed.")