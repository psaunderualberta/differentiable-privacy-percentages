from privacy.gdp_privacy import (
    approx_to_gdp,
    weights_to_sigma_schedule,
    sigma_schedule_to_weights,
    compute_mu_0,
    compute_eps,
)
import jax.numpy as jnp


def test_approx_to_gdp():
    epsilon = 3.0
    delta = 0.566737999092
    mu = approx_to_gdp(epsilon, delta)
    assert jnp.isclose(mu, 3.0, atol=1e-3), (
        f"Expected mu to be close to 3.0, but got {mu}"
    )

    epsilon = 0.5
    delta = 0.0524403232877
    mu = approx_to_gdp(epsilon, delta)
    assert jnp.isclose(mu, 0.5, atol=1e-3), (
        f"Expected mu to be close to 0.5, but got {mu}"
    )

    epsilon = 1.0
    delta = 0.126936737507
    mu = approx_to_gdp(epsilon, delta)
    assert jnp.isclose(mu, 1.0, atol=1e-3), (
        f"Expected mu to be close to 1.0, but got {mu}"
    )

    epsilon = 7
    delta = 0.811589893405
    mu = approx_to_gdp(epsilon, delta)
    assert jnp.isclose(mu, 5.0, atol=1e-3), (
        f"Expected mu to be close to 5.0, but got {mu}"
    )


def test_compute_mu_0():
    pass


def test_compute_eps():
    (eps, delta) = (0.4, 1e-7)
    print(approx_to_gdp(eps, delta))
    (mu, p, T, max_sigma) = (0.09, 250 / 60_000, 300, 20.0)

    est_eps = compute_eps(mu, p, T, max_sigma=max_sigma)
    mu_0 = compute_mu_0(mu, p, T)
    exp_eps = (jnp.exp(1 / max_sigma**2) - 1) / (jnp.exp(mu_0**2) - 1)

    assert jnp.isclose(est_eps, exp_eps, atol=1e-3), (
        f"Expected epsilon to be close to {exp_eps}, but got {est_eps}"
    )
