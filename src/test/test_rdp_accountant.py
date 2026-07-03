"""Tests for privacy/rdp_accountant.py.

The accountant is the fixed-size-without-replacement + replace-one (sensitivity
2C) subsampled-Gaussian RDP, written from scratch in JAX (ADR-0007). The primary
oracle is `autodp` (Theorem 9 of Wang-Balle-Kasiviswanathan, arXiv:1808.00087),
the same closed-form bound reproduced here.

DP-PSAC mapping: noise = C * sigma_mult, sensitivity = 2C, so the noise-to-
sensitivity multiplier is s = sigma_mult / 2 = 1 / (2w) with w = 1 / sigma_mult.
The base Gaussian RDP is therefore func(alpha) = 2 * alpha * w**2.
"""

import math
import warnings

import jax
import jax.numpy as jnp
import pytest


def _autodp_rho_step(sigma_mult: float, p: float, alpha: int) -> float:
    """Reference amplified per-step RDP at integer order `alpha` from autodp."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from autodp.mechanism_zoo import GaussianMechanism
        from autodp.transformer_zoo import AmplificationBySampling

    s = sigma_mult / 2.0  # noise-to-sensitivity multiplier under 2C
    gm = GaussianMechanism(sigma=s, name="gm", RDP_off=False)
    gm.neighboring = "replace_one"
    sub = AmplificationBySampling(PoissonSampling=False)
    amp = sub.amplify(gm, p, improved_bound_flag=False)
    return float(amp.RenyiDP(alpha))


# ---------------------------------------------------------------------------
# subsampled_gaussian_rdp_step — matches the autodp WOR bound
# ---------------------------------------------------------------------------


class TestSubsampledGaussianRDPStep:
    @pytest.mark.parametrize("alpha", [2, 3, 4, 8, 16, 32])
    def test_matches_autodp(self, alpha):
        from privacy.rdp_accountant import subsampled_gaussian_rdp_step

        sigma_mult, p = 2.0, 0.01
        w = 1.0 / sigma_mult
        expected = _autodp_rho_step(sigma_mult, p, alpha)
        got = float(subsampled_gaussian_rdp_step(alpha, jnp.asarray(w), p))
        assert jnp.isclose(jnp.asarray(got), jnp.asarray(expected), rtol=1e-5, atol=1e-10)


# ---------------------------------------------------------------------------
# rdp_total — additive composition over a (non-uniform) schedule
# ---------------------------------------------------------------------------


class TestRDPTotal:
    @pytest.mark.parametrize("alpha", [2, 8, 32])
    def test_matches_autodp_composition(self, alpha):
        from privacy.rdp_accountant import rdp_total

        p = 0.01
        sigma_mults = [1.5, 2.0, 2.5, 4.0, 8.0]  # non-uniform schedule
        weights = jnp.asarray([1.0 / s for s in sigma_mults])
        expected = sum(_autodp_rho_step(s, p, alpha) for s in sigma_mults)
        got = float(rdp_total(alpha, weights, p))
        assert jnp.isclose(jnp.asarray(got), jnp.asarray(expected), rtol=1e-5, atol=1e-10)


# ---------------------------------------------------------------------------
# rdp_to_epsilon — RDP -> (eps, delta) conversion, min over integer orders
# ---------------------------------------------------------------------------


class TestRDPToEpsilon:
    def test_matches_autodp_conversion_over_same_orders(self):
        from privacy.rdp_accountant import rdp_to_epsilon

        p, delta = 0.01, 1e-5
        sigma_mults = [1.5, 2.0, 2.5, 4.0, 8.0]
        weights = jnp.asarray([1.0 / s for s in sigma_mults])
        alphas = [2, 4, 8, 16, 32, 64]

        # Oracle: same min_alpha[rho_total(alpha) + log(1/delta)/(alpha-1)]
        # conversion applied to autodp's per-order composed RDP.
        import math

        expected = min(
            sum(_autodp_rho_step(s, p, a) for s in sigma_mults) + math.log(1.0 / delta) / (a - 1)
            for a in alphas
        )
        got = float(rdp_to_epsilon(alphas, weights, p, delta))
        assert jnp.isclose(jnp.asarray(got), jnp.asarray(expected), rtol=1e-5, atol=1e-8)


# ---------------------------------------------------------------------------
# select_optimal_alpha — the binding integer order alpha* (adaptive per step)
# ---------------------------------------------------------------------------


class TestSelectOptimalAlpha:
    def test_returns_argmin_order(self):
        from privacy.rdp_accountant import select_optimal_alpha

        p, delta = 0.01, 1e-5
        sigma_mults = [1.5, 2.0, 2.5, 4.0, 8.0]
        weights = jnp.asarray([1.0 / s for s in sigma_mults])
        alphas = [2, 4, 8, 16, 32, 64]

        expected = min(
            alphas,
            key=lambda a: (
                sum(_autodp_rho_step(s, p, a) for s in sigma_mults)
                + math.log(1.0 / delta) / (a - 1)
            ),
        )
        got = select_optimal_alpha(alphas, weights, p, delta)
        assert got == expected

    def test_selected_alpha_realises_epsilon(self):
        # eps at the selected order must equal the min-over-orders eps.
        from privacy.rdp_accountant import rdp_to_epsilon, rdp_total, select_optimal_alpha

        p, delta = 0.01, 1e-5
        weights = jnp.asarray([1.0 / s for s in [1.5, 2.0, 2.5, 4.0, 8.0]])
        alphas = [2, 4, 8, 16, 32, 64]

        astar = select_optimal_alpha(alphas, weights, p, delta)
        eps_at_astar = float(rdp_total(astar, weights, p) + math.log(1.0 / delta) / (astar - 1))
        assert jnp.isclose(
            jnp.asarray(eps_at_astar),
            jnp.asarray(float(rdp_to_epsilon(alphas, weights, p, delta))),
            rtol=1e-6,
        )


# ---------------------------------------------------------------------------
# rdp_total is JIT-compatible and differentiable w.r.t. the noise weights.
# The Riemannian projection (ADR-0008) gets its manifold normal by autodiff of
# the constraint scalar through rho_total, so this must hold.
# ---------------------------------------------------------------------------


class TestJitAndGrad:
    def test_rdp_total_jittable(self):
        from privacy.rdp_accountant import rdp_total

        weights = jnp.asarray([1.0 / s for s in [1.5, 2.0, 4.0]])
        jitted = jax.jit(lambda w: rdp_total(8, w, 0.01))
        assert jnp.isclose(jitted(weights), rdp_total(8, weights, 0.01), rtol=1e-6)

    def test_rdp_total_grad_finite_and_increasing(self):
        from privacy.rdp_accountant import rdp_total

        # rho_total increases in each weight (less noise -> more privacy cost),
        # so d rho_total / d w_i > 0 and finite.
        weights = jnp.asarray([1.0 / s for s in [1.5, 2.0, 4.0]])
        grad = jax.grad(lambda w: rdp_total(8, w, 0.01))(weights)
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.all(grad > 0)
