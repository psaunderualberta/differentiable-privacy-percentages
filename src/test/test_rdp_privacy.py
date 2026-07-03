"""Tests for privacy/rdp_privacy.py — the RDPPrivacyParameters budget object.

Holds (eps, delta, p, T, candidate alphas, binding alpha*) and exposes the
differentiable budget constraint g(w) = rho_total(alpha*; w) - c(alpha*), where
c(alpha*) = eps - log(1/delta)/(alpha* - 1) is the RDP budget at the binding
order (ADR-0007/0008). Weights are the accountant's w = 1/sigma_mult.
"""

import math

import jax.numpy as jnp


class TestWithAlphaStar:
    def test_matches_select_optimal_alpha(self):
        from privacy.rdp_accountant import select_optimal_alpha
        from privacy.rdp_privacy import RDPPrivacyParameters

        eps, delta, p, T = 1.0, 1e-5, 0.01, 5
        alphas = (2, 4, 8, 16, 32, 64)
        weights = jnp.asarray([1.0 / s for s in [1.5, 2.0, 2.5, 4.0, 8.0]])

        params = RDPPrivacyParameters(eps, delta, p, T, alphas=alphas)
        updated = params.with_alpha_star(weights)

        expected = select_optimal_alpha(list(alphas), weights, p, delta)
        assert updated.alpha_star == expected
        # Original is unchanged (immutable update).
        assert params.alpha_star == alphas[0]


class TestProjectScale:
    def test_result_is_on_budget(self):
        from privacy.rdp_accountant import rdp_to_epsilon
        from privacy.rdp_privacy import RDPPrivacyParameters

        eps, delta, p, T = 1.0, 1e-5, 0.01, 5
        alphas = (2, 4, 8, 16, 32, 64)
        # Under-noised (over budget) schedule.
        sigma_mults = jnp.asarray([1.0, 1.2, 0.8, 1.5, 1.1])

        params = RDPPrivacyParameters(eps, delta, p, T, alphas=alphas)

        scaled = params.project_scale(sigma_mults)
        # On budget: the min-over-alpha conversion realises exactly eps at delta.
        got_eps = rdp_to_epsilon(list(alphas), 1.0 / scaled, p, delta)
        assert jnp.isclose(got_eps, eps, atol=1e-4)

    def test_constraint_zero_at_binding_order_on_budget(self):
        # At the on-budget point, the binding order alpha* has c(alpha*) =
        # rho_total(alpha*) >= 0, so the single-alpha constraint is ~0 there.
        from privacy.rdp_privacy import RDPPrivacyParameters

        sigma_mults = jnp.asarray([1.0, 1.2, 0.8, 1.5, 1.1])
        params = RDPPrivacyParameters(1.0, 1e-5, 0.01, 5, alphas=(2, 4, 8, 16, 32, 64))
        scaled = params.project_scale(sigma_mults)
        params = params.with_alpha_star(1.0 / scaled)
        assert jnp.isclose(params.constraint(1.0 / scaled), 0.0, atol=1e-4)

    def test_scaling_is_a_common_factor(self):
        # The retraction multiplies every sigma_mult by the same scalar (exact in
        # the BSpline family), so ratios between entries are preserved.
        from privacy.rdp_privacy import RDPPrivacyParameters

        sigma_mults = jnp.asarray([1.0, 1.2, 0.8, 1.5, 1.1])
        params = RDPPrivacyParameters(1.0, 1e-5, 0.01, 5, alphas=(2, 4, 8, 16, 32, 64))
        params = params.with_alpha_star(1.0 / sigma_mults)

        scaled = params.project_scale(sigma_mults)
        ratios = scaled / sigma_mults
        assert jnp.allclose(ratios, ratios[0], rtol=1e-5)


class TestConstraint:
    def test_matches_rho_total_minus_budget(self):
        from privacy.rdp_accountant import rdp_total
        from privacy.rdp_privacy import RDPPrivacyParameters

        eps, delta, p, T = 1.0, 1e-5, 0.01, 5
        alphas = (2, 4, 8, 16, 32)
        alpha_star = 8
        weights = jnp.asarray([1.0 / s for s in [1.5, 2.0, 2.5, 4.0, 8.0]])

        params = RDPPrivacyParameters(eps, delta, p, T, alphas=alphas, alpha_star=alpha_star)

        budget = eps - math.log(1.0 / delta) / (alpha_star - 1)
        expected = float(rdp_total(alpha_star, weights, p)) - budget
        got = float(params.constraint(weights))
        assert jnp.isclose(jnp.asarray(got), jnp.asarray(expected), rtol=1e-6)
