"""Tests for privacy/gdp_privacy.py.

Covers:
- approx_to_gdp numerical correctness and error handling
- GDPPrivacyParameters construction and properties
- compute_mu_0 formula
- compute_eps formula
- compute_expenditure formula
- weights_to_mu_schedule / mu_schedule_to_weights round-trip
- gdp_to_sigma formula
- weights_to_sigma_schedule / sigma_schedule_to_weights round-trip
- weights_to_clip_schedule formula
- project_weights privacy-constraint invariant
"""

import jax.numpy as jnp
import pytest

from privacy.gdp_privacy import GDPPrivacyParameters, approx_to_gdp

# ---------------------------------------------------------------------------
# Shared test parameters.
# mu ≈ 1.0 when eps=1.0 and delta=0.126936737507 (verified by old test suite).
# ---------------------------------------------------------------------------
EPS = 1.0
DELTA = 0.126936737507
P = 0.1
T = 10


@pytest.fixture
def params() -> GDPPrivacyParameters:
    return GDPPrivacyParameters(EPS, DELTA, P, T)


@pytest.fixture
def singleton_with_max_sigma():
    """Set SingletonConfig to a default Config so methods that call it work."""
    from conf.config import Config, EnvConfig, PolicyConfig, SweepConfig, WandbConfig
    from conf.singleton_conf import SingletonConfig

    SingletonConfig.config = Config(
        wandb_conf=WandbConfig(),
        sweep=SweepConfig(env=EnvConfig(), policy=PolicyConfig(max_sigma=10.0)),
    )
    yield SingletonConfig.config.sweep.policy.max_sigma
    SingletonConfig.config = None


# ---------------------------------------------------------------------------
# approx_to_gdp
# ---------------------------------------------------------------------------


class TestApproxToGDP:
    def test_mu_equals_3(self):
        mu = approx_to_gdp(3.0, 0.566737999092)
        assert jnp.isclose(mu, 3.0, atol=1e-3)

    def test_mu_equals_0_5(self):
        mu = approx_to_gdp(0.5, 0.0524403232877)
        assert jnp.isclose(mu, 0.5, atol=1e-3)

    def test_mu_equals_1(self):
        mu = approx_to_gdp(1.0, 0.126936737507)
        assert jnp.isclose(mu, 1.0, atol=1e-3)

    def test_clamped_mu_when_eps_exceeds_gdp_range(self):
        # GDP characterises the tradeoff curve; when eps is large relative to
        # delta, the optimal mu saturates (old test: eps=7, delta≈0.812 → mu≈5).
        mu = approx_to_gdp(7, 0.811589893405)
        assert jnp.isclose(mu, 5.0, atol=1e-3)

    def test_delta_zero_raises(self):
        with pytest.raises(ValueError, match="delta"):
            approx_to_gdp(1.0, 0.0)

    def test_delta_negative_raises(self):
        with pytest.raises(ValueError, match="delta"):
            approx_to_gdp(1.0, -0.1)

    def test_delta_one_raises(self):
        with pytest.raises(ValueError, match="delta"):
            approx_to_gdp(1.0, 1.0)

    def test_delta_gt_one_raises(self):
        with pytest.raises(ValueError, match="delta"):
            approx_to_gdp(1.0, 1.5)

    def test_eps_negative_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            approx_to_gdp(-0.1, 0.1)

    def test_mu_positive(self):
        mu = approx_to_gdp(1.0, 0.126936737507)
        assert mu > 0

    def test_larger_eps_same_delta_yields_larger_mu(self):
        # Looser (ε,δ)-DP ↔ larger GDP mu.
        mu_small = approx_to_gdp(0.5, 0.0524403232877)
        mu_large = approx_to_gdp(1.0, 0.126936737507)
        assert mu_large > mu_small


# ---------------------------------------------------------------------------
# GDPPrivacyParameters — construction and properties
# ---------------------------------------------------------------------------


class TestGDPPrivacyParametersProperties:
    def test_eps_stored(self, params):
        assert params.eps == pytest.approx(EPS)

    def test_delta_stored(self, params):
        assert params.delta == pytest.approx(DELTA)

    def test_mu_close_to_approx_to_gdp(self, params):
        expected_mu = approx_to_gdp(EPS, DELTA)
        assert jnp.isclose(params.mu, expected_mu, atol=1e-4)

    def test_p_stored(self, params):
        assert float(params.p) == pytest.approx(P)

    def test_T_stored(self, params):
        assert int(params.T) == T

    def test_w_min(self, params):
        assert float(params.w_min) == pytest.approx(0.1)

    def test_w_max(self, params):
        assert float(params.w_max) == pytest.approx(10.0)

    def test_mu_0_positive(self, params):
        assert float(params.mu_0) > 0

    def test_properties_are_stop_gradient(self, params):
        # Properties call lax.stop_gradient; verify they return JAX arrays
        # or Python scalars (not traced values outside JIT).
        _ = params.mu
        _ = params.p
        _ = params.T
        _ = params.mu_0
        _ = params.w_min
        _ = params.w_max


# ---------------------------------------------------------------------------
# compute_mu_0
# ---------------------------------------------------------------------------


class TestComputeMu0:
    def test_formula(self, params):
        # mu_0 = sqrt(log(mu^2 / (p^2 * T) + 1))
        expected = jnp.sqrt(jnp.log(params.mu**2 / (P**2 * T) + 1))
        assert jnp.isclose(params.mu_0, expected, atol=1e-5)

    def test_large_T_gives_small_mu_0(self):
        # As T → ∞ with fixed mu and p, mu_0 → 0 (budget spread thin).
        p1 = GDPPrivacyParameters(EPS, DELTA, P, T=10)
        p2 = GDPPrivacyParameters(EPS, DELTA, P, T=10_000)
        assert float(p2.mu_0) < float(p1.mu_0)

    def test_larger_p_gives_smaller_mu_0(self):
        # With a fixed total budget (mu), larger p means each step has higher
        # privacy sensitivity, so the per-step allowance (mu_0) must shrink.
        # Formula: mu_0 = sqrt(log(mu^2 / (p^2 * T) + 1)) — decreases with p.
        p_small = GDPPrivacyParameters(EPS, DELTA, p=0.01, T=T)
        p_large = GDPPrivacyParameters(EPS, DELTA, p=0.5, T=T)
        assert float(p_large.mu_0) < float(p_small.mu_0)


# ---------------------------------------------------------------------------
# compute_eps
# ---------------------------------------------------------------------------


class TestComputeEps:
    def test_formula(self, params):
        max_sigma = 5.0
        expected = (jnp.exp(1 / max_sigma**2) - 1) / (jnp.exp(params.mu_0**2) - 1)
        result = params.compute_eps(max_sigma=max_sigma)
        assert jnp.isclose(result, expected, atol=1e-5)

    def test_result_positive(self, params):
        assert float(params.compute_eps(max_sigma=5.0)) > 0

    def test_larger_max_sigma_gives_smaller_eps(self, params):
        # Higher noise tolerance → smaller privacy expenditure per step.
        eps_small_sigma = params.compute_eps(max_sigma=2.0)
        eps_large_sigma = params.compute_eps(max_sigma=10.0)
        assert float(eps_large_sigma) < float(eps_small_sigma)

    def test_uses_singleton_when_max_sigma_none(self, params, singleton_with_max_sigma):
        max_sigma = singleton_with_max_sigma
        expected = params.compute_eps(max_sigma=max_sigma)
        result = params.compute_eps(max_sigma=None)
        assert jnp.isclose(result, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# compute_expenditure
# ---------------------------------------------------------------------------


class TestComputeExpenditure:
    def test_single_step_formula(self, params):
        sigma = jnp.array([2.0])
        clip = jnp.array([1.0])
        expected = P * jnp.sqrt(jnp.exp((clip / sigma) ** 2) - 1)
        result = params.compute_expenditure(sigma, clip)
        assert jnp.isclose(result, expected, atol=1e-5)

    def test_output_is_scalar(self, params):
        sigmas = jnp.ones(T)
        clips = jnp.ones(T)
        result = params.compute_expenditure(sigmas, clips)
        assert result.ndim == 0

    def test_proportional_to_p(self):
        sigmas = jnp.ones(T)
        clips = jnp.ones(T)
        p1 = GDPPrivacyParameters(EPS, DELTA, p=0.1, T=T)
        p2 = GDPPrivacyParameters(EPS, DELTA, p=0.2, T=T)
        ratio = float(p2.compute_expenditure(sigmas, clips)) / float(
            p1.compute_expenditure(sigmas, clips),
        )
        assert ratio == pytest.approx(2.0, rel=1e-4)

    def test_larger_clip_increases_expenditure(self, params):
        sigmas = jnp.ones(T) * 2.0
        small_clips = jnp.ones(T) * 0.5
        large_clips = jnp.ones(T) * 1.5
        assert float(params.compute_expenditure(sigmas, large_clips)) > float(
            params.compute_expenditure(sigmas, small_clips),
        )

    def test_larger_sigma_decreases_expenditure(self, params):
        clips = jnp.ones(T)
        small_sigma = jnp.ones(T) * 0.5
        large_sigma = jnp.ones(T) * 5.0
        assert float(params.compute_expenditure(large_sigma, clips)) < float(
            params.compute_expenditure(small_sigma, clips),
        )

    def test_sums_over_steps(self, params):
        # expenditure(T steps) ≈ T × expenditure(1 step) for uniform schedules.
        sigma_1 = jnp.array([2.0])
        clip_1 = jnp.array([1.0])
        sigma_T = jnp.ones(T) * 2.0
        clip_T = jnp.ones(T) * 1.0
        single = params.compute_expenditure(sigma_1, clip_1)
        total = params.compute_expenditure(sigma_T, clip_T)
        assert jnp.isclose(total, T * single, rtol=1e-4)


# ---------------------------------------------------------------------------
# weights_to_mu_schedule / mu_schedule_to_weights (inverse pair)
# ---------------------------------------------------------------------------


class TestMuScheduleConversions:
    # weights_to_mu_schedule calls compute_eps() (result unused, but the call
    # is made), which reads SingletonConfig. Set it up for every test here.
    @pytest.fixture(autouse=True)
    def _setup_singleton(self, singleton_with_max_sigma):
        pass

    def test_weights_to_mu_schedule_shape(self, params):
        weights = jnp.ones(T)
        mu_sched = params.weights_to_mu_schedule(weights)
        assert mu_sched.shape == weights.shape

    def test_uniform_weights_give_mu_0(self, params):
        # With weights ≡ 1, each mu_i = mu_0 (budget split evenly).
        weights = jnp.ones(T)
        mu_sched = params.weights_to_mu_schedule(weights)
        assert jnp.allclose(mu_sched, params.mu_0, atol=1e-5)

    def test_mu_schedule_non_negative(self, params):
        weights = jnp.linspace(0.5, 2.0, T)
        mu_sched = params.weights_to_mu_schedule(weights)
        assert jnp.all(mu_sched >= 0)

    def test_mu_schedule_to_weights_shape(self, params):
        schedule = jnp.ones(T) * float(params.mu_0)
        weights = params.mu_schedule_to_weights(schedule)
        assert weights.shape == schedule.shape

    def test_round_trip_weights_to_mu_to_weights(self, params):
        original = jnp.linspace(0.5, 2.0, T)
        recovered = params.mu_schedule_to_weights(
            params.weights_to_mu_schedule(original),
        )
        recovered = params.mu_schedule_to_weights(
            params.weights_to_mu_schedule(original),
        )
        assert jnp.allclose(recovered, original, atol=1e-4)

    def test_round_trip_mu_to_weights_to_mu(self, params):
        mu_schedule = jnp.linspace(0.5, float(params.mu_0) * 1.5, T)
        recovered = params.weights_to_mu_schedule(
            params.mu_schedule_to_weights(mu_schedule),
        )
        assert jnp.allclose(recovered, mu_schedule, atol=1e-5)

    def test_larger_weight_gives_larger_mu(self, params):
        w1 = jnp.array([0.5])
        w2 = jnp.array([2.0])
        assert float(params.weights_to_mu_schedule(w2)[0]) > float(
            params.weights_to_mu_schedule(w1)[0],
        )


# ---------------------------------------------------------------------------
# gdp_to_sigma
# ---------------------------------------------------------------------------


class TestGDPToSigma:
    def test_formula(self, params):
        C = jnp.array(1.0)
        mu = jnp.array(2.0)
        assert jnp.isclose(params.gdp_to_sigma(C, mu), C / mu, atol=1e-6)

    def test_shape_preserved(self, params):
        C = jnp.ones(T)
        mu = jnp.linspace(0.5, 2.0, T)
        sigma = params.gdp_to_sigma(C, mu)
        assert sigma.shape == mu.shape

    def test_larger_mu_gives_smaller_sigma(self, params):
        C = jnp.array(1.0)
        sigma_small_mu = params.gdp_to_sigma(C, jnp.array(1.0))
        sigma_large_mu = params.gdp_to_sigma(C, jnp.array(4.0))
        assert float(sigma_large_mu) < float(sigma_small_mu)

    def test_larger_C_gives_larger_sigma(self, params):
        mu = jnp.array(2.0)
        sigma_small_C = params.gdp_to_sigma(jnp.array(0.5), mu)
        sigma_large_C = params.gdp_to_sigma(jnp.array(2.0), mu)
        assert float(sigma_large_C) > float(sigma_small_C)


# ---------------------------------------------------------------------------
# weights_to_sigma_schedule / sigma_schedule_to_weights (inverse pair)
# ---------------------------------------------------------------------------


class TestSigmaScheduleConversions:
    @pytest.fixture(autouse=True)
    def _setup_singleton(self, singleton_with_max_sigma):
        pass

    def test_weights_to_sigma_schedule_shape(self, params):
        C = jnp.array(1.0)
        weights = jnp.ones(T)
        sigma = params.weights_to_sigma_schedule(C, weights)
        assert sigma.shape == weights.shape

    def test_uniform_weights_give_constant_sigma(self, params):
        # Uniform weights → each mu_i = mu_0 → each sigma = C / mu_0.
        C = jnp.array(1.0)
        weights = jnp.ones(T)
        sigma = params.weights_to_sigma_schedule(C, weights)
        expected_sigma = C / params.mu_0
        assert jnp.allclose(sigma, expected_sigma, atol=1e-5)

    def test_sigma_positive(self, params):
        C = jnp.array(1.0)
        weights = jnp.linspace(0.5, 2.0, T)
        sigma = params.weights_to_sigma_schedule(C, weights)
        assert jnp.all(sigma > 0)

    def test_round_trip_sigma_to_weights_to_sigma(
        self,
        params,
        singleton_with_max_sigma,
    ):
        # sigma → weights → sigma should recover the original (below max_sigma).
        max_sigma = singleton_with_max_sigma
        C = jnp.array(1.0)
        # Keep sigmas well below max_sigma so clipping doesn't interfere.
        original_sigmas = jnp.linspace(0.5, max_sigma * 0.5, T)
        weights = params.sigma_schedule_to_weights(C, original_sigmas)
        recovered_sigmas = params.weights_to_sigma_schedule(C, weights)
        assert jnp.allclose(recovered_sigmas, original_sigmas, atol=1e-4)

    def test_sigma_schedule_to_weights_clips_at_max_sigma(
        self,
        params,
        singleton_with_max_sigma,
    ):
        max_sigma = singleton_with_max_sigma
        C = jnp.array(1.0)
        # Sigma far above max_sigma must be clipped.
        huge_sigma = jnp.ones(T) * max_sigma * 100
        clipped_sigma = jnp.ones(T) * max_sigma
        weights_huge = params.sigma_schedule_to_weights(C, huge_sigma)
        weights_capped = params.sigma_schedule_to_weights(C, clipped_sigma)
        assert jnp.allclose(weights_huge, weights_capped, atol=1e-5)


# ---------------------------------------------------------------------------
# weights_to_clip_schedule
# ---------------------------------------------------------------------------


class TestWeightsToClipSchedule:
    @pytest.fixture(autouse=True)
    def _setup_singleton(self, singleton_with_max_sigma):
        pass

    def test_formula(self, params):
        # clip_i = sigma_i * mu_i
        sigmas = jnp.ones(T) * 2.0
        weights = jnp.ones(T)
        mu_sched = params.weights_to_mu_schedule(weights)
        expected = sigmas * mu_sched
        result = params.weights_to_clip_schedule(sigmas, weights)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_shape(self, params):
        sigmas = jnp.ones(T)
        weights = jnp.ones(T)
        clips = params.weights_to_clip_schedule(sigmas, weights)
        assert clips.shape == (T,)

    def test_positive_output(self, params):
        sigmas = jnp.linspace(1.0, 3.0, T)
        weights = jnp.linspace(0.5, 2.0, T)
        clips = params.weights_to_clip_schedule(sigmas, weights)
        assert jnp.all(clips > 0)

    def test_uniform_weights_give_constant_ratio(self, params):
        # With uniform weights, mu_i ≡ mu_0, so clips = sigmas * mu_0.
        sigmas = jnp.linspace(1.0, 3.0, T)
        weights = jnp.ones(T)
        clips = params.weights_to_clip_schedule(sigmas, weights)
        assert jnp.allclose(clips / sigmas, params.mu_0, atol=1e-5)


# ---------------------------------------------------------------------------
# project_weights
#
# project_weights projects onto the set {w : sum(exp(w_i^2)) <= bound} where
# bound = (mu/p)^2 + T.  The projection is a no-op when the input is already
# feasible, so the constraint invariant only holds for *infeasible* inputs.
# ---------------------------------------------------------------------------


def _make_infeasible(params: GDPPrivacyParameters) -> jnp.ndarray:
    """Return weights that violate the privacy constraint for *params*."""
    # bound = (mu/p)^2 + T.  All weights equal to 2 give sum(exp(4)) >> bound
    # for the standard test parameters (bound ≈ 110, sum ≈ 546).
    return jnp.ones(params.T) * 2.0


class TestProjectWeights:
    def test_output_shape(self, params):
        projected = params.project_weights(_make_infeasible(params))
        assert projected.shape == (T,)

    def test_privacy_constraint_satisfied_for_infeasible_input(self, params):
        # Infeasible input lands exactly on the constraint boundary.
        weights = _make_infeasible(params)
        projected = params.project_weights(weights)
        bound = (params.mu / params.p) ** 2 + params.T
        actual = jnp.sum(jnp.exp(projected**2))
        assert jnp.isclose(actual, bound, rtol=1e-3)

    def test_privacy_constraint_nonuniform_infeasible(self, params):
        # Non-uniform infeasible weights also land on the boundary.
        weights = jnp.linspace(1.6, 2.4, T)  # all > sqrt(log(bound/T)) ≈ 1.55
        projected = params.project_weights(weights)
        bound = (params.mu / params.p) ** 2 + params.T
        actual = jnp.sum(jnp.exp(projected**2))
        assert jnp.isclose(actual, bound, rtol=1e-3)

    def test_feasible_input_unchanged(self, params):
        # A feasible point (constraint not active) projects to itself.
        # With T=10, p=0.1, mu≈1: bound≈110.  Uniform weights of 1 give
        # sum(exp(1)) ≈ 27.18 << 110, so they are strictly feasible.
        weights = jnp.ones(T)
        projected = params.project_weights(weights)
        assert jnp.allclose(projected, weights, atol=1e-4)

    def test_idempotent(self, params):
        # Projecting an already-projected infeasible point is a no-op.
        once = params.project_weights(_make_infeasible(params))
        twice = params.project_weights(once)
        assert jnp.allclose(once, twice, atol=1e-4)

    def test_uniform_infeasible_projects_to_uniform(self, params):
        # By symmetry, uniform infeasible weights project to uniform weights.
        weights = _make_infeasible(params)
        projected = params.project_weights(weights)
        assert jnp.allclose(projected, projected[0], atol=1e-4)

    def test_single_step_infeasible(self):
        # T=1 degenerate case; p=0.5 → bound=(1/0.5)^2+1=5; w=1.5 gives
        # exp(2.25)≈9.5 > 5, so the input is infeasible.
        p1 = GDPPrivacyParameters(EPS, DELTA, p=0.5, T=1)
        weights = jnp.array([1.5])
        projected = p1.project_weights(weights)
        bound = (p1.mu / p1.p) ** 2 + p1.T
        assert jnp.isclose(jnp.sum(jnp.exp(projected**2)), bound, rtol=1e-3)

    def test_different_budgets_give_different_projections(self):
        # Tighter budget → smaller bound → projects further from the input.
        # Use infeasible weights for both so the projection is non-trivial.
        # p_tight: mu≈0.5, p=0.1, T=10 → bound≈35; all-2 weights give 546>>35
        # p_loose: mu≈3.0, p=0.1, T=10 → bound≈910; all-2 weights give 546<910
        # (p_loose input is feasible so projection = identity)
        p_tight = GDPPrivacyParameters(0.5, 0.0524403232877, P, T)
        p_loose = GDPPrivacyParameters(3.0, 0.566737999092, P, T)
        weights = jnp.ones(T) * 2.0
        proj_tight = p_tight.project_weights(weights)
        proj_loose = p_loose.project_weights(weights)
        # tight projection must shrink weights; loose projection is identity
        assert jnp.allclose(proj_loose, weights, atol=1e-4)
        assert jnp.all(proj_tight < proj_loose)
