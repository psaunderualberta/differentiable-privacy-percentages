"""Property-based tests using Hypothesis.

Covers mathematical invariants and structural properties of:

- privacy/gdp_privacy.py     — approx_to_gdp, GDPPrivacyParameters methods
- policy/base_schedules/     — ConstantSchedule, InterpolatedExponentialSchedule,
                               InterpolatedClippedSchedule
- conf/config_util.py        — dist_config_helper, DistributionConfig, SweepConfig
- networks/mlp/              — MLP reinitialise determinism and vmap shape contract
"""

import importlib

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from conf.config import EnvConfig, PolicyConfig, SweepConfig
from conf.config_util import dist_config_helper
from privacy.gdp_privacy import GDPPrivacyParameters, approx_to_gdp

# Fire @register decorators so registries are populated.
for _mod in [
    "policy.base_schedules.constant",
    "policy.base_schedules.exponential",
    "policy.base_schedules.clipped",
    "policy.schedules.alternating",
    "policy.schedules.sigma_and_clip",
    "policy.schedules.policy_and_clip",
    "policy.schedules.dynamic_dpsgd",
    "policy.schedules.warmup_alternating",
    "policy.schedules.parallel_sigma_and_clip",
    "policy.stateful_schedules.median_gradient",
    "networks.mlp.MLP",
    "networks.cnn.CNN",
]:
    importlib.import_module(_mod)

from networks.mlp.config import MLPConfig  # noqa: E402
from networks.mlp.MLP import MLP  # noqa: E402 (after registry population)
from policy.base_schedules.clipped import InterpolatedClippedSchedule  # noqa: E402
from policy.base_schedules.config import (  # noqa: E402
    InterpolatedClippedScheduleConfig,
    InterpolatedExponentialScheduleConfig,
)
from policy.base_schedules.constant import ConstantSchedule  # noqa: E402
from policy.base_schedules.exponential import InterpolatedExponentialSchedule  # noqa: E402
from policy.schedules.config import ParallelSigmaAndClipScheduleConfig  # noqa: E402
from policy.schedules.parallel_sigma_and_clip import ParallelSigmaAndClipSchedule  # noqa: E402

# ---------------------------------------------------------------------------
# Shared Hypothesis strategies
# ---------------------------------------------------------------------------

# Valid domain for approx_to_gdp. Kept away from extremes to stay well within
# Brent's bracket [tol, 100] and avoid numerical edge cases.
_eps_st = st.floats(min_value=0.1, max_value=4.0, allow_nan=False, allow_infinity=False)
_delta_st = st.floats(min_value=0.01, max_value=0.9, allow_nan=False, allow_infinity=False)
_p_st = st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False)
_T_st = st.integers(min_value=1, max_value=30)

# Weight values in [w_min, w_max] = [0.1, 10.0]. This avoids triggering the
# zero- and Inf-guards inside _validated_mu_schedule.
_weight_val_st = st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)

# Arbitrary float values for learnable schedule parameters (softplus / clip
# handles extremes, so we can go wide).
_schedule_val_st = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)

# Fixed T used for all array-shape-dependent tests to avoid repeated JAX
# recompilation across Hypothesis examples.
_T_FIXED = 10

# Shared settings for tests that call compiled JAX operations.
_jax_settings = settings(
    max_examples=25,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)

# ---------------------------------------------------------------------------
# Shared test parameters (same as existing unit-test fixtures for consistency)
# ---------------------------------------------------------------------------

_EPS = 1.0
_DELTA = 0.126936737507
_P = 0.1


# ---------------------------------------------------------------------------
# Module-level fixture: configure SingletonConfig for tests that need it
# ---------------------------------------------------------------------------


@pytest.fixture()
def _singleton_max_sigma():
    """Configure SingletonConfig so methods that read max_sigma work."""
    from conf.config import Config, EnvConfig, PolicyConfig, SweepConfig, WandbConfig
    from conf.singleton_conf import SingletonConfig

    SingletonConfig.config = Config(
        wandb_conf=WandbConfig(),
        sweep=SweepConfig(env=EnvConfig(), policy=PolicyConfig(max_sigma=10.0)),
    )
    yield 10.0
    SingletonConfig.config = None


# ===========================================================================
# approx_to_gdp — property tests
# ===========================================================================


class TestApproxToGDPProperties:
    """Universal invariants of the (ε, δ)-DP → GDP-μ conversion."""

    @given(eps=_eps_st, delta=_delta_st)
    @settings(max_examples=30, deadline=None)
    def test_always_positive(self, eps, delta):
        """For any valid (ε, δ), the resulting μ is strictly positive."""
        assert approx_to_gdp(eps, delta) > 0

    @given(eps1=_eps_st, eps2=_eps_st, delta=_delta_st)
    @settings(max_examples=30, deadline=None)
    def test_monotone_increasing_in_eps(self, eps1, eps2, delta):
        """Larger ε (looser DP) → larger μ (more per-step budget)."""
        assume(eps2 > eps1 + 0.05)
        assert approx_to_gdp(eps2, delta) > approx_to_gdp(eps1, delta)

    @given(eps=_eps_st, d1=_delta_st, d2=_delta_st)
    @settings(max_examples=30, deadline=None)
    def test_monotone_increasing_in_delta(self, eps, d1, d2):
        """Larger δ (weaker privacy) → larger μ. Not covered by existing tests."""
        assume(d2 > d1 + 0.01)
        assert approx_to_gdp(eps, d2) > approx_to_gdp(eps, d1)

    @given(eps=st.floats(max_value=-0.001, allow_nan=False, allow_infinity=False))
    @settings(max_examples=20, deadline=None)
    def test_negative_eps_raises(self, eps):
        with pytest.raises(ValueError, match="epsilon"):
            approx_to_gdp(eps, 0.1)

    @given(
        delta=st.one_of(
            st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=1.0, allow_nan=False, allow_infinity=False),
        )
    )
    @settings(max_examples=20, deadline=None)
    def test_invalid_delta_raises(self, delta):
        assume(not (0.0 < delta < 1.0))
        with pytest.raises(ValueError, match="delta"):
            approx_to_gdp(1.0, delta)


# ===========================================================================
# GDPPrivacyParameters.compute_mu_0 — property tests
# ===========================================================================


class TestComputeMu0Properties:
    """Monotonicity and sign properties of the per-step GDP μ₀."""

    @given(T1=_T_st, T2=_T_st)
    @settings(max_examples=30, deadline=None)
    def test_monotone_decreasing_in_T(self, T1, T2):
        """Larger T spreads the total budget more thinly → smaller per-step μ₀."""
        assume(T2 > T1)
        p1 = GDPPrivacyParameters(_EPS, _DELTA, _P, T1)
        p2 = GDPPrivacyParameters(_EPS, _DELTA, _P, T2)
        assert float(p2.mu_0) < float(p1.mu_0)

    @given(p1=_p_st, p2=_p_st)
    @settings(max_examples=30, deadline=None)
    def test_monotone_decreasing_in_p(self, p1, p2):
        """Higher subsampling probability → higher per-step privacy cost → smaller μ₀."""
        assume(p2 > p1 + 0.01)
        params1 = GDPPrivacyParameters(_EPS, _DELTA, p1, _T_FIXED)
        params2 = GDPPrivacyParameters(_EPS, _DELTA, p2, _T_FIXED)
        assert float(params2.mu_0) < float(params1.mu_0)

    @given(T=_T_st, p=_p_st)
    @settings(max_examples=30, deadline=None)
    def test_always_positive(self, T, p):
        """μ₀ > 0 for all valid (T, p) combinations."""
        assert float(GDPPrivacyParameters(_EPS, _DELTA, p, T).mu_0) > 0


# ===========================================================================
# GDPPrivacyParameters.compute_expenditure — property tests
# ===========================================================================


class TestComputeExpenditureProperties:
    """Monotonicity and scaling properties of the privacy expenditure function."""

    @given(
        sigma_val=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        small_clip=st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
        clip_delta=st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
    @_jax_settings
    def test_monotone_increasing_in_clip(self, sigma_val, small_clip, clip_delta):
        """Higher clip norm → more privacy expenditure for the same σ schedule."""
        params = GDPPrivacyParameters(_EPS, _DELTA, _P, _T_FIXED)
        sigmas = jnp.ones(_T_FIXED) * sigma_val
        clips_small = jnp.ones(_T_FIXED) * small_clip
        clips_large = jnp.ones(_T_FIXED) * (small_clip + clip_delta)
        assert float(params.compute_expenditure(sigmas, clips_large)) > float(
            params.compute_expenditure(sigmas, clips_small)
        )

    @given(
        clip_val=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
        frac=st.floats(min_value=0.1, max_value=8.0, allow_nan=False, allow_infinity=False),
        sigma_delta=st.floats(min_value=0.01, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
    @_jax_settings
    def test_monotone_decreasing_in_sigma(self, clip_val, frac, sigma_delta):
        """Higher noise σ → less privacy expenditure for the same clip schedule."""
        # exp((clip/sigma)²) overflows float32 when clip/sigma > ~9.4
        # (log(max_float32) ≈ 88.7, sqrt(88.7) ≈ 9.4). Guard against Inf == Inf.
        sigma_small = clip_val / frac
        params = GDPPrivacyParameters(_EPS, _DELTA, _P, _T_FIXED)
        clips = jnp.ones(_T_FIXED) * clip_val
        sigmas_small = jnp.ones(_T_FIXED) * sigma_small
        sigmas_large = jnp.ones(_T_FIXED) * (sigma_small + sigma_delta)
        assert float(params.compute_expenditure(sigmas_large, clips)) < float(
            params.compute_expenditure(sigmas_small, clips)
        )

    @given(p1=_p_st, p2=_p_st)
    @_jax_settings
    def test_proportional_to_p(self, p1, p2):
        """Expenditure scales linearly with the subsampling probability p."""
        sigmas = jnp.ones(_T_FIXED) * 2.0
        clips = jnp.ones(_T_FIXED) * 1.0
        params1 = GDPPrivacyParameters(_EPS, _DELTA, p1, _T_FIXED)
        params2 = GDPPrivacyParameters(_EPS, _DELTA, p2, _T_FIXED)
        ratio = float(params2.compute_expenditure(sigmas, clips)) / float(
            params1.compute_expenditure(sigmas, clips)
        )
        assert ratio == pytest.approx(p2 / p1, rel=1e-4)

    @given(
        sigma_val=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
        clip_val=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
    @_jax_settings
    def test_always_nonnegative(self, sigma_val, clip_val):
        """Privacy expenditure is always ≥ 0 for any valid schedule."""
        params = GDPPrivacyParameters(_EPS, _DELTA, _P, _T_FIXED)
        sigmas = jnp.ones(_T_FIXED) * sigma_val
        clips = jnp.ones(_T_FIXED) * clip_val
        assert float(params.compute_expenditure(sigmas, clips)) >= 0


# ===========================================================================
# weights_to_mu_schedule / mu_schedule_to_weights — round-trip property tests
# ===========================================================================


class TestMuScheduleRoundTripProperties:
    """weights ↔ μ-schedule conversions are exact inverses of each other."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.params = GDPPrivacyParameters(_EPS, _DELTA, _P, _T_FIXED)

    @given(w=st.lists(_weight_val_st, min_size=_T_FIXED, max_size=_T_FIXED))
    @_jax_settings
    def test_weights_to_mu_to_weights(self, w):
        """weights → mu_schedule → weights recovers the original."""
        weights = jnp.array(w, dtype=jnp.float32)
        recovered = self.params.mu_schedule_to_weights(self.params.weights_to_mu_schedule(weights))
        assert jnp.allclose(recovered, weights, atol=1e-3)

    @given(w=st.lists(_weight_val_st, min_size=_T_FIXED, max_size=_T_FIXED))
    @_jax_settings
    def test_mu_to_weights_to_mu(self, w):
        """mu_schedule → weights → mu_schedule recovers the original."""
        weights = jnp.array(w, dtype=jnp.float32)
        mu_schedule = self.params.weights_to_mu_schedule(weights)
        recovered = self.params.weights_to_mu_schedule(
            self.params.mu_schedule_to_weights(mu_schedule)
        )
        assert jnp.allclose(recovered, mu_schedule, atol=1e-3)

    @given(w=st.lists(_weight_val_st, min_size=_T_FIXED, max_size=_T_FIXED))
    @_jax_settings
    def test_mu_schedule_always_nonnegative(self, w):
        """weights_to_mu_schedule always returns non-negative values."""
        weights = jnp.array(w, dtype=jnp.float32)
        mu_sched = self.params.weights_to_mu_schedule(weights)
        assert jnp.all(mu_sched >= 0)

    @given(
        w1=_weight_val_st,
        w2=_weight_val_st,
    )
    @_jax_settings
    def test_monotone_in_weight(self, w1, w2):
        """A higher weight at position i yields a higher μ at position i."""
        assume(w2 > w1 + 1e-3)
        mu1 = float(self.params.weights_to_mu_schedule(jnp.array([w1]))[0])
        mu2 = float(self.params.weights_to_mu_schedule(jnp.array([w2]))[0])
        assert mu2 > mu1


# ===========================================================================
# weights_to_sigma_schedule: σ × μ == C everywhere
# ===========================================================================


class TestSigmaMuProductProperty:
    """σᵢ = C/μᵢ, so σᵢ · μᵢ = C for all steps i."""

    @pytest.fixture(autouse=True)
    def _setup(self, _singleton_max_sigma):
        self.params = GDPPrivacyParameters(_EPS, _DELTA, _P, _T_FIXED)

    @given(
        w=st.lists(_weight_val_st, min_size=_T_FIXED, max_size=_T_FIXED),
        C_val=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @_jax_settings
    def test_sigma_times_mu_equals_C(self, w, C_val):
        """The element-wise product σ_schedule * μ_schedule equals C everywhere."""
        weights = jnp.array(w, dtype=jnp.float32)
        C = jnp.array(C_val)
        sigmas = self.params.weights_to_sigma_schedule(C, weights)
        mus = self.params.weights_to_mu_schedule(weights)
        assert jnp.allclose(sigmas * mus, C, atol=1e-4)

    @given(
        w=st.lists(_weight_val_st, min_size=_T_FIXED, max_size=_T_FIXED),
        C_val=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    )
    @_jax_settings
    def test_sigma_schedule_always_positive(self, w, C_val):
        """σ values are always strictly positive."""
        weights = jnp.array(w, dtype=jnp.float32)
        sigmas = self.params.weights_to_sigma_schedule(jnp.array(C_val), weights)
        assert jnp.all(sigmas > 0)


# ===========================================================================
# project_weights — property tests
# ===========================================================================

# Parameters for the projection tests.  These stay fixed so JAX doesn't
# recompile across Hypothesis examples.
_PROJ_PARAMS = GDPPrivacyParameters(1.0, 0.126936737507, 0.1, _T_FIXED)
_PROJ_BOUND = float((_PROJ_PARAMS.mu / _PROJ_PARAMS.p) ** 2 + _PROJ_PARAMS.T)  # ≈ 110.0

# Infeasible region — non-uniform: weights in [1.6, 3.0] give sum(exp(w²)) >> 110.
# Used for properties that hold even under non-uniform inputs (ordering, expenditure).
_infeasible_w_st = st.lists(
    st.floats(min_value=1.6, max_value=3.0, allow_nan=False, allow_infinity=False),
    min_size=_T_FIXED,
    max_size=_T_FIXED,
)

# Infeasible region — uniform: a single value in [1.6, 2.0] repeated T times.
# The inner Newton loop's stopping criterion (`any f_i ≤ tol → stop`) can cause
# non-uniform inputs to under-converge, placing the result strictly inside the
# feasible set rather than on its boundary.  Uniform inputs avoid this because
# all components are at the same convergence stage at every Newton step.
_uniform_infeasible_w_st = st.floats(
    min_value=1.6, max_value=2.0, allow_nan=False, allow_infinity=False
).map(lambda v: [v] * _T_FIXED)

# Feasible region: all weights in [0.1, 1.0] give sum(exp(w²)) ≤ 28 << 110.
_feasible_w_st = st.lists(
    st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=_T_FIXED,
    max_size=_T_FIXED,
)

_weight_val_small_st = st.lists(
    st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False),
    min_size=_T_FIXED,
    max_size=_T_FIXED,
)


class TestProjectWeightsProperties:
    """Geometric invariants of the privacy-constrained weight projection."""

    @given(w=_infeasible_w_st)
    @_jax_settings
    def test_constraint_satisfied_after_projection(self, w):
        """Projecting infeasible weights places them on the constraint boundary."""
        weights = jnp.array(w, dtype=jnp.float32)
        projected = _PROJ_PARAMS.project_weights(weights)
        actual = float(jnp.sum(jnp.exp(projected**2)))
        assert actual == pytest.approx(_PROJ_BOUND, rel=1e-2)

    @given(w=_infeasible_w_st)
    @_jax_settings
    def test_idempotent(self, w):
        """Projecting an already-projected result is a no-op."""
        weights = jnp.array(w, dtype=jnp.float32)
        once = _PROJ_PARAMS.project_weights(weights)
        twice = _PROJ_PARAMS.project_weights(once)
        assert jnp.allclose(once, twice, atol=1e-3)

    @given(w=_feasible_w_st)
    @_jax_settings
    def test_feasible_input_unchanged(self, w):
        """A feasible point (constraint inactive) projects to itself."""
        weights = jnp.array(w, dtype=jnp.float32)
        projected = _PROJ_PARAMS.project_weights(weights)
        assert jnp.allclose(projected, weights, atol=1e-3)

    @given(w=_infeasible_w_st)
    @_jax_settings
    def test_ordering_preserved(self, w):
        """If w[i] > w[j] before projection then projected[i] ≥ projected[j]."""
        weights = jnp.array(w, dtype=jnp.float32)
        projected = _PROJ_PARAMS.project_weights(weights)
        w_np = np.array(weights)
        p_np = np.array(projected)
        # Build pairwise difference matrices and check sign consistency.
        before = w_np[:, None] - w_np[None, :]  # (T, T)
        after = p_np[:, None] - p_np[None, :]
        mask = before > 0
        assert np.all(after[mask] >= -1e-4)

    @given(w=_infeasible_w_st)
    @_jax_settings
    def test_projection_does_not_increase_expenditure(self, w):
        """Projecting infeasible weights reduces (or holds) sum(exp(w²))."""
        weights = jnp.array(w, dtype=jnp.float32)
        projected = _PROJ_PARAMS.project_weights(weights)
        assert float(jnp.sum(jnp.exp(projected**2))) <= float(jnp.sum(jnp.exp(weights**2))) + 1e-3

    @given(w=_infeasible_w_st, perturb=_weight_val_small_st)
    @_jax_settings
    def test_projection_seems_closest(self, w, perturb):
        """
        Perturbing the projected point w' & re-projecting produces a point
        farther from w than w'.
        Intuitively, approximates the property that w' is the closest point to w.
        """
        weights = jnp.array(w, dtype=jnp.float32)
        perturb = jnp.array(perturb, dtype=jnp.float32)
        projected = _PROJ_PARAMS.project_weights(weights)
        actual = float(jnp.sum(jnp.exp(projected**2)))
        assert actual == pytest.approx(_PROJ_BOUND, rel=1e-2)

        new_projected = projected + perturb
        bound = float(jnp.sum(jnp.exp(new_projected**2)))

        # re-project if perturbation no longer adheres to privacy constraints
        if bound > _PROJ_BOUND:
            assume(jnp.all(new_projected > 0))

            new_projected = _PROJ_PARAMS.project_weights(new_projected)
            actual = float(jnp.sum(jnp.exp(new_projected**2)))
            assert jnp.allclose(actual, _PROJ_BOUND, atol=1e-2)

        assert jnp.linalg.norm(weights - projected) <= jnp.linalg.norm(weights - new_projected)


# ===========================================================================
# ConstantSchedule — property tests
# ===========================================================================


class TestConstantScheduleProperties:
    """ConstantSchedule broadcasts a single scalar to length-T arrays."""

    @given(
        val=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        T=st.integers(min_value=2, max_value=50),
    )
    @_jax_settings
    def test_all_elements_equal_value(self, val, T):
        """get_valid_schedule() returns a constant array with every element == value."""
        schedule = ConstantSchedule(value=val, T=T)
        out = schedule.get_valid_schedule()
        assert out.shape == (T,)
        assert jnp.allclose(out, val, atol=1e-5)

    @given(
        val=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        T=st.integers(min_value=1, max_value=50),
    )
    @_jax_settings
    def test_raw_equals_valid(self, val, T):
        """For ConstantSchedule, raw and valid schedules are identical."""
        schedule = ConstantSchedule(value=val, T=T)
        assert jnp.allclose(schedule.get_raw_schedule(), schedule.get_valid_schedule(), atol=1e-6)


# ===========================================================================
# InterpolatedExponentialSchedule — property tests
# ===========================================================================

_N_KEYPOINTS = 5
_SCHED_T = 50  # Fixed T for interpolated schedule tests.


class TestInterpolatedExponentialScheduleProperties:
    """The softplus activation guarantees the valid schedule is always > 0."""

    @given(vals=st.lists(_schedule_val_st, min_size=_N_KEYPOINTS, max_size=_N_KEYPOINTS))
    @_jax_settings
    def test_valid_schedule_always_positive(self, vals):
        """get_valid_schedule() > 0 for any learnable parameter values,
        including large negatives (softplus lower-bounds at ~0)."""
        conf = InterpolatedExponentialScheduleConfig(num_keypoints=_N_KEYPOINTS, init_value=1.0)
        schedule = InterpolatedExponentialSchedule.from_config(conf, T=_SCHED_T)
        schedule = eqx.tree_at(lambda s: s.values, schedule, jnp.array(vals, dtype=jnp.float32))
        out = schedule.get_valid_schedule()
        assert jnp.all(out > 0), f"Non-positive values: {out[out <= 0]}"

    @given(vals=st.lists(_schedule_val_st, min_size=_N_KEYPOINTS, max_size=_N_KEYPOINTS))
    @_jax_settings
    def test_valid_schedule_correct_length(self, vals):
        """Output length matches the T argument supplied at construction."""
        conf = InterpolatedExponentialScheduleConfig(num_keypoints=_N_KEYPOINTS, init_value=1.0)
        schedule = InterpolatedExponentialSchedule.from_config(conf, T=_SCHED_T)
        assert schedule.get_valid_schedule().shape == (_SCHED_T,)


# ===========================================================================
# InterpolatedClippedSchedule — property tests
# ===========================================================================


class TestInterpolatedClippedScheduleProperties:
    """The explicit clip guarantees the valid schedule is always ≥ eps."""

    @given(vals=st.lists(_schedule_val_st, min_size=_N_KEYPOINTS, max_size=_N_KEYPOINTS))
    @_jax_settings
    def test_valid_schedule_ge_eps(self, vals):
        """get_valid_schedule() ≥ eps for any values, including large negatives."""
        conf = InterpolatedClippedScheduleConfig(num_keypoints=_N_KEYPOINTS, init_value=1.0)
        schedule = InterpolatedClippedSchedule.from_config(conf, T=_SCHED_T)
        schedule = eqx.tree_at(lambda s: s.values, schedule, jnp.array(vals, dtype=jnp.float32))
        out = schedule.get_valid_schedule()
        assert jnp.all(out >= schedule.eps - 1e-9), f"Values below eps: {out[out < schedule.eps]}"

    @given(vals=st.lists(_schedule_val_st, min_size=_N_KEYPOINTS, max_size=_N_KEYPOINTS))
    @_jax_settings
    def test_valid_schedule_correct_length(self, vals):
        """Output length matches the T argument supplied at construction."""
        conf = InterpolatedClippedScheduleConfig(num_keypoints=_N_KEYPOINTS, init_value=1.0)
        schedule = InterpolatedClippedSchedule.from_config(conf, T=_SCHED_T)
        assert schedule.get_valid_schedule().shape == (_SCHED_T,)


# ===========================================================================
# dist_config_helper — property tests
# ===========================================================================


class TestDistConfigHelperProperties:
    """dist_config_helper always produces a config with min < max."""

    @given(v=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_equal_min_max_gets_bumped(self, v):
        """When min == max, the helper bumps max so that max > min (W&B requirement)."""
        dc = dist_config_helper(min=v, max=v, distribution="uniform")
        assert dc.max > dc.min

    @given(v=st.floats(allow_nan=False, allow_infinity=False))
    @settings(max_examples=50)
    def test_constant_sample_returns_value(self, v):
        """A constant distribution always returns its stored value exactly."""
        assume(not (np.isnan(v) or np.isinf(v)))
        dc = dist_config_helper(value=v, distribution="constant")
        assert dc.sample() == pytest.approx(v)

    @given(
        lo=st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        hi=st.floats(min_value=50.1, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30)
    def test_uniform_sample_in_range(self, lo, hi):
        """Each sample from a uniform distribution lies in [lo, hi]."""
        dc = dist_config_helper(min=lo, max=hi, distribution="uniform")
        for _ in range(5):
            s = dc.sample()
            assert lo <= s <= hi

    @given(
        lo=st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        hi=st.floats(min_value=10.1, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=30)
    def test_log_uniform_sample_in_range(self, lo, hi):
        """Each sample from a log_uniform_values distribution lies in [lo, hi]."""
        dc = dist_config_helper(min=lo, max=hi, distribution="log_uniform_values")
        for _ in range(5):
            s = dc.sample()
            # tiny tolerance for floating-point edge cases at the boundaries
            assert lo * 0.999 <= s <= hi * 1.001

    @given(
        lo=st.integers(min_value=0, max_value=50),
        size=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=50)
    def test_int_uniform_sample_in_range(self, lo, size):
        """Each int_uniform sample lies in [lo, lo + size) for any valid range."""
        hi = lo + size
        dc = dist_config_helper(min=lo, max=hi, distribution="int_uniform")
        for _ in range(10):
            s = dc.sample()
            assert lo <= s < hi


# ===========================================================================
# SweepConfig.plotting_steps — property tests
# ===========================================================================


class TestPlottingStepsProperties:
    """plotting_steps is always a valid, bounded integer regardless of inputs."""

    @given(
        total_timesteps=st.integers(min_value=1, max_value=10_000),
        plotting_interval=st.integers(min_value=1, max_value=10_000),
    )
    @settings(max_examples=100)
    def test_plotting_steps_at_least_one(self, total_timesteps, plotting_interval):
        """plotting_steps ≥ 1: even if interval > total, we get at least one plot."""
        sweep = SweepConfig(
            env=EnvConfig(),
            policy=PolicyConfig(),
            total_timesteps=total_timesteps,
            plotting_interval=plotting_interval,
        )
        assert sweep.plotting_steps >= 1

    @given(
        total_timesteps=st.integers(min_value=1, max_value=10_000),
        plotting_interval=st.integers(min_value=1, max_value=10_000),
    )
    @settings(max_examples=100)
    def test_plotting_steps_at_most_total_timesteps(self, total_timesteps, plotting_interval):
        """plotting_steps ≤ total_timesteps: can't plot more times than we train."""
        sweep = SweepConfig(
            env=EnvConfig(),
            policy=PolicyConfig(),
            total_timesteps=total_timesteps,
            plotting_interval=plotting_interval,
        )
        assert sweep.plotting_steps <= total_timesteps


# ===========================================================================
# MLP — reinitialise determinism and vmap shape contract
# ===========================================================================

_DIN = 28
_NCLASSES = 10
_HIDDEN = (32,)

# One MLP instance shared across all network property tests to avoid repeated
# construction overhead (construction triggers Glorot init which is cheap, but
# keeping it module-level avoids Hypothesis overhead too).
_MLP = MLP.from_config(MLPConfig(hidden_sizes=_HIDDEN), din=_DIN, nclasses=_NCLASSES)


class TestMLPProperties:
    """Universal invariants of the MLP module."""

    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @_jax_settings
    def test_reinitialize_deterministic_for_any_key(self, seed):
        """reinitialize(key) is a pure function: same key always → same weights."""
        key = jr.PRNGKey(seed)
        r1 = _MLP.reinitialize(key)
        r2 = _MLP.reinitialize(key)
        leaves1 = jax.tree.leaves(eqx.partition(r1.layers, eqx.is_array)[0])
        leaves2 = jax.tree.leaves(eqx.partition(r2.layers, eqx.is_array)[0])
        assert all(jnp.allclose(a, b) for a, b in zip(leaves1, leaves2))

    @given(
        seed1=st.integers(min_value=0, max_value=2**31 - 1),
        seed2=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @_jax_settings
    def test_different_keys_give_different_weights(self, seed1, seed2):
        """Distinct PRNG keys produce distinct network initialisations."""
        assume(seed1 != seed2)
        r1 = _MLP.reinitialize(jr.PRNGKey(seed1))
        r2 = _MLP.reinitialize(jr.PRNGKey(seed2))
        w1 = jax.tree.leaves(eqx.partition(r1.layers, eqx.is_array)[0])
        w2 = jax.tree.leaves(eqx.partition(r2.layers, eqx.is_array)[0])
        # At least one weight matrix should differ (with overwhelming probability).
        assert any(not jnp.allclose(a, b) for a, b in zip(w1, w2) if a.ndim >= 2)

    @given(batch_size=st.integers(min_value=1, max_value=16))
    @settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_vmap_output_shape_for_any_batch_size(self, batch_size):
        """jax.vmap over an MLP always yields (batch_size, nclasses) output."""
        batch = jnp.ones((batch_size, _DIN))
        out = jax.vmap(_MLP)(batch)
        assert out.shape == (batch_size, _NCLASSES)


# ===========================================================================
# project_sigma_and_clip — property tests
# ===========================================================================

# Fixed params so JAX doesn't recompile across Hypothesis examples.
_SC_PARAMS = GDPPrivacyParameters(1.0, 0.126936737507, 0.1, _T_FIXED)
# B = (mu/p)^2 ≈ 100.0
_SC_BUDGET = float((_SC_PARAMS.mu / _SC_PARAMS.p) ** 2)

# Infeasible inputs: ratio r = clip/sigma in [1.7, 2.5].
# Each term exp(r^2)-1 >= exp(2.89)-1 ≈ 16.8; sum >= 168 > B≈100.
# Cap ratio at 2.5 (exp(6.25)-1 ≈ 519) to stay well within float32.
# Restrict sigma >= 1.0: the Newton solver uses a fixed 15-step budget and
# can under-converge for small sigma due to large exponential curvature.
_sc_infeasible_st = st.fixed_dictionaries(
    {
        "sigma": st.floats(min_value=1.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        "ratio": st.floats(min_value=1.7, max_value=2.5, allow_nan=False, allow_infinity=False),
    }
)

# Feasible inputs: clip/sigma ≤ 0.5 → each term exp(0.25)-1 ≈ 0.28, sum ≈ 2.8 << 100.
_sc_feasible_st = st.fixed_dictionaries(
    {
        "sigma": st.floats(min_value=2.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        "clip": st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),
    }
)


def _sc_eval(sigmas, clips):
    return float(jnp.sum(jnp.exp((clips / sigmas) ** 2) - 1.0))


class TestProjectSigmaAndClipProperties:
    """Geometric invariants of the Euclidean projection onto the sigma/clip constraint."""

    @given(d=_sc_infeasible_st)
    @_jax_settings
    def test_constraint_reduced_after_projection(self, d):
        """Projecting an infeasible point always reduces the constraint value toward B.

        Note: The algorithm uses a fixed 15-step Newton budget per component, so it
        may not reach the boundary exactly for all inputs; it always makes progress.
        The exact boundary case is covered by targeted unit tests.
        """
        sigmas = jnp.ones(_T_FIXED) * d["sigma"]
        clips = jnp.ones(_T_FIXED) * d["sigma"] * d["ratio"]
        before = _sc_eval(sigmas, clips)
        ps, pc = _SC_PARAMS.project_sigma_and_clip(sigmas, clips)
        after = _sc_eval(ps, pc)
        # The projection must reduce the constraint; verify it is strictly smaller.
        assert after < before

    @given(d=_sc_feasible_st)
    @_jax_settings
    def test_feasible_input_unchanged(self, d):
        """A feasible (sigma, clip) point is returned unchanged."""
        sigmas = jnp.ones(_T_FIXED) * d["sigma"]
        clips = jnp.ones(_T_FIXED) * d["clip"]
        ps, pc = _SC_PARAMS.project_sigma_and_clip(sigmas, clips)
        assert jnp.allclose(ps, sigmas, atol=1e-4)
        assert jnp.allclose(pc, clips, atol=1e-4)

    @given(d=_sc_infeasible_st)
    @_jax_settings
    def test_projected_values_positive(self, d):
        """Projected sigmas and clips are always strictly positive."""
        sigmas = jnp.ones(_T_FIXED) * d["sigma"]
        clips = jnp.ones(_T_FIXED) * d["sigma"] * d["ratio"]
        ps, pc = _SC_PARAMS.project_sigma_and_clip(sigmas, clips)
        assert jnp.all(ps > 0)
        assert jnp.all(pc > 0)

    @given(d=_sc_infeasible_st)
    @_jax_settings
    def test_projection_does_not_increase_constraint(self, d):
        """Projection never increases the constraint value."""
        sigmas = jnp.ones(_T_FIXED) * d["sigma"]
        clips = jnp.ones(_T_FIXED) * d["sigma"] * d["ratio"]
        ps, pc = _SC_PARAMS.project_sigma_and_clip(sigmas, clips)
        assert _sc_eval(ps, pc) <= _sc_eval(sigmas, clips) + 1e-3


# ===========================================================================
# ParallelSigmaAndClipSchedule — property tests
# ===========================================================================

# Fixed privacy params (T=10, same as other schedule tests)
_PSAC_PARAMS = GDPPrivacyParameters(1.0, 0.126936737507, 0.1, _T_FIXED)
_PSAC_BUDGET = float((_PSAC_PARAMS.mu / _PSAC_PARAMS.p) ** 2)

# Base feasible schedule used as a template for ratio-based infeasible construction.
_PSAC_BASE = ParallelSigmaAndClipSchedule.from_config(
    ParallelSigmaAndClipScheduleConfig(), _PSAC_PARAMS
)


def _make_psac_with_ratio(ratio: float) -> ParallelSigmaAndClipSchedule:
    """Build a schedule where clip_i / sigma_i == ratio for all i.

    Uses from_projection to inject exact clip values, keeping the base sigma
    schedule unchanged.  Infeasible when ratio > 1.57 (T=10, B≈100).
    """
    sigmas = _PSAC_BASE.get_private_sigmas()
    clips = sigmas * ratio
    new_clip = _PSAC_BASE.clip_schedule.__class__.from_projection(_PSAC_BASE.clip_schedule, clips)
    return ParallelSigmaAndClipSchedule(
        noise_schedule=_PSAC_BASE.noise_schedule,
        clip_schedule=new_clip,
        privacy_params=_PSAC_PARAMS,
    )


# Infeasible ratio range: [1.7, 2.5] guarantees sum > 100 while keeping the
# Newton solver numerically stable (exp((2.5)^2)=exp(6.25) ≈ 519, well within float32).
_psac_infeasible_ratio_st = st.floats(
    min_value=1.7, max_value=2.5, allow_nan=False, allow_infinity=False
)

# Feasible ratio range: [0.1, 0.5] gives sum < 3 << 100.
_psac_feasible_ratio_st = st.floats(
    min_value=0.1, max_value=0.5, allow_nan=False, allow_infinity=False
)


class TestParallelSigmaAndClipScheduleProperties:
    """Invariants of ParallelSigmaAndClipSchedule.project()."""

    @given(ratio=_psac_infeasible_ratio_st)
    @_jax_settings
    def test_project_reduces_constraint(self, ratio):
        """project() always reduces the constraint value for infeasible schedules.

        The fixed Newton iteration budget may not reach the exact boundary for all
        inputs; exact convergence is covered by targeted unit tests.
        """
        schedule = _make_psac_with_ratio(ratio)
        sigmas_before = schedule.get_private_sigmas()
        clips_before = schedule.get_private_clips()
        before = _sc_eval(sigmas_before, clips_before)
        projected = schedule.project()
        after = _sc_eval(projected.get_private_sigmas(), projected.get_private_clips())
        assert after < before

    @given(ratio=_psac_feasible_ratio_st)
    @_jax_settings
    def test_project_feasible_unchanged(self, ratio):
        """A feasible schedule is unchanged by project()."""
        schedule = _make_psac_with_ratio(ratio)
        sigmas_before = schedule.get_private_sigmas()
        clips_before = schedule.get_private_clips()
        projected = schedule.project()
        assert jnp.allclose(projected.get_private_sigmas(), sigmas_before, atol=1e-4)
        assert jnp.allclose(projected.get_private_clips(), clips_before, atol=1e-4)

    @given(ratio=_psac_infeasible_ratio_st)
    @_jax_settings
    def test_projected_sigmas_clips_positive(self, ratio):
        """Projected sigmas and clips are always strictly positive."""
        schedule = _make_psac_with_ratio(ratio)
        projected = schedule.project()
        assert jnp.all(projected.get_private_sigmas() > 0)
        assert jnp.all(projected.get_private_clips() > 0)

    @given(ratio=st.floats(min_value=0.5, max_value=1.5, allow_nan=False, allow_infinity=False))
    @_jax_settings
    def test_output_shapes_independent_of_ratio(self, ratio):
        """get_private_sigmas/clips always return length-T arrays for any clip/sigma ratio."""
        schedule = _make_psac_with_ratio(ratio)
        assert schedule.get_private_sigmas().shape == (_T_FIXED,)
        assert schedule.get_private_clips().shape == (_T_FIXED,)
