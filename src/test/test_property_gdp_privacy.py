"""Property-based tests for privacy/gdp_privacy.py.

Covers: approx_to_gdp, GDPPrivacyParameters (compute_mu_0, compute_expenditure,
weights_to_mu_schedule, weights_to_sigma_schedule, project_weights,
project_sigma_and_clip).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from privacy.gdp_privacy import GDPPrivacyParameters, approx_to_gdp

from ._shared import (
    _DELTA,
    _EPS,
    _P,
    _T_FIXED,
    _T_FIXED_LONG,
    _delta_st,
    _eps_st,
    _jax_settings,
    _p_st,
    _T_st,
    _weight_val_st,
)

# ---------------------------------------------------------------------------
# approx_to_gdp
# ---------------------------------------------------------------------------


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
        """Larger δ (weaker privacy) → larger μ."""
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


# ---------------------------------------------------------------------------
# GDPPrivacyParameters.compute_mu_0
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# GDPPrivacyParameters.compute_expenditure
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# weights_to_mu_schedule / mu_schedule_to_weights round-trip
# ---------------------------------------------------------------------------


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
        assert jnp.all(self.params.weights_to_mu_schedule(weights) >= 0)

    @given(w1=_weight_val_st, w2=_weight_val_st)
    @_jax_settings
    def test_monotone_in_weight(self, w1, w2):
        """A higher weight at position i yields a higher μ at position i."""
        assume(w2 > w1 + 1e-3)
        mu1 = float(self.params.weights_to_mu_schedule(jnp.array([w1]))[0])
        mu2 = float(self.params.weights_to_mu_schedule(jnp.array([w2]))[0])
        assert mu2 > mu1


# ---------------------------------------------------------------------------
# weights_to_sigma_schedule: σ × μ == C everywhere
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# project_weights
# ---------------------------------------------------------------------------

_PROJ_PARAMS = GDPPrivacyParameters(1.0, 0.126936737507, 0.1, _T_FIXED)
_PROJ_BOUND = float((_PROJ_PARAMS.mu / _PROJ_PARAMS.p) ** 2 + _PROJ_PARAMS.T)

_infeasible_w_st = st.lists(
    st.floats(min_value=1.6, max_value=3.0, allow_nan=False, allow_infinity=False),
    min_size=_T_FIXED,
    max_size=_T_FIXED,
)
_uniform_infeasible_w_st = st.floats(
    min_value=1.6, max_value=2.0, allow_nan=False, allow_infinity=False
).map(lambda v: [v] * _T_FIXED)
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
        before = w_np[:, None] - w_np[None, :]
        after = p_np[:, None] - p_np[None, :]
        assert np.all(after[before > 0] >= -1e-4)

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
        """Perturbing the projected point and re-projecting gives a farther point."""
        weights = jnp.array(w, dtype=jnp.float32)
        perturb = jnp.array(perturb, dtype=jnp.float32)
        projected = _PROJ_PARAMS.project_weights(weights)
        actual = float(jnp.sum(jnp.exp(projected**2)))
        assert actual == pytest.approx(_PROJ_BOUND, rel=1e-2)

        new_projected = projected + perturb
        bound = float(jnp.sum(jnp.exp(new_projected**2)))

        if bound > _PROJ_BOUND:
            assume(jnp.all(new_projected > 0))
            new_projected = _PROJ_PARAMS.project_weights(new_projected)
            actual = float(jnp.sum(jnp.exp(new_projected**2)))
            assert jnp.allclose(actual, _PROJ_BOUND, atol=1e-2)

        assert jnp.linalg.norm(weights - projected) <= jnp.linalg.norm(weights - new_projected)


# ---------------------------------------------------------------------------
# project_sigma_and_clip
# ---------------------------------------------------------------------------

_SC_PARAMS = GDPPrivacyParameters(1.0, 0.126936737507, 0.1, _T_FIXED)
_SC_BUDGET = float((_SC_PARAMS.mu / _SC_PARAMS.p) ** 2)

_sc_infeasible_st = st.fixed_dictionaries(
    {
        "sigma": st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
        "ratio": st.floats(min_value=1.7, max_value=2.5, allow_nan=False, allow_infinity=False),
    }
)
_sc_feasible_st = st.fixed_dictionaries(
    {
        "sigma": st.floats(min_value=2.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        "clip": st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
    }
)
_sc_large_vals = st.fixed_dictionaries(
    {
        "sigma": st.floats(min_value=2.0, max_value=20.0, allow_nan=False, allow_infinity=False),
        "clip": st.floats(min_value=0.1, max_value=20.0, allow_nan=False, allow_infinity=False),
    }
)
_sc_nonuniform_clips_st = st.lists(
    st.floats(min_value=2.0, max_value=2.5, allow_nan=False, allow_infinity=False),
    min_size=_T_FIXED,
    max_size=_T_FIXED,
)
_SC_NONUNIFORM_SIGMA = 1.0
_sc_small_perturb_st = st.lists(
    st.floats(min_value=-0.05, max_value=0.05, allow_nan=False, allow_infinity=False),
    min_size=_T_FIXED,
    max_size=_T_FIXED,
)


def _sc_eval(sigmas, clips):
    return float(jnp.sum(jnp.exp((clips / sigmas) ** 2) - 1.0))


class TestProjectSigmaAndClipProperties:
    """Geometric invariants of the Euclidean projection onto the sigma/clip constraint."""

    @given(d=_sc_infeasible_st)
    @_jax_settings
    def test_constraint_satisfied_after_projection(self, d):
        """Projecting infeasible (sigma, clip) places the result on the constraint boundary."""
        sigmas = jnp.ones(_T_FIXED) * d["sigma"]
        clips = jnp.ones(_T_FIXED) * d["sigma"] * d["ratio"]
        ps, pc = _SC_PARAMS.project_sigma_and_clip(sigmas, clips)
        assert _sc_eval(ps, pc) == pytest.approx(_SC_BUDGET, rel=1e-2)

    @given(d=_sc_feasible_st)
    @_jax_settings
    def test_feasible_input_unchanged(self, d):
        """A feasible (sigma, clip) point is returned unchanged."""
        sigmas = jnp.ones(_T_FIXED) * d["sigma"]
        clips = jnp.ones(_T_FIXED) * d["clip"]
        assume(_SC_PARAMS.compute_expenditure(sigmas, clips) < _SC_PARAMS.mu)
        ps, pc = _SC_PARAMS.project_sigma_and_clip(sigmas, clips)
        assert jnp.allclose(ps, sigmas, atol=1e-4)
        assert jnp.allclose(pc, clips, atol=1e-4)

    @given(d=_sc_infeasible_st)
    @_jax_settings
    def test_idempotent(self, d):
        """Projecting an already-projected result is a no-op."""
        sigmas = jnp.ones(_T_FIXED) * d["sigma"]
        clips = jnp.ones(_T_FIXED) * d["sigma"] * d["ratio"]
        ps1, pc1 = _SC_PARAMS.project_sigma_and_clip(sigmas, clips)
        ps2, pc2 = _SC_PARAMS.project_sigma_and_clip(ps1, pc1)
        assert jnp.allclose(ps1, ps2, atol=1e-3)
        assert jnp.allclose(pc1, pc2, atol=1e-3)

    @given(clips=_sc_nonuniform_clips_st)
    @_jax_settings
    def test_ordering_preserved(self, clips):
        """If clip[i] > clip[j] before projection (uniform sigma) then pc[i] >= pc[j] after."""
        sigmas = jnp.ones(_T_FIXED) * _SC_NONUNIFORM_SIGMA
        clips_arr = jnp.array(clips, dtype=jnp.float32)
        _, pc = _SC_PARAMS.project_sigma_and_clip(sigmas, clips_arr)
        clips_np = np.array(clips_arr)
        pc_np = np.array(pc)
        before = clips_np[:, None] - clips_np[None, :]
        after = pc_np[:, None] - pc_np[None, :]
        assert np.all(after[before > 0] >= -1e-4)

    @given(d=_sc_infeasible_st)
    @_jax_settings
    def test_projection_does_not_increase_expenditure(self, d):
        """Projection never increases the constraint value."""
        sigmas = jnp.ones(_T_FIXED) * d["sigma"]
        clips = jnp.ones(_T_FIXED) * d["sigma"] * d["ratio"]
        ps, pc = _SC_PARAMS.project_sigma_and_clip(sigmas, clips)
        assert _sc_eval(ps, pc) <= _sc_eval(sigmas, clips) + 1e-3

    @given(d=_sc_infeasible_st, perturb=_sc_small_perturb_st)
    @_jax_settings
    def test_projection_seems_closest(self, d, perturb):
        """Perturbing the projected point and re-projecting gives a point farther from the input."""
        sigmas = jnp.ones(_T_FIXED) * d["sigma"]
        clips = jnp.ones(_T_FIXED) * d["sigma"] * d["ratio"]
        ps, pc = _SC_PARAMS.project_sigma_and_clip(sigmas, clips)
        assert _sc_eval(ps, pc) == pytest.approx(_SC_BUDGET, rel=1e-2)

        perturb_arr = jnp.array(perturb, dtype=jnp.float32)
        ps_p = jnp.maximum(ps + perturb_arr, jnp.float32(1e-6))
        pc_p = jnp.maximum(pc + perturb_arr, jnp.float32(1e-6))

        if _sc_eval(ps_p, pc_p) > _SC_BUDGET:
            assume(jnp.all(ps_p > 0) and jnp.all(pc_p > 0))
            ps_p, pc_p = _SC_PARAMS.project_sigma_and_clip(ps_p, pc_p)

        dist_orig = float(jnp.linalg.norm(jnp.concatenate([sigmas - ps, clips - pc])))
        dist_perturb = float(jnp.linalg.norm(jnp.concatenate([sigmas - ps_p, clips - pc_p])))
        assert dist_orig <= dist_perturb + 1e-4

    @given(d=_sc_infeasible_st)
    @_jax_settings
    def test_projected_values_positive(self, d):
        """Projected sigmas and clips are always strictly positive."""
        sigmas = jnp.ones(_T_FIXED) * d["sigma"]
        clips = jnp.ones(_T_FIXED) * d["sigma"] * d["ratio"]
        ps, pc = _SC_PARAMS.project_sigma_and_clip(sigmas, clips)
        assert jnp.all(ps > 0)
        assert jnp.all(pc > 0)

    @given(d=_sc_large_vals)
    @_jax_settings
    def test_very_large_projection_is_finite(self, d):
        sigmas = jnp.ones(_T_FIXED_LONG) * d["sigma"]
        clips = jnp.ones(_T_FIXED_LONG) * d["clip"]
        params = GDPPrivacyParameters(8.0, _DELTA, _P, _T_FIXED_LONG)
        with jax.debug_nans(True), jax.debug_infs(True):
            ps, pc = params.project_sigma_and_clip(sigmas, clips)
        assert jnp.isfinite(ps).all() and jnp.isfinite(pc).all()

    @given(d=_sc_large_vals)
    @_jax_settings
    def test_very_large_projection_adheres(self, d):
        sigmas = jnp.ones(_T_FIXED_LONG) * d["sigma"]
        clips = jnp.ones(_T_FIXED_LONG) * d["clip"]
        params = GDPPrivacyParameters(8.0, _DELTA, _P, _T_FIXED_LONG)
        _mu = params.compute_expenditure(sigmas, clips)
        assume(_mu > params.mu)
        ps, pc = params.project_sigma_and_clip(sigmas, clips)
        _mu = params.compute_expenditure(ps, pc)
        assert jnp.isclose(_mu, params.mu, atol=1e-5)
