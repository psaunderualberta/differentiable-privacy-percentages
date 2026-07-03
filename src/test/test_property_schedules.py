"""Property-based tests for policy/schedules/.

Covers: DynamicDPSGDSchedule (positivity, shape, budget, monotonicity, clamping).
"""

import jax.numpy as jnp
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from policy.schedules.dynamic_dpsgd import DynamicDPSGDSchedule
from privacy.gdp_privacy import GDPPrivacyParameters

from ._shared import _T_FIXED, _jax_settings

# ---------------------------------------------------------------------------
# DynamicDPSGDSchedule
# ---------------------------------------------------------------------------

_DYN_PARAMS = GDPPrivacyParameters(1.0, 0.126936737507, 0.1, _T_FIXED)
_DYN_EPS = 0.01

_rho_st = st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False)
_c0_st = st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)


def _make_dynamic(rho_mu: float, rho_C: float, c_0: float) -> DynamicDPSGDSchedule:
    return DynamicDPSGDSchedule(rho_mu, rho_C, c_0, _DYN_PARAMS, _DYN_EPS)


class TestDynamicDPSGDScheduleProperties:
    """Universal invariants of DynamicDPSGDSchedule across arbitrary parameters."""

    @given(rho_mu=_rho_st, rho_C=_rho_st, c_0=_c0_st)
    @settings(max_examples=25, deadline=None)
    def test_mu_0_always_positive(self, rho_mu, rho_C, c_0):
        """mu_0 > 0 for any valid (rho_mu, rho_C, c_0)."""
        assert float(_make_dynamic(rho_mu, rho_C, c_0).mu_0) > 0

    @given(rho_mu=_rho_st, rho_C=_rho_st, c_0=_c0_st)
    @settings(max_examples=25, deadline=None)
    def test_mu_0_satisfies_privacy_budget(self, rho_mu, rho_C, c_0):
        """Solved mu_0 reconstructs the total GDP mu to within 0.1% relative error."""
        s = _make_dynamic(rho_mu, rho_C, c_0)
        pows = s.rho_mu ** (s.iters / s.iters.size)
        mu_reconstructed = jnp.sqrt(_DYN_PARAMS.p**2 * jnp.sum(jnp.exp((pows * s.mu_0) ** 2) - 1))
        assert jnp.isclose(mu_reconstructed, _DYN_PARAMS.mu, rtol=1e-3)

    @given(rho_mu=_rho_st, rho_C=_rho_st, c_0=_c0_st)
    @settings(max_examples=25, deadline=None)
    def test_sigma_and_clips_satisfies_privacy_budget(self, rho_mu, rho_C, c_0):
        """Sigma and clip schedules reconstruct the total GDP mu to within 0.1%."""
        s = _make_dynamic(rho_mu, rho_C, c_0)
        mu_reconstructed = _DYN_PARAMS.compute_expenditure(
            s.get_private_noise_scales(), s.get_private_clips()
        )
        assert jnp.isclose(mu_reconstructed, _DYN_PARAMS.mu, rtol=1e-3)

    @given(rho_mu=_rho_st, rho_C=_rho_st, c_0=_c0_st)
    @_jax_settings
    def test_sigmas_always_positive(self, rho_mu, rho_C, c_0):
        """get_private_noise_scales() > 0 for any valid parameters."""
        assert jnp.all(_make_dynamic(rho_mu, rho_C, c_0).get_private_noise_scales() > 0)

    @given(rho_mu=_rho_st, rho_C=_rho_st, c_0=_c0_st)
    @_jax_settings
    def test_sigmas_correct_shape(self, rho_mu, rho_C, c_0):
        """get_private_noise_scales() returns shape (T,) for any parameters."""
        assert _make_dynamic(rho_mu, rho_C, c_0).get_private_noise_scales().shape == (_T_FIXED,)

    @given(rho_mu=_rho_st, rho_C=_rho_st, c_0=_c0_st)
    @_jax_settings
    def test_sigmas_finite(self, rho_mu, rho_C, c_0):
        """All sigma values are finite."""
        assert jnp.all(jnp.isfinite(_make_dynamic(rho_mu, rho_C, c_0).get_private_noise_scales()))

    @given(rho_mu=_rho_st, rho_C=_rho_st, c_0=_c0_st)
    @_jax_settings
    def test_clips_always_positive(self, rho_mu, rho_C, c_0):
        """get_private_clips() > 0 for any valid parameters."""
        assert jnp.all(_make_dynamic(rho_mu, rho_C, c_0).get_private_clips() > 0)

    @given(rho_mu=_rho_st, rho_C=_rho_st, c_0=_c0_st)
    @_jax_settings
    def test_clips_correct_shape(self, rho_mu, rho_C, c_0):
        """get_private_clips() returns shape (T,) for any parameters."""
        assert _make_dynamic(rho_mu, rho_C, c_0).get_private_clips().shape == (_T_FIXED,)

    @given(rho_mu=_rho_st, rho_C=_rho_st, c_0=_c0_st)
    @_jax_settings
    def test_clips_finite(self, rho_mu, rho_C, c_0):
        """All clip values are finite."""
        assert jnp.all(jnp.isfinite(_make_dynamic(rho_mu, rho_C, c_0).get_private_clips()))

    @given(rho_mu=_rho_st, rho_C=_rho_st, c_0=_c0_st)
    @_jax_settings
    def test_clip_sigma_ratio_formula(self, rho_mu, rho_C, c_0):
        """clip_i / sigma_i = mu_0 * rho_mu^(i/T) for all i."""
        s = _make_dynamic(rho_mu, rho_C, c_0)
        ratio = s.get_private_clips() / s.get_private_noise_scales()
        expected = s.mu_0 * s.rho_mu ** (s.iters / _T_FIXED)
        assert jnp.allclose(ratio, expected, rtol=1e-4)

    @given(
        rho_mu=st.floats(min_value=1.01, max_value=3.0, allow_nan=False, allow_infinity=False),
        rho_C=_rho_st,
        c_0=_c0_st,
    )
    @_jax_settings
    def test_clip_sigma_ratio_increases_when_rho_mu_gt_1(self, rho_mu, rho_C, c_0):
        """rho_mu > 1 → clip/sigma ratio increases with step index."""
        s = _make_dynamic(rho_mu, rho_C, c_0)
        ratio = s.get_private_clips() / s.get_private_noise_scales()
        assert jnp.all(jnp.diff(ratio) > -1e-7)

    @given(
        rho_mu=st.floats(min_value=0.1, max_value=0.99, allow_nan=False, allow_infinity=False),
        rho_C=_rho_st,
        c_0=_c0_st,
    )
    @_jax_settings
    def test_clip_sigma_ratio_decreases_when_rho_mu_lt_1(self, rho_mu, rho_C, c_0):
        """rho_mu < 1 → clip/sigma ratio decreases with step index."""
        s = _make_dynamic(rho_mu, rho_C, c_0)
        ratio = s.get_private_clips() / s.get_private_noise_scales()
        assert jnp.all(jnp.diff(ratio) < 1e-7)

    @given(
        rho_mu=st.floats(min_value=0.1, max_value=0.99, allow_nan=False, allow_infinity=False),
        rho_C=st.floats(min_value=0.1, max_value=0.99, allow_nan=False, allow_infinity=False),
        c_0=_c0_st,
    )
    @_jax_settings
    def test_sigmas_increasing_when_rho_product_lt_1(self, rho_mu, rho_C, c_0):
        """rho_mu * rho_C < 1 → sigmas strictly increase over steps."""
        s = _make_dynamic(rho_mu, rho_C, c_0)
        assert jnp.all(jnp.diff(s.get_private_noise_scales()) > -1e-7)

    @given(
        rho_mu=st.floats(min_value=1.01, max_value=3.0, allow_nan=False, allow_infinity=False),
        rho_C=st.floats(min_value=1.01, max_value=3.0, allow_nan=False, allow_infinity=False),
        c_0=_c0_st,
    )
    @_jax_settings
    def test_sigmas_decreasing_when_rho_product_gt_1(self, rho_mu, rho_C, c_0):
        """rho_mu * rho_C > 1 → sigmas strictly decrease over steps."""
        s = _make_dynamic(rho_mu, rho_C, c_0)
        assert jnp.all(jnp.diff(s.get_private_noise_scales()) < 1e-7)

    @given(
        rho_mu=_rho_st,
        rho_C=_rho_st,
        c_0=st.floats(min_value=0.1, max_value=2.5, allow_nan=False, allow_infinity=False),
    )
    @_jax_settings
    def test_clips_scale_linearly_with_c_0(self, rho_mu, rho_C, c_0):
        """Doubling c_0 doubles all clip values (mu_0 is independent of c_0)."""
        s1 = _make_dynamic(rho_mu, rho_C, c_0)
        s2 = _make_dynamic(rho_mu, rho_C, c_0 * 2.0)
        assert jnp.allclose(s2.get_private_clips(), 2.0 * s1.get_private_clips(), rtol=1e-5)

    @given(
        rho_mu=st.floats(
            min_value=1e-6, max_value=_DYN_EPS - 1e-6, allow_nan=False, allow_infinity=False
        ),
        rho_C=_rho_st,
        c_0=_c0_st,
    )
    @_jax_settings
    def test_project_clamps_rho_mu_to_eps(self, rho_mu, rho_C, c_0):
        """rho_mu below eps is clamped to eps after project()."""
        assert float(_make_dynamic(rho_mu, rho_C, c_0).project().rho_mu) >= _DYN_EPS - 1e-8

    @given(
        rho_mu=_rho_st,
        rho_C=st.floats(
            min_value=1e-6, max_value=_DYN_EPS - 1e-6, allow_nan=False, allow_infinity=False
        ),
        c_0=_c0_st,
    )
    @_jax_settings
    def test_project_clamps_rho_C_to_eps(self, rho_mu, rho_C, c_0):
        """rho_C below eps is clamped to eps after project()."""
        assert float(_make_dynamic(rho_mu, rho_C, c_0).project().rho_C) >= _DYN_EPS - 1e-8

    @given(
        rho_mu=_rho_st,
        rho_C=_rho_st,
        c_0=st.floats(
            min_value=1e-6, max_value=_DYN_EPS - 1e-6, allow_nan=False, allow_infinity=False
        ),
    )
    @_jax_settings
    def test_project_clamps_C_0_to_eps(self, rho_mu, rho_C, c_0):
        """c_0 below eps is clamped to eps after project()."""
        assert float(_make_dynamic(rho_mu, rho_C, c_0).project().C_0) >= _DYN_EPS - 1e-8

    @given(rho_mu=_rho_st, rho_C=_rho_st, c_0=_c0_st)
    @_jax_settings
    def test_project_idempotent(self, rho_mu, rho_C, c_0):
        """project(project(s)) has the same parameters as project(s)."""
        s = _make_dynamic(rho_mu, rho_C, c_0)
        once = s.project()
        twice = once.project()
        assert float(twice.rho_mu) == pytest.approx(float(once.rho_mu))
        assert float(twice.rho_C) == pytest.approx(float(once.rho_C))
        assert float(twice.C_0) == pytest.approx(float(once.C_0))

    @given(rho_mu=_rho_st, rho_C=_rho_st, c_0=_c0_st)
    @_jax_settings
    def test_project_sigmas_positive(self, rho_mu, rho_C, c_0):
        """Projected schedule always has positive sigmas."""
        assert jnp.all(_make_dynamic(rho_mu, rho_C, c_0).project().get_private_noise_scales() > 0)

    @given(rho_mu=_rho_st, rho_C=_rho_st, c_0=_c0_st)
    @_jax_settings
    def test_project_clips_positive(self, rho_mu, rho_C, c_0):
        """Projected schedule always has positive clips."""
        assert jnp.all(_make_dynamic(rho_mu, rho_C, c_0).project().get_private_clips() > 0)

    @given(
        rho_mu=st.floats(
            min_value=1e-6, max_value=_DYN_EPS * 0.5, allow_nan=False, allow_infinity=False
        ),
        rho_C=_rho_st,
        c_0=_c0_st,
    )
    @settings(max_examples=15, deadline=None)
    def test_project_mu_0_satisfies_budget_after_clamp(self, rho_mu, rho_C, c_0):
        """After clamping rho_mu, the re-solved mu_0 still satisfies the privacy budget."""
        s = _make_dynamic(rho_mu, rho_C, c_0)
        projected = s.project()
        pows = projected.rho_mu ** (projected.iters / projected.iters.size)
        mu_reconstructed = jnp.sqrt(
            _DYN_PARAMS.p**2 * jnp.sum(jnp.exp((pows * projected.mu_0) ** 2) - 1)
        )
        assert jnp.isclose(mu_reconstructed, _DYN_PARAMS.mu, rtol=1e-3)
