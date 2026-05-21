"""Property-based tests for policy/base_schedules/.

Covers: ConstantSchedule, InterpolatedExponentialSchedule,
InterpolatedClippedSchedule, BSplineSchedule.
"""

import equinox as eqx
import jax.numpy as jnp
from hypothesis import assume, given
from hypothesis import strategies as st

from policy.base_schedules.bspline import BSplineSchedule
from policy.base_schedules.clipped import InterpolatedClippedSchedule
from policy.base_schedules.config import (
    BSplineScheduleConfig,
    InterpolatedClippedScheduleConfig,
    InterpolatedExponentialScheduleConfig,
)
from policy.base_schedules.constant import ConstantSchedule
from policy.base_schedules.exponential import InterpolatedExponentialSchedule

from ._shared import _jax_settings, _schedule_val_st

# ---------------------------------------------------------------------------
# ConstantSchedule
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# InterpolatedExponentialSchedule
# ---------------------------------------------------------------------------

_N_KEYPOINTS = 5
_SCHED_T = 50


class TestInterpolatedExponentialScheduleProperties:
    """The softplus activation guarantees the valid schedule is always > 0."""

    @given(vals=st.lists(_schedule_val_st, min_size=_N_KEYPOINTS, max_size=_N_KEYPOINTS))
    @_jax_settings
    def test_valid_schedule_always_positive(self, vals):
        """get_valid_schedule() > 0 for any learnable parameter values."""
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


# ---------------------------------------------------------------------------
# InterpolatedClippedSchedule
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# BSplineSchedule
# ---------------------------------------------------------------------------

_BSPLINE_T = 50
_BSPLINE_N_CP = 8
_BSPLINE_DEGREE = 3

_bspline_cp_st = st.lists(_schedule_val_st, min_size=_BSPLINE_N_CP, max_size=_BSPLINE_N_CP)


def _make_bspline() -> BSplineSchedule:
    conf = BSplineScheduleConfig(
        num_control_points=_BSPLINE_N_CP,
        degree=_BSPLINE_DEGREE,
        init_value=1.0,
    )
    return BSplineSchedule.from_config(conf, T=_BSPLINE_T)


class TestBSplineScheduleProperties:
    """Structural and positivity invariants of BSplineSchedule."""

    @given(T=st.integers(min_value=_BSPLINE_N_CP, max_value=200))
    @_jax_settings
    def test_output_length_matches_T(self, T):
        """get_valid_schedule() and get_raw_schedule() both return length-T arrays."""
        conf = BSplineScheduleConfig(num_control_points=_BSPLINE_N_CP, degree=_BSPLINE_DEGREE)
        sched = BSplineSchedule.from_config(conf, T=T)
        assert sched.get_valid_schedule().shape == (T,)
        assert sched.get_raw_schedule().shape == (T,)

    @given(cp=_bspline_cp_st)
    @_jax_settings
    def test_valid_schedule_positive_softplus(self, cp):
        """softplus variant: get_valid_schedule() > 0 for any unconstrained control points."""
        sched = _make_bspline()
        sched = eqx.tree_at(lambda s: s.control_points, sched, jnp.array(cp, dtype=jnp.float32))
        out = sched.get_valid_schedule()
        assert jnp.all(out > 0), f"Non-positive values: {out[out <= 0]}"

    @given(
        cp=st.lists(
            st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
            min_size=_BSPLINE_N_CP,
            max_size=_BSPLINE_N_CP,
        )
    )
    @_jax_settings
    def test_valid_schedule_positive_exp(self, cp):
        """exp variant: get_valid_schedule() > 0 for any finite control points."""
        sched = _make_bspline()
        sched = eqx.tree_at(lambda s: s.control_points, sched, jnp.array(cp, dtype=jnp.float32))
        out = sched.get_valid_schedule()
        assert jnp.all(out > 0), f"Non-positive values: {out[out <= 0]}"

    @given(init=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False))
    @_jax_settings
    def test_init_value_is_uniform(self, init):
        """After from_config, get_valid_schedule() is uniformly equal to init_value."""
        conf = BSplineScheduleConfig(
            num_control_points=_BSPLINE_N_CP,
            degree=_BSPLINE_DEGREE,
            init_value=init,
        )
        sched = BSplineSchedule.from_config(conf, T=_BSPLINE_T)
        out = sched.get_valid_schedule()
        assert jnp.allclose(out, init, atol=1e-4), f"Expected uniform {init}, got {out}"

    @given(init=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False))
    @_jax_settings
    def test_init_value_is_uniform_exp(self, init):
        """Same check for the exp positivity variant."""
        conf = BSplineScheduleConfig(
            num_control_points=_BSPLINE_N_CP,
            degree=_BSPLINE_DEGREE,
            init_value=init,
        )
        sched = BSplineSchedule.from_config(conf, T=_BSPLINE_T)
        out = sched.get_valid_schedule()
        assert jnp.allclose(out, init, atol=1e-4), f"Expected uniform {init}, got {out}"

    @given(target=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False))
    @_jax_settings
    def test_from_projection_uniform_target(self, target):
        """from_projection on a uniform target recovers a nearly-uniform schedule."""
        sched = _make_bspline()
        projection = jnp.ones(_BSPLINE_T) * target
        sched2 = BSplineSchedule.from_projection(sched, projection)
        out = sched2.get_valid_schedule()
        assert jnp.allclose(out, target, atol=1e-3), (
            f"Expected ~{target}, got range [{out.min():.4f}, {out.max():.4f}]"
        )

    @given(target=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False))
    @_jax_settings
    def test_from_projection_uniform_target_exp(self, target):
        """Same round-trip check for the exp variant."""
        sched = _make_bspline()
        projection = jnp.ones(_BSPLINE_T) * target
        sched2 = BSplineSchedule.from_projection(sched, projection)
        out = sched2.get_valid_schedule()
        assert jnp.allclose(out, target, atol=1e-3), (
            f"Expected ~{target}, got range [{out.min():.4f}, {out.max():.4f}]"
        )

    @given(target=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False))
    @_jax_settings
    def test_from_projection_output_positive(self, target):
        """from_projection always produces a positive schedule regardless of target."""
        sched = _make_bspline()
        projection = jnp.ones(_BSPLINE_T) * target
        sched2 = BSplineSchedule.from_projection(sched, projection)
        assert jnp.all(sched2.get_valid_schedule() > 0)

    @given(
        n_cp=st.integers(min_value=4, max_value=15),
        degree=st.integers(min_value=1, max_value=3),
    )
    @_jax_settings
    def test_valid_schedule_positive_various_configs(self, n_cp, degree):
        """Valid schedule is positive for any valid (num_control_points, degree) combo."""
        assume(n_cp >= degree + 1)
        conf = BSplineScheduleConfig(
            num_control_points=n_cp,
            degree=degree,
            init_value=1.0,
        )
        sched = BSplineSchedule.from_config(conf, T=_BSPLINE_T)
        assert jnp.all(sched.get_valid_schedule() > 0)
