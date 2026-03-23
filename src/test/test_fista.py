"""Unit and property-based tests for FISTA acceleration in
ParallelSigmaAndClipSchedule and WarmupParallelSigmaAndClipSchedule."""

import importlib

import jax
import jax.numpy as jnp
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from policy.base_schedules.constant import ConstantSchedule
from policy.base_schedules.exponential import InterpolatedExponentialSchedule
from policy.schedules.config import (
    ParallelSigmaAndClipScheduleConfig,
    WarmupParallelSigmaAndClipScheduleConfig,
)
from policy.schedules.parallel_sigma_and_clip import ParallelSigmaAndClipSchedule
from policy.schedules.warmup_parallel_sigma_and_clip import WarmupParallelSigmaAndClipSchedule
from privacy.gdp_privacy import GDPPrivacyParameters

# Fire @register decorators before importing schedule classes.
for _mod in [
    "policy.base_schedules.constant",
    "policy.base_schedules.exponential",
    "policy.schedules.parallel_sigma_and_clip",
    "policy.schedules.warmup_parallel_sigma_and_clip",
]:
    importlib.import_module(_mod)


# ── Constants ─────────────────────────────────────────────────────────────────

EPS = 1.0
DELTA = 0.126936737507
P = 0.1
T = 10
WARMUP_STEPS = 5


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def params() -> GDPPrivacyParameters:
    return GDPPrivacyParameters(EPS, DELTA, P, T)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_parallel(
    params: GDPPrivacyParameters,
    *,
    use_fista: bool = True,
    noise_val: float = 1.0,
    clip_val: float = 0.5,
    fista_t: float = 1.0,
    x_curr_sig: jnp.ndarray | None = None,
    x_curr_clip: jnp.ndarray | None = None,
    x_prev_sig: jnp.ndarray | None = None,
    x_prev_clip: jnp.ndarray | None = None,
) -> ParallelSigmaAndClipSchedule:
    ones = jnp.ones(params.T)
    return ParallelSigmaAndClipSchedule(
        noise_schedule=ConstantSchedule(noise_val, params.T),
        clip_schedule=ConstantSchedule(clip_val, params.T),
        privacy_params=params,
        use_fista=use_fista,
        _x_curr_sigmas=ones if x_curr_sig is None else x_curr_sig,
        _x_curr_clips=ones if x_curr_clip is None else x_curr_clip,
        _x_prev_sigmas=ones if x_prev_sig is None else x_prev_sig,
        _x_prev_clips=ones if x_prev_clip is None else x_prev_clip,
        _fista_t=jnp.asarray(fista_t),
    )


def _make_warmup(
    params: GDPPrivacyParameters,
    *,
    use_fista: bool = True,
    warmup_steps: int = WARMUP_STEPS,
    step_count: int = 0,
    fista_t: float = 1.0,
    warmup_noise: float = 1.0,
    warmup_clip: float = 0.5,
    x_curr_sig: jnp.ndarray | None = None,
    x_curr_clip: jnp.ndarray | None = None,
    x_prev_sig: jnp.ndarray | None = None,
    x_prev_clip: jnp.ndarray | None = None,
) -> WarmupParallelSigmaAndClipSchedule:
    ones = jnp.ones(params.T)
    keypoints = jnp.linspace(0, params.T, 5, dtype=jnp.int32)
    values = jnp.zeros(5, dtype=jnp.float32)
    return WarmupParallelSigmaAndClipSchedule(
        noise_warmup=ConstantSchedule(warmup_noise, params.T),
        clip_warmup=ConstantSchedule(warmup_clip, params.T),
        noise_tail=InterpolatedExponentialSchedule(keypoints, values, params.T),
        clip_tail=InterpolatedExponentialSchedule(keypoints, values, params.T),
        privacy_params=params,
        step_count=step_count,
        warmup_steps=warmup_steps,
        use_fista=use_fista,
        _x_curr_sigmas=ones if x_curr_sig is None else x_curr_sig,
        _x_curr_clips=ones if x_curr_clip is None else x_curr_clip,
        _x_prev_sigmas=ones if x_prev_sig is None else x_prev_sig,
        _x_prev_clips=ones if x_prev_clip is None else x_prev_clip,
        _fista_t=jnp.asarray(fista_t),
    )


def _t_next(t: float) -> float:
    return float((1 + jnp.sqrt(1 + 4 * t**2)) / 2)


def _mom(t: float) -> float:
    return (t - 1) / _t_next(t)


# ── ParallelSigmaAndClipSchedule FISTA tests ──────────────────────────────────


class TestParallelSigmaAndClipFISTA:
    # ── config / flag ────────────────────────────────────────────────────────

    def test_use_fista_false_by_default_in_config(self):
        assert ParallelSigmaAndClipScheduleConfig().use_fista is False

    def test_use_fista_stored_on_schedule(self, params):
        assert _make_parallel(params, use_fista=True).use_fista is True
        assert _make_parallel(params, use_fista=False).use_fista is False

    def test_use_fista_preserved_through_project(self, params):
        assert _make_parallel(params, use_fista=True).project().use_fista is True

    def test_use_fista_preserved_through_extrapolate(self, params):
        assert _make_parallel(params, use_fista=True).fista_extrapolate().use_fista is True

    def test_use_fista_preserved_through_advance(self, params):
        s = _make_parallel(params, use_fista=True)
        x_new = _make_parallel(params)
        assert s.fista_advance(x_new).use_fista is True

    # ── initial state ────────────────────────────────────────────────────────

    def test_fista_t_initialised_to_one(self, params):
        assert float(_make_parallel(params)._fista_t) == pytest.approx(1.0)

    # ── fista_extrapolate ────────────────────────────────────────────────────

    def test_first_extrapolate_is_noop_on_sigmas(self, params):
        # t=1 => mom=0 => y = x; project first so sigmas/clips are on-constraint.
        s = _make_parallel(params).project()
        y = s.fista_extrapolate()
        assert jnp.allclose(y.get_private_sigmas(), s.get_private_sigmas(), atol=1e-5)

    def test_first_extrapolate_is_noop_on_clips(self, params):
        s = _make_parallel(params).project()
        y = s.fista_extrapolate()
        assert jnp.allclose(y.get_private_clips(), s.get_private_clips(), atol=1e-5)

    def test_first_extrapolate_seeds_x_curr_from_schedule(self, params):
        s = _make_parallel(params).project()
        y = s.fista_extrapolate()
        assert jnp.allclose(y._x_curr_sigmas, s.get_private_sigmas(), atol=1e-5)
        assert jnp.allclose(y._x_curr_clips, s.get_private_clips(), atol=1e-5)

    def test_first_extrapolate_seeds_x_prev_equal_to_x_curr(self, params):
        # x_prev == x_curr on first call so subsequent momentum starts from zero.
        s = _make_parallel(params).project()
        y = s.fista_extrapolate()
        assert jnp.allclose(y._x_prev_sigmas, y._x_curr_sigmas, atol=1e-5)
        assert jnp.allclose(y._x_prev_clips, y._x_curr_clips, atol=1e-5)

    def test_extrapolate_does_not_advance_t(self, params):
        s = _make_parallel(params, fista_t=2.5)
        assert float(s.fista_extrapolate()._fista_t) == pytest.approx(2.5)

    def test_extrapolate_carries_x_curr(self, params):
        # _x_curr after extrapolate == get_private_sigmas() at the time of the call.
        s = _make_parallel(params, noise_val=2.0)
        y = s.fista_extrapolate()
        assert jnp.allclose(y._x_curr_sigmas, s.get_private_sigmas(), atol=1e-5)

    def test_extrapolation_matches_formula(self, params):
        # y = x + mom*(x - x_prev) for a non-trivial t.
        t = 3.0
        x_val, x_prev_val = 2.0, 1.0
        expected_y = x_val + _mom(t) * (x_val - x_prev_val)

        s = _make_parallel(
            params,
            noise_val=x_val,
            x_prev_sig=jnp.ones(T) * x_prev_val,
            fista_t=t,
        )
        y = s.fista_extrapolate()
        assert jnp.allclose(y.get_private_sigmas(), expected_y * jnp.ones(T), atol=1e-5)

    def test_extrapolation_overshoots_x_when_moving_forward(self, params):
        # If x > x_prev, y should be > x (extrapolation continues the trend).
        s = _make_parallel(
            params,
            noise_val=2.0,
            x_prev_sig=jnp.ones(T) * 1.0,
            fista_t=3.0,
        )
        y = s.fista_extrapolate()
        assert jnp.all(y.get_private_sigmas() > s.get_private_sigmas())

    # ── fista_advance ────────────────────────────────────────────────────────

    def test_advance_advances_t_by_recurrence(self, params):
        t0 = 2.0
        s = _make_parallel(params, fista_t=t0)
        advanced = s.fista_advance(_make_parallel(params, noise_val=0.8))
        assert float(advanced._fista_t) == pytest.approx(_t_next(t0), rel=1e-5)

    def test_advance_sets_x_prev_to_old_x_curr(self, params):
        x_curr_val = jnp.ones(T) * 2.0
        s = _make_parallel(params, x_curr_sig=x_curr_val)
        advanced = s.fista_advance(_make_parallel(params, noise_val=0.8))
        assert jnp.allclose(advanced._x_prev_sigmas, x_curr_val, atol=1e-5)

    def test_advance_sets_x_curr_to_x_new_sigmas(self, params):
        x_new = _make_parallel(params, noise_val=0.8, clip_val=0.3)
        advanced = _make_parallel(params).fista_advance(x_new)
        assert jnp.allclose(advanced._x_curr_sigmas, x_new.get_private_sigmas(), atol=1e-5)
        assert jnp.allclose(advanced._x_curr_clips, x_new.get_private_clips(), atol=1e-5)

    # ── state preservation ───────────────────────────────────────────────────

    def test_apply_updates_preserves_fista_state(self, params):
        s = _make_parallel(
            params,
            fista_t=2.5,
            x_curr_sig=jnp.ones(T) * 2.0,
            x_curr_clip=jnp.ones(T) * 0.8,
            x_prev_sig=jnp.ones(T) * 1.5,
            x_prev_clip=jnp.ones(T) * 0.6,
        )
        zero_updates = jax.tree_util.tree_map(jnp.zeros_like, s)
        u = s.apply_updates(zero_updates)
        assert jnp.allclose(u._x_curr_sigmas, s._x_curr_sigmas)
        assert jnp.allclose(u._x_curr_clips, s._x_curr_clips)
        assert jnp.allclose(u._x_prev_sigmas, s._x_prev_sigmas)
        assert jnp.allclose(u._x_prev_clips, s._x_prev_clips)
        assert float(u._fista_t) == pytest.approx(float(s._fista_t))

    def test_project_preserves_fista_state(self, params):
        x_curr = jnp.ones(T) * 2.0
        x_prev = jnp.ones(T) * 1.5
        s = _make_parallel(params, fista_t=2.5, x_curr_sig=x_curr, x_prev_sig=x_prev)
        p = s.project()
        assert jnp.allclose(p._x_curr_sigmas, x_curr)
        assert jnp.allclose(p._x_prev_sigmas, x_prev)
        assert float(p._fista_t) == pytest.approx(2.5)

    # ── privacy constraint ───────────────────────────────────────────────────

    def test_project_satisfies_privacy_constraint(self, params):
        s = _make_parallel(params, noise_val=2.0, clip_val=1.0).project()
        expenditure = params.compute_expenditure(s.get_private_sigmas(), s.get_private_clips())
        assert float(expenditure) <= float(params.mu) + 1e-5


# ── WarmupParallelSigmaAndClipSchedule FISTA tests ────────────────────────────


class TestWarmupParallelSigmaAndClipFISTA:
    # ── config / flag ────────────────────────────────────────────────────────

    def test_use_fista_false_by_default_in_config(self):
        assert WarmupParallelSigmaAndClipScheduleConfig().use_fista is False

    def test_use_fista_stored_on_schedule(self, params):
        assert _make_warmup(params, use_fista=True).use_fista is True
        assert _make_warmup(params, use_fista=False).use_fista is False

    def test_use_fista_preserved_through_project(self, params):
        assert _make_warmup(params, use_fista=True).project().use_fista is True

    def test_use_fista_preserved_through_extrapolate(self, params):
        assert _make_warmup(params, use_fista=True).fista_extrapolate().use_fista is True

    def test_use_fista_preserved_through_advance(self, params):
        s = _make_warmup(params, use_fista=True)
        y = s.fista_extrapolate()
        x_new = y.project()
        assert y.fista_advance(x_new).use_fista is True

    # ── initial state ────────────────────────────────────────────────────────

    def test_fista_t_initialised_to_one(self, params):
        assert float(_make_warmup(params)._fista_t) == pytest.approx(1.0)

    # ── fista_extrapolate ────────────────────────────────────────────────────

    def test_first_extrapolate_is_noop(self, params):
        s = _make_warmup(params).project()
        y = s.fista_extrapolate()
        assert jnp.allclose(y.get_private_sigmas(), s.get_private_sigmas(), atol=1e-5)
        assert jnp.allclose(y.get_private_clips(), s.get_private_clips(), atol=1e-5)

    def test_extrapolate_does_not_advance_t(self, params):
        s = _make_warmup(params, fista_t=2.5)
        assert float(s.fista_extrapolate()._fista_t) == pytest.approx(2.5)

    def test_extrapolate_preserves_step_count(self, params):
        s = _make_warmup(params, step_count=3)
        assert int(s.fista_extrapolate().step_count) == 3

    # ── fista_advance: transition reset ──────────────────────────────────────

    def test_advance_resets_t_at_warmup_to_tail_transition(self, params):
        # At the last warmup step, project() bumps step_count to warmup_steps.
        # fista_advance must detect this and reset t to 1.
        s = _make_warmup(params, step_count=WARMUP_STEPS - 1, fista_t=5.0)
        y = s.fista_extrapolate()
        x_new = y.project()  # step_count -> WARMUP_STEPS => transition
        assert int(x_new.step_count) == WARMUP_STEPS  # confirm transition fired
        advanced = y.fista_advance(x_new)
        assert float(advanced._fista_t) == pytest.approx(1.0)

    def test_advance_resets_x_prev_to_x_new_at_transition(self, params):
        # After reset, x_prev == x_new so the first tail extrapolation has zero momentum.
        s = _make_warmup(params, step_count=WARMUP_STEPS - 1, fista_t=5.0)
        y = s.fista_extrapolate()
        x_new = y.project()
        advanced = y.fista_advance(x_new)
        assert jnp.allclose(advanced._x_prev_sigmas, x_new.get_private_sigmas(), atol=1e-5)
        assert jnp.allclose(advanced._x_prev_clips, x_new.get_private_clips(), atol=1e-5)

    # ── fista_advance: normal operation (no reset) ───────────────────────────

    def test_advance_does_not_reset_t_during_warmup(self, params):
        t0 = 3.0
        s = _make_warmup(params, step_count=1, fista_t=t0)
        y = s.fista_extrapolate()
        x_new = y.project()  # step_count -> 2, still < WARMUP_STEPS
        assert int(x_new.step_count) < WARMUP_STEPS  # confirm no transition
        advanced = y.fista_advance(x_new)
        assert float(advanced._fista_t) == pytest.approx(_t_next(t0), rel=1e-5)

    def test_advance_does_not_reset_t_in_tail_phase(self, params):
        t0 = 3.0
        s = _make_warmup(params, step_count=WARMUP_STEPS + 2, fista_t=t0)
        y = s.fista_extrapolate()
        x_new = y.project()  # step_count -> WARMUP_STEPS + 3, past transition
        advanced = y.fista_advance(x_new)
        assert float(advanced._fista_t) == pytest.approx(_t_next(t0), rel=1e-5)

    def test_advance_sets_x_curr_to_x_new_sigmas(self, params):
        s = _make_warmup(params, step_count=2)
        y = s.fista_extrapolate()
        x_new = y.project()
        advanced = y.fista_advance(x_new)
        assert jnp.allclose(advanced._x_curr_sigmas, x_new.get_private_sigmas(), atol=1e-5)

    def test_advance_sets_x_prev_to_old_x_curr_outside_transition(self, params):
        # Directly construct y with a known _x_curr to verify the promotion.
        x_curr_val = jnp.ones(T) * 2.0
        y = _make_warmup(params, step_count=1, fista_t=2.0, x_curr_sig=x_curr_val)
        x_new = y.project()  # step_count -> 2, no transition
        advanced = y.fista_advance(x_new)
        assert jnp.allclose(advanced._x_prev_sigmas, x_curr_val, atol=1e-5)

    # ── state preservation ───────────────────────────────────────────────────

    def test_apply_updates_preserves_fista_state(self, params):
        s = _make_warmup(
            params,
            fista_t=2.5,
            step_count=2,
            x_curr_sig=jnp.ones(T) * 2.0,
            x_curr_clip=jnp.ones(T) * 0.8,
            x_prev_sig=jnp.ones(T) * 1.5,
            x_prev_clip=jnp.ones(T) * 0.6,
        )
        zero_updates = jax.tree_util.tree_map(jnp.zeros_like, s)
        u = s.apply_updates(zero_updates)
        assert jnp.allclose(u._x_curr_sigmas, s._x_curr_sigmas)
        assert jnp.allclose(u._x_curr_clips, s._x_curr_clips)
        assert jnp.allclose(u._x_prev_sigmas, s._x_prev_sigmas)
        assert jnp.allclose(u._x_prev_clips, s._x_prev_clips)
        assert float(u._fista_t) == pytest.approx(float(s._fista_t))
        assert int(u.step_count) == int(s.step_count)

    def test_project_preserves_fista_state(self, params):
        x_curr = jnp.ones(T) * 2.0
        x_prev = jnp.ones(T) * 1.5
        s = _make_warmup(params, fista_t=2.5, x_curr_sig=x_curr, x_prev_sig=x_prev)
        p = s.project()
        assert jnp.allclose(p._x_curr_sigmas, x_curr)
        assert jnp.allclose(p._x_prev_sigmas, x_prev)
        assert float(p._fista_t) == pytest.approx(2.5)

    def test_project_increments_step_count(self, params):
        s = _make_warmup(params, step_count=2)
        assert int(s.project().step_count) == 3

    # ── privacy constraint ───────────────────────────────────────────────────

    def test_project_satisfies_privacy_constraint_in_warmup(self, params):
        s = _make_warmup(params, step_count=1, warmup_noise=2.0, warmup_clip=1.0).project()
        expenditure = params.compute_expenditure(s.get_private_sigmas(), s.get_private_clips())
        assert float(expenditure) <= float(params.mu) + 1e-5

    def test_project_satisfies_privacy_constraint_in_tail(self, params):
        # The tail schedule is seeded on the last warmup step, so we must run
        # through full warmup before testing the tail constraint.
        s = _make_warmup(params, warmup_noise=2.0, warmup_clip=1.0)
        for _ in range(WARMUP_STEPS):
            s = s.project()
        s = s.project()  # one tail-phase projection
        expenditure = params.compute_expenditure(s.get_private_sigmas(), s.get_private_clips())
        assert float(expenditure) <= float(params.mu) + 1e-5


# ── Property-based tests ──────────────────────────────────────────────────────

_t_st = st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False)
_pos_val_st = st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)
_fista_settings = settings(
    suppress_health_check=[HealthCheck.too_slow], deadline=None, max_examples=100
)


class TestFISTAMomentumProperties:
    def test_momentum_is_zero_at_t_equals_one(self):
        assert _mom(1.0) == pytest.approx(0.0, abs=1e-7)

    @given(_t_st)
    @_fista_settings
    def test_t_recurrence_is_strictly_increasing(self, t):
        assert _t_next(t) > t

    @given(_t_st)
    @_fista_settings
    def test_t_next_is_always_at_least_one(self, t):
        assert _t_next(t) >= 1.0

    @given(_t_st)
    @_fista_settings
    def test_momentum_coefficient_is_non_negative(self, t):
        assert _mom(t) >= 0.0

    @given(_t_st)
    @_fista_settings
    def test_momentum_coefficient_is_strictly_less_than_one(self, t):
        assert _mom(t) < 1.0

    @given(t0=_t_st, n_steps=st.integers(min_value=1, max_value=20))
    @_fista_settings
    def test_t_sequence_remains_above_one_after_multiple_steps(self, t0, n_steps):
        t = t0
        for _ in range(n_steps):
            t = _t_next(t)
        assert t >= 1.0

    @given(
        x_val=_pos_val_st,
        x_prev_val=_pos_val_st,
        t=st.floats(min_value=2.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], deadline=None, max_examples=50)
    def test_extrapolation_matches_formula_for_any_t(self, x_val, x_prev_val, t):
        """fista_extrapolate output == x + mom*(x - x_prev) for all valid inputs."""
        params = GDPPrivacyParameters(EPS, DELTA, P, T)
        expected_y = x_val + _mom(t) * (x_val - x_prev_val)
        s = _make_parallel(
            params,
            noise_val=x_val,
            x_prev_sig=jnp.ones(T) * x_prev_val,
            fista_t=t,
        )
        y = s.fista_extrapolate()
        assert jnp.allclose(y.get_private_sigmas(), expected_y * jnp.ones(T), atol=1e-4)

    @given(t0=_t_st)
    @_fista_settings
    def test_advance_t_matches_recurrence(self, t0):
        """fista_advance always sets _fista_t to exactly (1 + sqrt(1 + 4*t^2)) / 2."""
        params = GDPPrivacyParameters(EPS, DELTA, P, T)
        s = _make_parallel(params, fista_t=t0)
        x_new = _make_parallel(params, noise_val=0.9)
        advanced = s.fista_advance(x_new)
        assert float(advanced._fista_t) == pytest.approx(_t_next(t0), rel=1e-5)

    @given(x_curr_val=_pos_val_st, x_new_val=_pos_val_st)
    @_fista_settings
    def test_advance_x_prev_always_equals_old_x_curr(self, x_curr_val, x_new_val):
        """After advance, _x_prev always reflects what _x_curr was before the call."""
        params = GDPPrivacyParameters(EPS, DELTA, P, T)
        x_curr = jnp.ones(T) * x_curr_val
        s = _make_parallel(params, fista_t=2.0, x_curr_sig=x_curr)
        x_new = _make_parallel(params, noise_val=x_new_val)
        advanced = s.fista_advance(x_new)
        assert jnp.allclose(advanced._x_prev_sigmas, x_curr, atol=1e-5)
