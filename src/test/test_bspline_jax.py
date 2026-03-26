"""JAX-compatibility tests for BSplineSchedule.

Verifies that BSplineSchedule is fully compatible with the JAX transformations
used in the outer training loop:

  - eqx.filter_jit            — JIT compilation of forward passes
  - eqx.filter_grad           — gradient flow through control_points
  - eqx.filter_value_and_grad — the pattern used by the outer training loop
  - jax.vmap                  — batched evaluation over control-point arrays
  - jax.lax.scan              — using the schedule inside a scan loop (inner loop)
  - from_projection under JIT — called by project(), which is @eqx.filter_jit
"""

import equinox as eqx
import jax
import jax.lax as jlax
import jax.numpy as jnp
import jax.random as jr
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from policy.base_schedules.bspline import BSplineSchedule
from policy.base_schedules.config import BSplineScheduleConfig

# ---------------------------------------------------------------------------
# Shared constants — fixed so JAX doesn't recompile across test cases
# ---------------------------------------------------------------------------

T = 50
N_CP = 8
DEGREE = 3

_jax_settings = settings(
    max_examples=25,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)

_cp_st = st.lists(
    st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    min_size=N_CP,
    max_size=N_CP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sched(init: float = 1.0) -> BSplineSchedule:
    conf = BSplineScheduleConfig(
        num_control_points=N_CP,
        degree=DEGREE,
        init_value=init,
    )
    return BSplineSchedule.from_config(conf, T=T)


# ===========================================================================
# JIT compatibility
# ===========================================================================


class TestBSplineJITCompatibility:
    """eqx.filter_jit compiles forward passes and from_projection cleanly."""

    def test_filter_jit_get_valid_schedule(self):
        """JIT-compiled get_valid_schedule returns correct shape and finite values."""
        sched = _make_sched()
        out = eqx.filter_jit(lambda s: s.get_valid_schedule())(sched)
        assert out.shape == (T,)
        assert jnp.all(jnp.isfinite(out))

    def test_filter_jit_get_raw_schedule(self):
        """JIT-compiled get_raw_schedule returns correct shape and finite values."""
        sched = _make_sched()
        out = eqx.filter_jit(lambda s: s.get_raw_schedule())(sched)
        assert out.shape == (T,)
        assert jnp.all(jnp.isfinite(out))

    def test_jit_matches_eager(self):
        """JIT and eager produce identical outputs."""
        sched = _make_sched()
        eager = sched.get_valid_schedule()
        jitted = eqx.filter_jit(lambda s: s.get_valid_schedule())(sched)
        assert jnp.allclose(eager, jitted, atol=1e-6)

    def test_jit_consistent_across_calls(self):
        """Multiple calls with the same schedule structure return the same result."""
        sched = _make_sched()
        f = eqx.filter_jit(lambda s: s.get_valid_schedule())
        assert jnp.allclose(f(sched), f(sched))

    def test_filter_jit_from_projection(self):
        """from_projection compiles under JIT (it is called inside project())."""
        sched = _make_sched()
        projection = jnp.ones(T) * 2.0

        @eqx.filter_jit
        def project_and_eval(s, p):
            s2 = BSplineSchedule.from_projection(s, p)
            return s2.get_valid_schedule()

        out = project_and_eval(sched, projection)
        assert out.shape == (T,)
        assert jnp.all(jnp.isfinite(out))
        assert jnp.allclose(out, 2.0, atol=1e-3)


# ===========================================================================
# Gradient compatibility
# ===========================================================================


class TestBSplineGradientCompatibility:
    """Gradients flow through control_points; basis/basis_pinv are stop_gradient-ed."""

    def test_filter_grad_control_points_nonzero(self):
        """filter_grad gives non-zero finite gradients w.r.t. control_points."""
        sched = _make_sched()

        @eqx.filter_jit
        @eqx.filter_grad
        def loss_fn(s):
            return jnp.sum(s.get_valid_schedule())

        grads = loss_fn(sched)
        assert jnp.all(jnp.isfinite(grads.control_points))
        assert jnp.any(grads.control_points != 0)

    def test_basis_grad_is_zero(self):
        """stop_gradient on basis → its gradient leaf is zero."""
        sched = _make_sched()

        @eqx.filter_jit
        @eqx.filter_grad
        def loss_fn(s):
            return jnp.sum(s.get_valid_schedule())

        grads = loss_fn(sched)
        assert jnp.all(grads.basis == 0)

    def test_basis_pinv_grad_is_zero(self):
        """stop_gradient on basis_pinv → its gradient leaf is zero."""
        sched = _make_sched()

        @eqx.filter_jit
        @eqx.filter_grad
        def loss_fn(s):
            return jnp.sum(s.get_valid_schedule())

        grads = loss_fn(sched)
        assert jnp.all(grads.basis_pinv == 0)

    def test_filter_value_and_grad(self):
        """filter_value_and_grad — the pattern used by the outer training loop."""
        sched = _make_sched()

        @eqx.filter_jit
        @eqx.filter_value_and_grad
        def loss_and_grad(s):
            return jnp.mean(s.get_valid_schedule() ** 2)

        val, grads = loss_and_grad(sched)
        assert jnp.isfinite(val)
        assert jnp.all(jnp.isfinite(grads.control_points))
        assert jnp.any(grads.control_points != 0)

    def test_grad_through_from_projection(self):
        """Gradients flow through from_projection when projection depends on control_points.

        Uses get_valid_schedule() as the projection target so that the gradient
        path: loss → s2.control_points → from_projection → projection → s.control_points
        is non-zero.
        """
        sched = _make_sched()

        @eqx.filter_jit
        @eqx.filter_grad
        def loss_fn(s):
            projection = s.get_valid_schedule() * 2.0
            s2 = BSplineSchedule.from_projection(s, projection)
            return jnp.sum(s2.get_valid_schedule())

        grads = loss_fn(sched)
        assert jnp.all(jnp.isfinite(grads.control_points))
        assert jnp.any(grads.control_points != 0)

    @given(cp=_cp_st)
    @_jax_settings
    def test_grad_finite_for_any_control_points(self, cp):
        """Gradients are finite for any control-point values in [-5, 5]."""
        sched = _make_sched()
        sched = eqx.tree_at(lambda s: s.control_points, sched, jnp.array(cp, jnp.float32))

        @eqx.filter_jit
        @eqx.filter_grad
        def loss_fn(s):
            return jnp.sum(s.get_valid_schedule())

        grads = loss_fn(sched)
        assert jnp.all(jnp.isfinite(grads.control_points))


# ===========================================================================
# vmap compatibility
# ===========================================================================


class TestBSplineVmapCompatibility:
    """jax.vmap batches evaluation and differentiation over control-point arrays."""

    def test_vmap_get_valid_schedule(self):
        """vmap over a batch of control-point arrays yields (B, T) output."""
        B = 4
        sched = _make_sched()
        cp_batch = jnp.ones((B, N_CP))

        def eval_one(cp):
            s = eqx.tree_at(lambda s: s.control_points, sched, cp)
            return s.get_valid_schedule()

        out = jax.vmap(eval_one)(cp_batch)
        assert out.shape == (B, T)
        assert jnp.all(jnp.isfinite(out))
        assert jnp.all(out > 0)

    def test_vmap_grad_over_control_points(self):
        """vmap over per-sample gradients of a sum loss gives (B, N_CP) grads."""
        B = 4
        sched = _make_sched()
        cp_batch = jr.normal(jr.PRNGKey(0), (B, N_CP))

        def loss_one(cp):
            s = eqx.tree_at(lambda s: s.control_points, sched, cp)
            return jnp.sum(s.get_valid_schedule())

        grads = jax.vmap(jax.grad(loss_one))(cp_batch)
        assert grads.shape == (B, N_CP)
        assert jnp.all(jnp.isfinite(grads))

    def test_vmap_jit_get_valid_schedule(self):
        """vmap inside filter_jit produces correct shapes."""
        B = 6
        sched = _make_sched()
        cp_batch = jnp.ones((B, N_CP))

        @eqx.filter_jit
        def batched_eval(cp_b):
            def eval_one(cp):
                s = eqx.tree_at(lambda s: s.control_points, sched, cp)
                return s.get_valid_schedule()

            return jax.vmap(eval_one)(cp_b)

        out = batched_eval(cp_batch)
        assert out.shape == (B, T)
        assert jnp.all(jnp.isfinite(out))


# ===========================================================================
# lax.scan compatibility
# ===========================================================================


class TestBSplineScanCompatibility:
    """jax.lax.scan patterns that mirror the DP-SGD inner loop."""

    def test_scan_over_schedule_values(self):
        """lax.scan can iterate over get_valid_schedule() output."""
        sched = _make_sched()
        schedule = sched.get_valid_schedule()

        def step(carry, sigma_t):
            return carry + sigma_t, sigma_t

        total, outputs = jlax.scan(step, jnp.float32(0.0), schedule)
        assert outputs.shape == (T,)
        assert jnp.all(jnp.isfinite(outputs))
        assert jnp.isfinite(total)

    def test_scan_inside_filter_jit(self):
        """Schedule evaluated inside a jit+scan loop is JIT-compatible."""
        sched = _make_sched()

        @eqx.filter_jit
        def run(s):
            schedule = s.get_valid_schedule()

            def step(carry, x):
                return carry + x, x

            total, _ = jlax.scan(step, jnp.float32(0.0), schedule)
            return total

        result = run(sched)
        assert jnp.isfinite(result)
        assert float(result) == pytest.approx(float(jnp.sum(sched.get_valid_schedule())), rel=1e-5)

    def test_grad_through_scan(self):
        """Gradients propagate back through lax.scan that uses schedule values."""
        sched = _make_sched()

        @eqx.filter_jit
        @eqx.filter_grad
        def loss_fn(s):
            schedule = s.get_valid_schedule()

            def step(carry, sigma_t):
                return carry + sigma_t, None

            total, _ = jlax.scan(step, jnp.float32(0.0), schedule)
            return total

        grads = loss_fn(sched)
        assert jnp.all(jnp.isfinite(grads.control_points))
        assert jnp.any(grads.control_points != 0)
