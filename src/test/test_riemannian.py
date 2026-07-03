"""Tests for the Riemannian tangent-projection optax transform (ADR-0008).

Per the no-momentum design, the transform is *stateless*: it removes the
component of the gradient along the budget-manifold normal
n = ∇_θ constraint_value(schedule), leaving a tangent (Riemannian) gradient.
The retraction (schedule.project()) then returns to the manifold.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from policy.base_schedules.constant import ConstantSchedule
from policy.schedules.decoupled_sigma_and_clip import DecoupledSigmaAndClipSchedule
from privacy.rdp_privacy import RDPPrivacyParameters

T = 10


def _vdot(a, b) -> jnp.ndarray:
    la = jax.tree.leaves(eqx.filter(a, eqx.is_inexact_array))
    lb = jax.tree.leaves(eqx.filter(b, eqx.is_inexact_array))
    return sum(jnp.vdot(x, y) for x, y in zip(la, lb))


@pytest.fixture
def schedule() -> DecoupledSigmaAndClipSchedule:
    params = RDPPrivacyParameters(1.0, 1e-5, 0.1, T)
    noise = ConstantSchedule(value=0.5, T=T)
    clip = ConstantSchedule(value=1.0, T=T)
    sched = DecoupledSigmaAndClipSchedule(noise, clip, params)
    return sched.project().refresh_alpha_star()


class TestTangentProjection:
    def test_output_orthogonal_to_normal(self, schedule):
        from policy.riemannian import riemannian_tangent_projection

        # Arbitrary loss gradient touching both noise and clip leaves.
        grads = eqx.filter_grad(
            lambda s: (
                jnp.sum(s.get_private_noise_scales() ** 2) + jnp.sum(s.get_private_clips() ** 2)
            )
        )(schedule)

        tx = riemannian_tangent_projection()
        state = tx.init(schedule)
        proj, _ = tx.update(grads, state, schedule)

        normal = eqx.filter_grad(lambda s: s.constraint_value())(schedule)
        assert jnp.isclose(_vdot(normal, proj), 0.0, atol=1e-4)

    def test_pure_normal_maps_to_zero(self, schedule):
        from policy.riemannian import riemannian_tangent_projection

        normal = eqx.filter_grad(lambda s: s.constraint_value())(schedule)

        tx = riemannian_tangent_projection()
        state = tx.init(schedule)
        proj, _ = tx.update(normal, state, schedule)

        # Projecting the normal itself onto the tangent space yields ~0.
        assert jnp.sqrt(_vdot(proj, proj)) < 1e-4

    def test_clip_gradient_passes_through(self, schedule):
        from policy.riemannian import riemannian_tangent_projection

        # Gradient only on the clip side; the normal is zero there, so it is
        # left untouched (clips are free Euclidean params).
        grads = eqx.filter_grad(lambda s: jnp.sum(s.get_private_clips() ** 2))(schedule)

        tx = riemannian_tangent_projection()
        state = tx.init(schedule)
        proj, _ = tx.update(grads, state, schedule)

        assert jnp.allclose(proj.clip_schedule.value, grads.clip_schedule.value, atol=1e-6)
