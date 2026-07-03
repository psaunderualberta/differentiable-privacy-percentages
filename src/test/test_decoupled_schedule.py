"""Tests for policy/schedules/decoupled_sigma_and_clip.py.

The DecoupledSigmaAndClipSchedule parameterises noise as σ_noise = C · σ_mult,
with σ_mult = 1/w, where w is learned directly in the base noise schedule and
C is a separate, fully-decoupled clip schedule. Privacy constraint:
    Σ exp(w_i²) ≤ (μ/p)² + T
C is not part of the constraint.
"""

import jax.numpy as jnp
import pytest

from policy.base_schedules.constant import ConstantSchedule
from policy.schedules.decoupled_sigma_and_clip import DecoupledSigmaAndClipSchedule
from privacy.gdp_privacy import GDPPrivacyParameters

EPS = 1.0
DELTA = 0.126936737507
P = 0.1
T = 10


@pytest.fixture
def params() -> GDPPrivacyParameters:
    return GDPPrivacyParameters(EPS, DELTA, P, T)


def _is_budget(params: GDPPrivacyParameters) -> float:
    """The RHS of the project_inverse_sigmas constraint: (mu/p)^2 + T."""
    return float((params.mu / params.p) ** 2 + params.T)


def _is_constraint(sigmas: jnp.ndarray) -> float:
    """Evaluate sum_i exp(1 / sigma_i)."""
    return float(jnp.sum(jnp.exp(1.0 / sigmas)))


@pytest.fixture
def schedule(params) -> DecoupledSigmaAndClipSchedule:
    # Initial w deliberately *above* the privacy budget so project() fires.
    # Σ exp(1/s²) = T·exp(4) ≈ 546 >> bound ≈ 110.
    noise = ConstantSchedule(value=0.1, T=T)
    clip = ConstantSchedule(value=1.0, T=T)
    return DecoupledSigmaAndClipSchedule(
        noise_schedule=noise,
        clip_schedule=clip,
        privacy_params=params,
    )


class TestInitProjectsToMu0:
    def test_uniform_w_projects_to_mu_0(self, schedule, params):
        """After one project() call from a uniform-but-off-surface init,
        the w-side schedule is uniform at μ₀."""

        projected = schedule.project()
        w = projected.get_private_weights()
        assert w.shape == (T,)
        assert jnp.allclose(w, params.mu_0**2, atol=1e-5)


class TestGetPrivateNoiseScales:
    def test_returns_C_times_one_over_w(self, schedule):
        """get_private_noise_scales() == C · (1/w) elementwise."""
        w = schedule.get_private_weights()
        c = schedule.get_private_clips()
        expected = c * (1.0 / w)
        noise_scales = schedule.get_private_noise_scales()
        assert noise_scales.shape == (T,)
        assert jnp.all(jnp.isfinite(noise_scales))
        assert jnp.all(noise_scales > 0)
        assert jnp.allclose(noise_scales, expected, atol=1e-6)


class TestPrivacyBudgetSatisfied:
    def test_budget_holds_after_project_nonuniform_init(self, params):
        """Σ exp(w_i²) ≤ (μ/p)² + T after project() from a non-uniform init."""
        import equinox as eqx_

        from policy.base_schedules.bspline import BSplineSchedule
        from policy.base_schedules.config import BSplineScheduleConfig

        # Non-uniform initial w via BSpline w/ varied control points, large
        # enough to violate the budget so project() actually fires.
        bspline = BSplineSchedule.from_config(
            BSplineScheduleConfig(num_control_points=5, degree=3, init_value=2.0),
            T=T,
        )
        noise = eqx_.tree_at(
            lambda s: s.control_points,
            bspline,
            jnp.asarray([1.5, 2.0, 2.5, 2.0, 1.5], dtype=jnp.float32),
        )
        clip = ConstantSchedule(value=1.0, T=T)
        sched = DecoupledSigmaAndClipSchedule(noise, clip, params)
        projected = sched.project()
        w = projected.get_private_weights()
        budget = (params.mu / params.p) ** 2 + params.T
        lhs = float(jnp.sum(jnp.exp(w**2)))
        assert lhs <= budget + 1e-3


class TestDPEquivalenceSmoke:
    def test_matches_joint_schedule_outputs(self, params):
        """A decoupled schedule with (C=clip, w=clip/σ) yields the same
        (noise_std, clips) as the joint SigmaAndClipSchedule it mirrors —
        i.e. dp.py would see identical inputs."""
        from policy.schedules.sigma_and_clip import SigmaAndClipSchedule

        sigma_val = 1.2
        clip_val = 0.6
        joint = SigmaAndClipSchedule(
            noise_schedule=ConstantSchedule(value=sigma_val, T=T),
            clip_schedule=ConstantSchedule(value=clip_val, T=T),
            privacy_params=params,
        )
        decoupled = DecoupledSigmaAndClipSchedule(
            noise_schedule=ConstantSchedule(value=sigma_val / clip_val, T=T),
            clip_schedule=ConstantSchedule(value=clip_val, T=T),
            privacy_params=params,
        )
        assert jnp.allclose(
            decoupled.get_private_noise_scales(),
            joint.get_private_noise_scales(),
            atol=1e-6,
        )
        assert jnp.allclose(
            decoupled.get_private_clips(),
            joint.get_private_clips(),
            atol=1e-6,
        )


class TestClipUnchangedUnderProject:
    def test_clip_schedule_identity_through_project(self, schedule):
        """project() must leave the clip side bit-exact identical — C is decoupled."""
        c_before = schedule.get_private_clips()
        projected = schedule.project()
        c_after = projected.get_private_clips()
        assert jnp.array_equal(c_before, c_after)


class TestProjectionIdempotent:
    def test_project_twice_equals_project_once(self, schedule):
        """project() ∘ project() == project() on the w-side."""
        once = schedule.project()
        twice = once.project()
        assert jnp.allclose(
            twice.get_private_weights(),
            once.get_private_weights(),
            atol=1e-5,
        )
