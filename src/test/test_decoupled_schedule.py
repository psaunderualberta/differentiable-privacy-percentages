"""Tests for policy/schedules/decoupled_sigma_and_clip.py.

The DecoupledSigmaAndClipSchedule parameterises noise as σ_noise = C · σ_mult,
with σ_mult = 1/w, where w is the accountant's noise weight learned in the base
noise schedule and C is a separate, fully-decoupled clip schedule. Under RDP
accounting (ADR-0007/0008) the per-step cost depends on w only:
    ρ_total(α*; w) = c(α*)   is the on-budget manifold.
project() is the equality scaling retraction onto that manifold; C is untouched.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest

from policy.base_schedules.constant import ConstantSchedule
from policy.schedules.decoupled_sigma_and_clip import DecoupledSigmaAndClipSchedule
from privacy.rdp_accountant import rdp_to_epsilon
from privacy.rdp_privacy import RDPPrivacyParameters

EPS = 1.0
DELTA = 1e-5
P = 0.1
T = 10
ALPHAS = (2, 3, 4, 6, 8, 16, 32, 64)


@pytest.fixture
def params() -> RDPPrivacyParameters:
    return RDPPrivacyParameters(EPS, DELTA, P, T, alphas=ALPHAS)


@pytest.fixture
def schedule(params) -> DecoupledSigmaAndClipSchedule:
    # Initial σ_mult deliberately tiny (w = 10, way over budget) so project() fires.
    noise = ConstantSchedule(value=0.1, T=T)
    clip = ConstantSchedule(value=1.0, T=T)
    return DecoupledSigmaAndClipSchedule(
        noise_schedule=noise,
        clip_schedule=clip,
        privacy_params=params,
    )


class TestConstraintValue:
    def test_matches_params_constraint(self, schedule):
        expected = schedule.privacy_params.constraint(schedule.get_private_weights())
        assert jnp.isclose(schedule.constraint_value(), expected, rtol=1e-6)

    def test_grad_flows_to_noise_only(self, schedule):
        grads = eqx.filter_grad(lambda s: s.constraint_value())(schedule)
        # Cost depends on w = 1/σ_mult, so the noise side gets a nonzero gradient
        # and the decoupled clip side gets exactly zero.
        assert jnp.all(jnp.abs(grads.noise_schedule.value) > 0)
        assert jnp.all(grads.clip_schedule.value == 0)


class TestProjectOnBudget:
    def test_project_realises_target_eps(self, schedule):
        projected = schedule.project()
        w = projected.get_private_weights()
        got_eps = rdp_to_epsilon(list(ALPHAS), w, P, DELTA)
        assert jnp.isclose(got_eps, EPS, atol=1e-4)

    def test_constraint_zero_after_project_and_refresh(self, schedule):
        projected = schedule.project().refresh_alpha_star()
        assert jnp.isclose(projected.constraint_value(), 0.0, atol=1e-4)


class TestGetPrivateNoiseScales:
    def test_returns_C_times_sigma_mult(self, schedule):
        w = schedule.get_private_weights()
        c = schedule.get_private_clips()
        expected = c * (1.0 / w)
        noise_scales = schedule.get_private_noise_scales()
        assert noise_scales.shape == (T,)
        assert jnp.all(jnp.isfinite(noise_scales))
        assert jnp.all(noise_scales > 0)
        assert jnp.allclose(noise_scales, expected, atol=1e-6)


class TestClipUnchangedUnderProject:
    def test_clip_schedule_identity_through_project(self, schedule):
        """project() must leave the clip side bit-exact identical — C is decoupled."""
        c_before = schedule.get_private_clips()
        c_after = schedule.project().get_private_clips()
        assert jnp.array_equal(c_before, c_after)


class TestProjectionIdempotent:
    def test_project_twice_equals_project_once(self, schedule):
        once = schedule.project()
        twice = once.project()
        assert jnp.allclose(
            twice.get_private_weights(),
            once.get_private_weights(),
            atol=1e-5,
        )
