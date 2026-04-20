"""Unit tests for policy/schedules/dynamic_dpsgd.py::DynamicDPSGDSchedule."""

import importlib

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

importlib.import_module("policy.schedules.dynamic_dpsgd")

from policy.schedules.config import DynamicDPSGDScheduleConfig  # noqa: E402
from policy.schedules.dynamic_dpsgd import DynamicDPSGDSchedule  # noqa: E402
from privacy.gdp_privacy import GDPPrivacyParameters  # noqa: E402

# ---------------------------------------------------------------------------
# Shared test parameters
# ---------------------------------------------------------------------------
EPS = 1.0
DELTA = 0.126936737507
P = 0.1
T = 10

RHO_MU = 0.5
RHO_C = 0.5
C_0 = 1.5
EPS_CLAMP = 0.01


@pytest.fixture
def privacy_params() -> GDPPrivacyParameters:
    return GDPPrivacyParameters(EPS, DELTA, P, T)


@pytest.fixture
def singleton_with_max_sigma():
    from conf.config import Config, EnvConfig, ScheduleOptimizerConfig, SweepConfig, WandbConfig
    from conf.singleton_conf import SingletonConfig

    SingletonConfig.config = Config(
        wandb_conf=WandbConfig(),
        sweep=SweepConfig(
            env=EnvConfig(), schedule_optimizer=ScheduleOptimizerConfig(max_sigma=10.0)
        ),
    )
    yield 10.0
    SingletonConfig.config = None


@pytest.fixture
def schedule(privacy_params) -> DynamicDPSGDSchedule:
    return DynamicDPSGDSchedule(RHO_MU, RHO_C, C_0, privacy_params, EPS_CLAMP)


@pytest.fixture
def schedule_with_singleton(privacy_params, singleton_with_max_sigma) -> DynamicDPSGDSchedule:
    return DynamicDPSGDSchedule(RHO_MU, RHO_C, C_0, privacy_params, EPS_CLAMP)


# ---------------------------------------------------------------------------
# Construction and stored attributes
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_is_eqx_module(self, schedule):
        assert isinstance(schedule, eqx.Module)

    def test_rho_mu_stored(self, schedule):
        assert float(schedule.rho_mu) == pytest.approx(RHO_MU)

    def test_rho_C_stored(self, schedule):
        assert float(schedule.rho_C) == pytest.approx(RHO_C)

    def test_C_0_stored(self, schedule):
        assert float(schedule.C_0) == pytest.approx(C_0)

    def test_eps_property(self, schedule):
        assert float(schedule.eps) == pytest.approx(EPS_CLAMP)

    def test_privacy_params_T(self, schedule):
        assert int(schedule.privacy_params.T) == T

    def test_from_config(self, privacy_params):
        conf = DynamicDPSGDScheduleConfig(rho_mu=RHO_MU, rho_c=RHO_C, c_0=C_0)
        s = DynamicDPSGDSchedule.from_config(conf, privacy_params)
        assert float(s.rho_mu) == pytest.approx(RHO_MU)
        assert float(s.rho_C) == pytest.approx(RHO_C)
        assert float(s.C_0) == pytest.approx(C_0)

    def test_from_config_default_eps(self, privacy_params):
        """from_config uses the default eps=0.01 (not passed via config)."""
        conf = DynamicDPSGDScheduleConfig()
        s = DynamicDPSGDSchedule.from_config(conf, privacy_params)
        assert float(s.eps) == pytest.approx(0.01)

    def test_different_T_gives_different_iters(self, privacy_params):
        pp2 = GDPPrivacyParameters(EPS, DELTA, P, T * 2)
        s2 = DynamicDPSGDSchedule(RHO_MU, RHO_C, C_0, pp2)
        assert s2.iters.shape == (T * 2,)


# ---------------------------------------------------------------------------
# Properties: iters, mu_0, eps
# ---------------------------------------------------------------------------


class TestProperties:
    def test_iters_shape(self, schedule):
        assert schedule.iters.shape == (T,)

    def test_iters_values(self, schedule):
        assert jnp.array_equal(schedule.iters, jnp.arange(1, T + 1))

    def test_iters_start_at_1(self, schedule):
        assert int(schedule.iters[0]) == 1

    def test_iters_end_at_T(self, schedule):
        assert int(schedule.iters[-1]) == T

    def test_mu_0_positive(self, schedule):
        assert float(schedule.mu_0) > 0

    def test_mu_0_scalar(self, schedule):
        assert schedule.mu_0.ndim == 0

    def test_eps_clamp_value(self, schedule):
        assert float(schedule.eps) == pytest.approx(EPS_CLAMP)

    def test_mu_0_satisfies_privacy_budget(self, schedule, privacy_params):
        """Solved mu_0 reconstructs the correct total GDP mu."""
        pows = schedule.rho_mu ** (schedule.iters / schedule.iters.size)
        mu_reconstructed = jnp.sqrt(
            privacy_params.p**2 * jnp.sum(jnp.exp((pows * schedule.mu_0) ** 2) - 1)
        )
        assert jnp.isclose(mu_reconstructed, privacy_params.mu, rtol=1e-4)

    def test_larger_T_gives_smaller_mu_0(self, privacy_params):
        """More steps → smaller per-step GDP budget."""
        s1 = DynamicDPSGDSchedule(RHO_MU, RHO_C, C_0, privacy_params)
        pp2 = GDPPrivacyParameters(EPS, DELTA, P, T * 5)
        s2 = DynamicDPSGDSchedule(RHO_MU, RHO_C, C_0, pp2)
        assert float(s2.mu_0) < float(s1.mu_0)

    def test_larger_rho_mu_gives_smaller_mu_0(self, privacy_params):
        """With rho_mu > 1 the later steps spend more budget, so mu_0 must be smaller."""
        s_lo = DynamicDPSGDSchedule(0.5, RHO_C, C_0, privacy_params)
        s_hi = DynamicDPSGDSchedule(2.0, RHO_C, C_0, privacy_params)
        assert float(s_hi.mu_0) < float(s_lo.mu_0)


# ---------------------------------------------------------------------------
# get_private_sigmas
# ---------------------------------------------------------------------------


class TestGetPrivateSigmas:
    def test_shape(self, schedule):
        assert schedule.get_private_sigmas().shape == (T,)

    def test_all_positive(self, schedule):
        assert jnp.all(schedule.get_private_sigmas() > 0)

    def test_formula(self, schedule):
        """sigma_i = (C_0 / mu_0) * (rho_mu * rho_C)^(-i/T)"""
        expected = (schedule.C_0 / schedule.mu_0) * (schedule.rho_mu * schedule.rho_C) ** (
            -schedule.iters / T
        )
        assert jnp.allclose(schedule.get_private_sigmas(), expected, atol=1e-6)

    def test_monotone_increasing_when_rho_product_lt_1(self, privacy_params):
        """rho_mu * rho_C = 0.25 < 1 → (product)^(-i/T) = 4^(i/T) increases → sigmas increase."""
        s = DynamicDPSGDSchedule(0.5, 0.5, C_0, privacy_params)
        sigmas = s.get_private_sigmas()
        assert jnp.all(jnp.diff(sigmas) > 0)

    def test_monotone_decreasing_when_rho_product_gt_1(self, privacy_params):
        """rho_mu * rho_C = 4 > 1 → (product)^(-i/T) decreases → sigmas decrease."""
        s = DynamicDPSGDSchedule(2.0, 2.0, C_0, privacy_params)
        sigmas = s.get_private_sigmas()
        assert jnp.all(jnp.diff(sigmas) < 0)

    def test_constant_when_rho_product_eq_1(self, privacy_params):
        """rho_mu * rho_C = 1 → (product)^(-i/T) = 1 everywhere → constant sigmas."""
        s = DynamicDPSGDSchedule(2.0, 0.5, C_0, privacy_params)
        sigmas = s.get_private_sigmas()
        assert jnp.allclose(sigmas, sigmas[0], atol=1e-5)

    def test_finite_values(self, schedule):
        assert jnp.all(jnp.isfinite(schedule.get_private_sigmas()))

    def test_first_element_formula(self, schedule):
        expected_first = (C_0 / float(schedule.mu_0)) * (RHO_MU * RHO_C) ** (-1.0 / T)
        assert float(schedule.get_private_sigmas()[0]) == pytest.approx(expected_first, rel=1e-5)

    def test_last_element_formula(self, schedule):
        expected_last = (C_0 / float(schedule.mu_0)) * (RHO_MU * RHO_C) ** (-1.0)
        assert float(schedule.get_private_sigmas()[-1]) == pytest.approx(expected_last, rel=1e-5)


# ---------------------------------------------------------------------------
# get_private_clips
# ---------------------------------------------------------------------------


class TestGetPrivateClips:
    def test_shape(self, schedule):
        assert schedule.get_private_clips().shape == (T,)

    def test_all_positive(self, schedule):
        assert jnp.all(schedule.get_private_clips() > 0)

    def test_formula(self, schedule):
        """clip_i = C_0 * rho_C^(-i/T)"""
        expected = schedule.C_0 * schedule.rho_C ** (-schedule.iters / T)
        assert jnp.allclose(schedule.get_private_clips(), expected, atol=1e-6)

    def test_monotone_increasing_when_rho_C_lt_1(self, privacy_params):
        """rho_C < 1 → rho_C^(-i/T) increases with i → clips increase."""
        s = DynamicDPSGDSchedule(RHO_MU, 0.5, C_0, privacy_params)
        clips = s.get_private_clips()
        assert jnp.all(jnp.diff(clips) > 0)

    def test_monotone_decreasing_when_rho_C_gt_1(self, privacy_params):
        """rho_C > 1 → rho_C^(-i/T) decreases with i → clips decrease."""
        s = DynamicDPSGDSchedule(RHO_MU, 2.0, C_0, privacy_params)
        clips = s.get_private_clips()
        assert jnp.all(jnp.diff(clips) < 0)

    def test_constant_when_rho_C_eq_1(self, privacy_params):
        """rho_C = 1 → all clips equal C_0."""
        s = DynamicDPSGDSchedule(RHO_MU, 1.0, C_0, privacy_params)
        clips = s.get_private_clips()
        assert jnp.allclose(clips, C_0, atol=1e-5)

    def test_finite_values(self, schedule):
        assert jnp.all(jnp.isfinite(schedule.get_private_clips()))

    def test_first_element_formula(self, schedule):
        expected_first = C_0 * RHO_C ** (-1.0 / T)
        assert float(schedule.get_private_clips()[0]) == pytest.approx(expected_first, rel=1e-5)

    def test_last_element_formula(self, schedule):
        expected_last = C_0 * RHO_C ** (-1.0)
        assert float(schedule.get_private_clips()[-1]) == pytest.approx(expected_last, rel=1e-5)

    def test_scales_linearly_with_C_0(self, privacy_params):
        """Doubling C_0 doubles all clips (mu_0 is independent of C_0 for fixed rho_mu)."""
        s1 = DynamicDPSGDSchedule(RHO_MU, RHO_C, 1.0, privacy_params)
        s2 = DynamicDPSGDSchedule(RHO_MU, RHO_C, 2.0, privacy_params)
        assert jnp.allclose(s2.get_private_clips(), 2.0 * s1.get_private_clips(), atol=1e-5)


# ---------------------------------------------------------------------------
# clip / sigma ratio invariant
# ---------------------------------------------------------------------------


class TestClipSigmaRatio:
    def test_ratio_equals_mu_0_times_rho_mu_power(self, schedule):
        """clip_i / sigma_i = mu_0 * rho_mu^(i/T) (derived from the two formulas)."""
        sigmas = schedule.get_private_sigmas()
        clips = schedule.get_private_clips()
        expected_ratio = schedule.mu_0 * schedule.rho_mu ** (schedule.iters / T)
        assert jnp.allclose(clips / sigmas, expected_ratio, atol=1e-5)

    def test_ratio_increases_with_step_when_rho_mu_gt_1(self, privacy_params):
        """rho_mu > 1 → mu_0 * rho_mu^(i/T) increases with i → ratio increases."""
        s = DynamicDPSGDSchedule(2.0, RHO_C, C_0, privacy_params)
        ratio = s.get_private_clips() / s.get_private_sigmas()
        assert jnp.all(jnp.diff(ratio) > 0)

    def test_ratio_decreases_with_step_when_rho_mu_lt_1(self, privacy_params):
        """rho_mu < 1 → mu_0 * rho_mu^(i/T) decreases with i → ratio decreases."""
        s = DynamicDPSGDSchedule(0.5, RHO_C, C_0, privacy_params)
        ratio = s.get_private_clips() / s.get_private_sigmas()
        assert jnp.all(jnp.diff(ratio) < 0)

    def test_constant_ratio_when_rho_mu_eq_1(self, privacy_params):
        """rho_mu = 1 → ratio = mu_0 everywhere (constant)."""
        s = DynamicDPSGDSchedule(1.0, RHO_C, C_0, privacy_params)
        ratio = s.get_private_clips() / s.get_private_sigmas()
        assert jnp.allclose(ratio, s.mu_0, atol=1e-5)


# ---------------------------------------------------------------------------
# get_private_weights (requires SingletonConfig)
# ---------------------------------------------------------------------------


class TestGetPrivateWeights:
    @pytest.fixture(autouse=True)
    def _singleton(self, singleton_with_max_sigma):
        pass

    def test_shape(self, schedule_with_singleton):
        assert schedule_with_singleton.get_private_weights().shape == (T,)

    def test_all_positive(self, schedule_with_singleton):
        assert jnp.all(schedule_with_singleton.get_private_weights() > 0)

    def test_all_finite(self, schedule_with_singleton):
        assert jnp.all(jnp.isfinite(schedule_with_singleton.get_private_weights()))

    def test_privacy_constraint_satisfied(self, schedule_with_singleton, privacy_params):
        """Projected weights satisfy sum(exp(w_i^2)) <= (mu/p)^2 + T."""
        weights = schedule_with_singleton.get_private_weights()
        bound = float((privacy_params.mu / privacy_params.p) ** 2 + privacy_params.T)
        actual = float(jnp.sum(jnp.exp(weights**2)))
        assert actual <= bound + 1e-2


# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------


class TestProject:
    def test_returns_same_type(self, schedule):
        assert isinstance(schedule.project(), DynamicDPSGDSchedule)

    def test_clamps_rho_mu_below_eps(self, privacy_params):
        s = DynamicDPSGDSchedule(0.0001, RHO_C, C_0, privacy_params, eps=0.01)
        assert float(s.project().rho_mu) == pytest.approx(0.01, abs=1e-6)

    def test_clamps_rho_C_below_eps(self, privacy_params):
        s = DynamicDPSGDSchedule(RHO_MU, 0.0001, C_0, privacy_params, eps=0.01)
        assert float(s.project().rho_C) == pytest.approx(0.01, abs=1e-6)

    def test_clamps_C_0_below_eps(self, privacy_params):
        s = DynamicDPSGDSchedule(RHO_MU, RHO_C, 0.0001, privacy_params, eps=0.01)
        assert float(s.project().C_0) == pytest.approx(0.01, abs=1e-6)

    def test_valid_params_unchanged(self, schedule):
        """All params above eps — project() returns the same values."""
        projected = schedule.project()
        assert float(projected.rho_mu) == pytest.approx(float(schedule.rho_mu))
        assert float(projected.rho_C) == pytest.approx(float(schedule.rho_C))
        assert float(projected.C_0) == pytest.approx(float(schedule.C_0))

    def test_idempotent(self, schedule):
        """project(project(s)) == project(s)."""
        once = schedule.project()
        twice = once.project()
        assert float(twice.rho_mu) == pytest.approx(float(once.rho_mu))
        assert float(twice.rho_C) == pytest.approx(float(once.rho_C))
        assert float(twice.C_0) == pytest.approx(float(once.C_0))

    def test_projected_sigmas_positive(self, schedule):
        assert jnp.all(schedule.project().get_private_sigmas() > 0)

    def test_projected_clips_positive(self, schedule):
        assert jnp.all(schedule.project().get_private_clips() > 0)

    def test_mu_0_recalculated_consistently_after_clamp(self, privacy_params):
        """After clamping rho_mu, the new mu_0 must still satisfy the privacy budget."""
        s = DynamicDPSGDSchedule(0.0001, RHO_C, C_0, privacy_params, eps=0.01)
        projected = s.project()
        pows = projected.rho_mu ** (projected.iters / projected.iters.size)
        mu_reconstructed = jnp.sqrt(
            privacy_params.p**2 * jnp.sum(jnp.exp((pows * projected.mu_0) ** 2) - 1)
        )
        assert jnp.isclose(mu_reconstructed, privacy_params.mu, rtol=1e-4)

    def test_all_params_ge_eps_after_project(self, privacy_params):
        """All three learnable parameters are >= eps after project()."""
        eps_clamp = 0.05
        s = DynamicDPSGDSchedule(0.001, 0.001, 0.001, privacy_params, eps=eps_clamp)
        projected = s.project()
        assert float(projected.rho_mu) == pytest.approx(eps_clamp, abs=1e-6)
        assert float(projected.rho_C) == pytest.approx(eps_clamp, abs=1e-6)
        assert float(projected.C_0) == pytest.approx(eps_clamp, abs=1e-6)


# ---------------------------------------------------------------------------
# apply_updates
# ---------------------------------------------------------------------------


class TestApplyUpdates:
    def test_returns_same_type(self, schedule):
        params = eqx.filter(schedule, eqx.is_array)
        opt = optax.sgd(0.0)
        opt_state = opt.init(params)
        grads = jax.tree.map(jnp.zeros_like, params)
        updates, _ = opt.update(grads, opt_state)
        result = schedule.apply_updates(updates)
        assert isinstance(result, DynamicDPSGDSchedule)

    def test_zero_updates_leave_params_unchanged(self, schedule):
        """Applying all-zero gradients does not move any learnable parameter."""
        params = eqx.filter(schedule, eqx.is_array)
        opt = optax.sgd(0.0)
        opt_state = opt.init(params)
        grads = jax.tree.map(jnp.zeros_like, params)
        updates, _ = opt.update(grads, opt_state)
        result = schedule.apply_updates(updates)
        assert float(result.rho_mu) == pytest.approx(float(schedule.rho_mu))
        assert float(result.rho_C) == pytest.approx(float(schedule.rho_C))
        assert float(result.C_0) == pytest.approx(float(schedule.C_0))

    def test_nonzero_update_changes_rho_mu(self, schedule):
        """A gradient step on rho_mu shifts it by approximately lr * grad."""
        lr = 0.1
        params = eqx.filter(schedule, eqx.is_array)
        opt = optax.sgd(lr)
        opt_state = opt.init(params)
        # Simulate gradient of 1.0 on rho_mu only
        grads = jax.tree.map(jnp.zeros_like, params)
        grads = eqx.tree_at(lambda m: m.rho_mu, grads, jnp.ones_like(schedule.rho_mu))
        updates, _ = opt.update(grads, opt_state)
        result = schedule.apply_updates(updates)
        assert float(result.rho_mu) == pytest.approx(float(schedule.rho_mu) - lr, rel=1e-4)


# ---------------------------------------------------------------------------
# _get_log_arrays (requires SingletonConfig)
# ---------------------------------------------------------------------------


class TestGetLogArrays:
    @pytest.fixture(autouse=True)
    def _singleton(self, singleton_with_max_sigma):
        pass

    def test_returns_dict(self, schedule_with_singleton):
        assert isinstance(schedule_with_singleton._get_log_arrays(), dict)

    def test_expected_keys(self, schedule_with_singleton):
        assert set(schedule_with_singleton._get_log_arrays().keys()) == {
            "sigmas",
            "clips",
            "weights",
            "mus",
        }

    def test_all_arrays_shape(self, schedule_with_singleton):
        for key, arr in schedule_with_singleton._get_log_arrays().items():
            assert arr.shape == (T,), f"'{key}' has shape {arr.shape}"

    def test_sigmas_match_get_private_sigmas(self, schedule_with_singleton):
        arrays = schedule_with_singleton._get_log_arrays()
        assert jnp.allclose(arrays["sigmas"], schedule_with_singleton.get_private_sigmas())

    def test_clips_match_get_private_clips(self, schedule_with_singleton):
        arrays = schedule_with_singleton._get_log_arrays()
        assert jnp.allclose(arrays["clips"], schedule_with_singleton.get_private_clips())

    def test_weights_positive(self, schedule_with_singleton):
        assert jnp.all(schedule_with_singleton._get_log_arrays()["weights"] > 0)

    def test_mus_positive(self, schedule_with_singleton):
        assert jnp.all(schedule_with_singleton._get_log_arrays()["mus"] > 0)

    def test_all_values_finite(self, schedule_with_singleton):
        for key, arr in schedule_with_singleton._get_log_arrays().items():
            assert jnp.all(jnp.isfinite(arr)), f"'{key}' has non-finite values"
