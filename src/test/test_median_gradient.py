"""Tests for the adaptive-clip baseline (`StatefulMedianGradientNoiseAndClipSchedule`).

The schedule steers the *within-clip fraction* toward the quantile target and
releases that fraction privately, as a second vector group of the same per-step
Gaussian mechanism as the gradient release (ADR 0013).
"""

import importlib
import re

import jax.numpy as jnp
import jax.random as jr
import pytest
from jaxtyping import Array

importlib.import_module("policy.stateful_schedules.median_gradient")

from conf.config import (  # noqa: E402
    Config,
    EnvConfig,
    ScheduleOptimizerConfig,
    SweepConfig,
    WandbConfig,
)
from conf.singleton_conf import SingletonConfig  # noqa: E402
from policy.stateful_schedules.median_gradient import (  # noqa: E402
    StatefulMedianGradientNoiseAndClipSchedule,
)
from privacy.gdp_privacy import GDPPrivacyParameters  # noqa: E402

EPS = 1.0
DELTA = 1e-6
P = 0.1
T = 100

C_0 = 1.0
ETA_C = 0.2
BATCH_SIZE = 250


@pytest.fixture
def singleton(request):
    batch_size = getattr(request, "param", BATCH_SIZE)
    SingletonConfig.config = Config(
        wandb_conf=WandbConfig(),
        sweep=SweepConfig(
            env=EnvConfig(batch_size=batch_size),
            schedule_optimizer=ScheduleOptimizerConfig(max_sigma=10.0),
        ),
    )
    yield batch_size
    SingletonConfig.config = None


@pytest.fixture
def privacy_params() -> GDPPrivacyParameters:
    return GDPPrivacyParameters(EPS, DELTA, P, T)


@pytest.fixture
def schedule(singleton, privacy_params) -> StatefulMedianGradientNoiseAndClipSchedule:
    return StatefulMedianGradientNoiseAndClipSchedule(C_0, ETA_C, privacy_params)


def test_count_release_carves_its_budget_out_of_the_per_step_mu(schedule, privacy_params):
    """μ_count and μ_grad split the un-amplified per-step budget μ₀ pythagoreanly."""
    mu_0 = privacy_params.mu_0

    assert schedule.mu_count**2 + schedule.mu_grad**2 == pytest.approx(float(mu_0**2), rel=1e-6)
    assert schedule.mu_grad < mu_0
    # Andrew's parameterisation: count noise is L/r, sensitivity of the ±½ sum is ½.
    assert schedule.mu_count == pytest.approx(0.5 * schedule.r / BATCH_SIZE, rel=1e-6)
    assert schedule.rho == pytest.approx(float(schedule.mu_count**2 / mu_0**2), rel=1e-6)


def test_gradient_noise_is_calibrated_to_the_reduced_gradient_budget(schedule, privacy_params):
    """σ = C/μ_grad, so paying for the count release raises the gradient noise."""
    state = schedule.get_initial_state()

    assert float(state.get_noise()) == pytest.approx(float(C_0 / schedule.mu_grad), rel=1e-6)
    assert float(state.get_noise()) > float(C_0 / privacy_params.mu_0)


def _grads_with_norms(norms: Array) -> Array:
    """Per-example gradients (one row each) whose global norms are `norms`."""
    return jnp.asarray(norms)[:, None] * jnp.ones((len(norms), 1))


def _step(schedule, state, norms, key, valid=None):
    grads = _grads_with_norms(norms)
    if valid is None:
        valid = jnp.ones(len(norms), dtype=bool)
    return schedule.update_state(
        state,
        grads,
        jnp.asarray(0),
        jnp.zeros((len(norms), 1)),
        jnp.zeros((len(norms), 1)),
        valid,
        key,
    )


def test_within_clip_fraction_is_released_with_noise(schedule):
    """The same batch yields different clip updates under different keys."""
    state = schedule.get_initial_state()
    norms = jnp.linspace(0.1, 2.0, 250)

    clips = [float(_step(schedule, state, norms, jr.PRNGKey(seed)).get_clip()) for seed in range(5)]

    assert len(set(clips)) == len(clips)


def _released_fractions(schedule, state, norms, seeds, valid=None):
    """Recover the released within-clip fraction from the resulting clip update."""
    clips = jnp.asarray(
        [float(_step(schedule, state, norms, jr.PRNGKey(s), valid).get_clip()) for s in seeds]
    )
    return jnp.log(clips / state.get_clip()) / -schedule.eta_c + schedule.gamma


def test_released_fraction_is_centred_on_the_truth_with_standard_error_one_over_r(schedule):
    """The count noise ratio r fixes std(b̄) = 1/r regardless of the privacy regime."""
    state = schedule.get_initial_state()
    # Half the batch sits at or below C_0, so the true within-clip fraction is 0.5.
    norms = jnp.concatenate([jnp.full(125, 0.5), jnp.full(125, 2.0)])

    released = _released_fractions(schedule, state, norms, range(400))

    assert float(released.mean()) == pytest.approx(0.5, abs=0.01)
    assert float(released.std()) == pytest.approx(1.0 / schedule.r, rel=0.15)


def test_count_is_divided_by_the_public_expected_batch_size(schedule):
    """The divisor is L, never the truncated-Poisson buffer nor realised occupancy."""
    state = schedule.get_initial_state()
    # A 300-row buffer whose 200 occupied rows are all within the clip. Under the ±½
    # encoding the three candidate divisors are distinguishable: the public L gives
    # 100/250 + ½ = 0.9, the realised occupancy 100/200 + ½ = 1.0, the buffer B 0.83.
    norms = jnp.full(300, 0.5)
    valid = jnp.arange(300) < 200

    released = _released_fractions(schedule, state, norms, range(400), valid=valid)

    assert float(released.mean()) == pytest.approx(100 / BATCH_SIZE + 0.5, abs=0.01)


def test_noisy_fraction_is_clamped_to_a_valid_fraction(singleton, privacy_params):
    """b̄ ∈ [0,1] bounds the geometric clip step, however heavy the count noise."""
    # r ≪ 20 makes the count noise swamp the signal, so an unclamped b̄ would
    # routinely land far outside [0,1] and blow the clip up or down.
    schedule = StatefulMedianGradientNoiseAndClipSchedule(C_0, ETA_C, privacy_params, r=0.5)
    state = schedule.get_initial_state()
    norms = jnp.full(250, 0.5)

    clips = jnp.asarray(
        [float(_step(schedule, state, norms, jr.PRNGKey(s)).get_clip()) for s in range(200)]
    )

    assert float(clips.min()) >= float(C_0 * jnp.exp(-ETA_C * (1 - schedule.gamma))) - 1e-6
    assert float(clips.max()) <= float(C_0 * jnp.exp(ETA_C * schedule.gamma)) + 1e-6


def test_clip_threshold_never_reaches_zero(singleton, privacy_params):
    """`postprocess_update` divides by C_t, so C carries an absolute floor."""
    schedule = StatefulMedianGradientNoiseAndClipSchedule(1e-30, ETA_C, privacy_params)
    state = schedule.get_initial_state()
    # Every gradient is within the clip, so the fraction is 1 and C shrinks.
    norms = jnp.zeros(250)

    new_state = _step(schedule, state, norms, jr.PRNGKey(0))

    assert float(new_state.get_clip()) == pytest.approx(schedule.c_min)
    assert float(new_state.get_noise()) > 0.0


@pytest.mark.parametrize("singleton", [32], indirect=True)
def test_construction_fails_when_the_budget_cannot_absorb_the_count(singleton, privacy_params):
    """A small batch makes μ_count rival μ₀; fail loudly rather than NaN thousands of steps later."""
    with pytest.raises(ValueError) as excinfo:
        StatefulMedianGradientNoiseAndClipSchedule(C_0, ETA_C, privacy_params)

    message = str(excinfo.value)
    assert "batch_size" in message and "32" in message
    # The operator needs to know what to change: r, the realised ρ, and the smallest
    # batch size that would work.
    assert "20" in message and "rho" in message.lower()


def test_construction_reports_the_derived_median_budget_fraction(singleton, privacy_params, capsys):
    """ρ is a reported diagnostic — how much privacy honest adaptivity cost."""
    schedule = StatefulMedianGradientNoiseAndClipSchedule(C_0, ETA_C, privacy_params)

    out = capsys.readouterr().out
    numbers = [float(m) for m in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", out)]

    assert "rho" in out.lower()
    assert any(n == pytest.approx(float(schedule.rho), rel=1e-2) for n in numbers)


def test_schedule_logs_nothing_per_timestep(schedule):
    """There is no per-step weight/μ schedule to log — the clip is chosen at runtime."""
    assert schedule.get_logging_schemas() == []
    assert schedule.get_loggables() == []
