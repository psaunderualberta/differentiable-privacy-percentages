"""Tests for the NES (Natural Evolution Strategies) gradient estimators.

These tests target the pure math in ``environments.nes`` so they don't need
to spin up the DP-SGD inner loop. Behaviour-level integration with the outer
loop is covered separately.
"""

import jax
import jax.numpy as jnp
import jax.random as jr

from environments.nes import (
    ESState,
    _nes_log_utilities,
    nes_es_step,
    nes_log_sigma_gradient,
)


def test_log_sigma_gradient_hand_computed_single_pair():
    """One antithetic pair, 2-D parameter, hand-computed expected value.

    eps = [[1.0, 0.0]]  →  ‖ε‖² = 1, d = 2.
    u_pos = u_neg = 1.0.
    Wierstra et al. (sNES, single scalar σ):
        ∇_{log σ} J = (1/(2d)) · Σ_pair (u_pos + u_neg) · (‖ε‖² − d)
                    = (1/4) · 2 · (1 − 2) = -0.5
    """
    eps = {"theta": jnp.array([[1.0, 0.0]])}  # leading axis = half_pop = 1
    u_pos = jnp.array([1.0])
    u_neg = jnp.array([1.0])

    g = nes_log_sigma_gradient(eps, u_pos, u_neg)

    assert g.shape == ()
    assert jnp.allclose(g, -0.5)


def test_log_sigma_gradient_unbiased_when_fitness_independent_of_eps():
    """Estimator unbiasedness: when fitnesses are statistically independent
    of the perturbations, the σ-gradient averages to ~0 over many trials.
    """
    half_pop = 16
    d = 8
    n_trials = 200

    def trial(seed):
        k_eps, k_fit = jr.split(jr.PRNGKey(seed))
        eps = {"theta": jr.normal(k_eps, (half_pop, d))}
        fitnesses = jr.normal(k_fit, (2 * half_pop,))
        u = _nes_log_utilities(fitnesses)
        return nes_log_sigma_gradient(eps, u[0::2], u[1::2])

    grads = jax.vmap(trial)(jnp.arange(n_trials))
    # Standard error of the mean is small relative to per-trial scale.
    assert jnp.abs(jnp.mean(grads)) < 0.05


def test_log_sigma_gradient_sign_matches_winning_norm():
    """Sign test: when low-loss samples have larger ‖ε‖² (so the search
    distribution is currently too narrow), the σ-gradient should be
    positive (NES maximisation convention) — i.e. "grow σ"."""
    half_pop = 64
    key = jr.PRNGKey(0)
    eps = {"theta": jr.normal(key, (half_pop, 10))}

    # Construct fitness so that large ‖ε‖² → lower loss for both pair members.
    sq = jnp.sum(eps["theta"] ** 2, axis=1)  # (half_pop,)
    # Full-population loss vector, antithetic-flattened [pos_0, neg_0, ...].
    loss = jnp.stack([-sq, -sq], axis=1).reshape(-1)
    u = _nes_log_utilities(loss)

    g_grow = nes_log_sigma_gradient(eps, u[0::2], u[1::2])
    assert g_grow > 0.0

    # And the mirrored case (large ‖ε‖² → higher loss) ⇒ negative gradient.
    loss_mirror = jnp.stack([sq, sq], axis=1).reshape(-1)
    u_m = _nes_log_utilities(loss_mirror)
    g_shrink = nes_log_sigma_gradient(eps, u_m[0::2], u_m[1::2])
    assert g_shrink < 0.0


def test_nes_es_step_is_jit_compatible():
    """Regression: ``_total_dimension`` previously did
    ``int(jnp.prod(jnp.array(leaf.shape[1:])))``, which raises
    ``ConcretizationTypeError`` under ``jax.jit`` because ``jnp.array(...)``
    of a shape tuple returns a tracer when traced. Shapes are static Python
    ints, so the dimension count must be computed in pure Python.
    """
    half_pop = 4
    eps = {"theta": jr.normal(jr.PRNGKey(0), (half_pop, 3, 5))}
    fitnesses = jr.normal(jr.PRNGKey(1), (2 * half_pop,))
    state = ESState(log_sigma=jnp.float32(0.0), eta_sigma=jnp.float32(0.1))

    jit_step = jax.jit(nes_es_step)
    grad, new_state = jit_step(state, eps, fitnesses)

    assert grad["theta"].shape == (3, 5)
    assert new_state.log_sigma.shape == ()


def test_adaptation_sampling_is_jit_compatible():
    """Same JIT-safety guarantee for the η_σ adaptation update."""
    from environments.nes import adaptation_sampling_update

    half_pop = 4
    eps = {"theta": jr.normal(jr.PRNGKey(0), (half_pop, 3, 5))}
    fitnesses = jr.normal(jr.PRNGKey(1), (2 * half_pop,))

    jit_update = jax.jit(adaptation_sampling_update)
    new_eta = jit_update(
        eps,
        fitnesses,
        jnp.float32(0.01),
        jnp.float32(0.01),
    )
    assert new_eta.shape == ()


def test_es_step_freezes_sigma_when_eta_zero():
    """Spec: η_σ = 0 disables the natural-gradient σ update — log_sigma
    must be byte-identical to the input, regardless of fitness."""
    half_pop = 8
    eps = {"theta": jr.normal(jr.PRNGKey(0), (half_pop, 5))}
    fitnesses = jr.normal(jr.PRNGKey(1), (2 * half_pop,))
    state = ESState(log_sigma=jnp.log(jnp.float32(0.1)), eta_sigma=jnp.float32(0.0))

    _grad, new_state = nes_es_step(state, eps, fitnesses)

    assert new_state.log_sigma == state.log_sigma
    assert new_state.eta_sigma == state.eta_sigma


def test_es_step_grows_sigma_when_large_norms_win():
    """When low-fitness samples have larger ‖ε‖² (search distribution too
    narrow), the natural-gradient update should INCREASE log_sigma."""
    half_pop = 64
    eps = {"theta": jr.normal(jr.PRNGKey(0), (half_pop, 10))}
    sq = jnp.sum(eps["theta"] ** 2, axis=1)
    fitnesses = jnp.stack([-sq, -sq], axis=1).reshape(-1)
    state = ESState(log_sigma=jnp.float32(0.0), eta_sigma=jnp.float32(0.1))

    _grad, new_state = nes_es_step(state, eps, fitnesses)

    assert new_state.log_sigma > state.log_sigma


# ---------------------------------------------------------------------------
# Adaptation sampling (Wierstra et al. 2014, §6.2 / xNES Algorithm 7)
# ---------------------------------------------------------------------------


def test_adaptation_sampling_increases_eta_when_sigma_too_small():
    """Setup: fitness sharply rewards large-‖ε‖² samples → hypothetical σ' = cσ
    would beat the current σ. The weighted Mann-Whitney comparison should
    detect this and grow η_σ above its baseline."""
    from environments.nes import adaptation_sampling_update

    half_pop = 64
    eps = {"theta": jr.normal(jr.PRNGKey(0), (half_pop, 10))}
    sq = jnp.sum(eps["theta"] ** 2, axis=1)
    fitnesses = jnp.stack([-sq, -sq], axis=1).reshape(-1)

    eta0 = jnp.float32(0.01)
    new_eta = adaptation_sampling_update(
        eps=eps,
        fitnesses=fitnesses,
        eta_sigma=eta0,
        eta_sigma_init=eta0,
        c=1.5,
        rho=0.5,
        step=0.1,
        eta_sigma_max=jnp.float32(1.0),
    )

    assert new_eta > eta0


def test_adaptation_sampling_decays_eta_when_no_signal():
    """When fitnesses are independent of ‖ε‖², no σ-direction is preferred,
    so η_σ should decay toward its init baseline rather than grow."""
    from environments.nes import adaptation_sampling_update

    half_pop = 64
    eps = {"theta": jr.normal(jr.PRNGKey(0), (half_pop, 10))}
    fitnesses = jr.normal(jr.PRNGKey(1), (2 * half_pop,))

    eta0_init = jnp.float32(0.01)
    eta_current = jnp.float32(0.05)
    new_eta = adaptation_sampling_update(
        eps=eps,
        fitnesses=fitnesses,
        eta_sigma=eta_current,
        eta_sigma_init=eta0_init,
        c=1.5,
        rho=0.5,
        step=0.1,
        eta_sigma_max=jnp.float32(1.0),
    )

    assert new_eta < eta_current
    assert new_eta >= eta0_init - 1e-6


def test_adaptation_sampling_respects_eta_max():
    """Growth is clamped to η_σ_max."""
    from environments.nes import adaptation_sampling_update

    half_pop = 64
    eps = {"theta": jr.normal(jr.PRNGKey(0), (half_pop, 10))}
    sq = jnp.sum(eps["theta"] ** 2, axis=1)
    fitnesses = jnp.stack([-sq, -sq], axis=1).reshape(-1)

    eta_max = jnp.float32(0.011)
    new_eta = adaptation_sampling_update(
        eps=eps,
        fitnesses=fitnesses,
        eta_sigma=jnp.float32(0.01),
        eta_sigma_init=jnp.float32(0.01),
        c=1.5,
        rho=0.5,
        step=0.5,
        eta_sigma_max=eta_max,
    )

    assert new_eta <= eta_max + 1e-6


# ---------------------------------------------------------------------------
# Sanity: ES estimator ≈ analytic gradient (no acceleration enabled)
# ---------------------------------------------------------------------------


def test_es_gradient_estimator_matches_analytic_on_quadratic():
    """In the high-N, small-σ regime, the antithetic ES estimator built by
    ``nes_es_step`` should point in the same direction as the true analytic
    gradient on a smooth objective. This is the "no acceleration" baseline
    sanity check: η_σ = 0 means σ is held fixed and the only thing exercised
    is the mean-parameter gradient estimator.
    """
    d = 20
    half_pop = 1024
    sigma = jnp.float32(0.01)
    theta = jr.normal(jr.PRNGKey(0), (d,))

    def f(x):
        return jnp.sum(x**2)

    true_grad = 2.0 * theta

    eps = jr.normal(jr.PRNGKey(1), (half_pop, d))
    f_pos = jax.vmap(lambda e: f(theta + sigma * e))(eps)
    f_neg = jax.vmap(lambda e: f(theta - sigma * e))(eps)
    fitnesses = jnp.stack([f_pos, f_neg], axis=1).reshape(-1)

    state = ESState(log_sigma=jnp.log(sigma), eta_sigma=jnp.float32(0.0))
    grad_est, new_state = nes_es_step(state, {"theta": eps}, fitnesses)

    g = grad_est["theta"]
    cos = jnp.dot(g, true_grad) / (jnp.linalg.norm(g) * jnp.linalg.norm(true_grad))
    assert cos > 0.95
    # η_σ = 0 must leave σ frozen even in this realistic-scale setting.
    assert new_state.log_sigma == state.log_sigma
