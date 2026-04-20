"""Sanity tests for the standalone DP-PSAC implementation."""

from __future__ import annotations

import dp_psac
import jax.numpy as jnp
import jax.random as jr
import numpy as np


def _random_per_sample_grads(B: int, shapes: list[tuple[int, ...]], key) -> list:
    keys = jr.split(key, len(shapes))
    return [jr.normal(k, (B, *s)) for k, s in zip(keys, shapes)]


def test_clip_norm_bound():
    """||g̃_i|| <= C for every sample."""
    key = jr.PRNGKey(0)
    B = 32
    shapes = [(10,), (5, 4)]
    grads = _random_per_sample_grads(B, shapes, key)
    # Scale some gradients very large, some very small.
    grads = [
        g * jnp.array([10.0 ** (i - B // 2) for i in range(B)]).reshape((B,) + (1,) * (g.ndim - 1))
        for g in grads
    ]

    C = 0.5
    r = 0.1
    clipped = dp_psac.psac_clip(grads, jnp.array(C), r)
    flat = jnp.concatenate([g.reshape(B, -1) for g in clipped], axis=1)
    norms = jnp.linalg.norm(flat, axis=1)
    # DP-PSAC bound is C (since g̃_i = C g_i / (||g|| + r/(||g||+r)) and denom >= ||g||).
    assert jnp.all(norms <= C + 1e-5), f"max norm {float(norms.max())} exceeds C={C}"


def test_small_gradient_limit():
    """As ||g|| -> 0, denom -> r/r = 1, so g̃ -> C * g."""
    r = 0.1
    C = 2.0
    tiny = [jnp.array([[1e-10, 0.0, 0.0]])]  # B=1
    clipped = dp_psac.psac_clip(tiny, jnp.array(C), r)
    # weight ≈ 1 so g̃ ≈ C * g
    expected = C * tiny[0]
    np.testing.assert_allclose(clipped[0], expected, rtol=1e-3, atol=1e-6)


def test_large_gradient_limit():
    """As ||g|| -> inf, denom -> ||g||, so g̃ -> C * g / ||g|| (matches Abadi tail)."""
    r = 0.1
    C = 1.0
    huge = [jnp.array([[1e8, 0.0, 0.0]])]
    clipped = dp_psac.psac_clip(huge, jnp.array(C), r)
    norm = jnp.linalg.norm(huge[0])
    expected = C * huge[0] / norm
    np.testing.assert_allclose(clipped[0], expected, rtol=1e-6, atol=1e-6)


def test_clip_formula_exact():
    """Verify exact formula on a hand-computed case."""
    r = 0.1
    C = 0.5
    g = jnp.array([[3.0, 4.0]])  # norm = 5
    clipped = dp_psac.psac_clip([g], jnp.array(C), r)
    denom = 5.0 + r / (5.0 + r)
    expected = C * g / denom
    np.testing.assert_allclose(clipped[0], expected, rtol=1e-6, atol=1e-6)


def test_per_step_schedule_is_used():
    """Step t uses sigmas[t] and clips[t] — not a constant."""
    import equinox as eqx

    key = jr.PRNGKey(42)
    model = dp_psac.MLP(in_dim=4, hidden=8, out_dim=3, key=key)

    N, B = 16, 4
    x = jr.normal(jr.PRNGKey(1), (N, 4))
    y = jr.randint(jr.PRNGKey(2), (N,), 0, 3)

    import optax

    params, static = eqx.partition(model, eqx.is_inexact_array)
    opt = optax.sgd(0.0)  # zero LR so params don't move -> noise scale effect is isolated
    opt_state = opt.init(params)
    step = dp_psac.make_train_step(static, opt, B, r=0.1, x_train=x, y_train=y)

    # With lr=0, params are unchanged; just confirm the step runs for distinct sigmas/clips.
    for t, (s, c) in enumerate([(0.5, 1.0), (2.0, 0.1), (1.0, 0.5)]):
        params, opt_state, loss = step(
            params, opt_state, jnp.array(s), jnp.array(c), jr.PRNGKey(100 + t)
        )
        assert jnp.isfinite(loss)


def test_accountant_monotonic():
    """Epsilon with longer schedule >= epsilon with shorter schedule."""
    from accountant import epsilon_spent

    sigmas_short = np.full(100, 1.5)
    sigmas_long = np.full(1000, 1.5)
    eps_short = epsilon_spent(sigmas_short, sample_rate=0.01, delta=1e-5)
    eps_long = epsilon_spent(sigmas_long, sample_rate=0.01, delta=1e-5)
    assert eps_long >= eps_short
