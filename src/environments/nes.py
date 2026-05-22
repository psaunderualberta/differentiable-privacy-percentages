"""Natural Evolution Strategies (NES) gradient estimators.

Pure-math building blocks for the ES outer loop. Kept separate from
``environments.outer_loop`` so the formulas can be exercised without
spinning up the DP-SGD inner loop.

References: Wierstra et al., *Natural Evolution Strategies*, JMLR 2014.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree


def _nes_log_utilities(fitnesses: Array) -> Array:
    """NES log-utility rank shaping (Wierstra et al.).

    For minimisation: lower fitness → higher utility. Returns utilities
    summing to zero, suitable as weights for the OpenAI-ES gradient sum.
    """
    n = fitnesses.shape[0]
    # Rank: lowest-loss sample gets rank 0 (highest utility).
    order = jnp.argsort(fitnesses)
    ranks = jnp.empty(n, dtype=jnp.int32).at[order].set(jnp.arange(n))
    raw = jnp.maximum(0.0, jnp.log(n / 2 + 1.0) - jnp.log(ranks.astype(jnp.float32) + 1.0))
    return raw / jnp.sum(raw) - 1.0 / n


class ESState(eqx.Module):
    """Per-run NES state threaded across outer-loop steps.

    ``log_sigma`` is the natural-gradient-updated log search radius;
    ``eta_sigma`` is its learning rate (mutated online by adaptation
    sampling when enabled, otherwise held fixed).
    """

    log_sigma: Array
    eta_sigma: Array


def _total_dimension(eps: PyTree) -> int:
    """Total parameter dimension d, summed over all leaves (excludes the
    leading population/pair axis)."""
    return sum(math.prod(leaf.shape[1:]) for leaf in jax.tree.leaves(eps))


def _pair_sq_norms(eps: PyTree) -> Array:
    """For each antithetic pair index p, sum-of-squares of the standardized
    perturbation across every leaf. Shape: (half_pop,)."""
    leaves = jax.tree.leaves(eps)
    return sum(jnp.sum(leaf.reshape(leaf.shape[0], -1) ** 2, axis=1) for leaf in leaves)


def nes_log_sigma_gradient(eps: PyTree, u_pos: Array, u_neg: Array) -> Array:
    """Natural gradient w.r.t. ``log σ`` for an isotropic Gaussian search
    distribution with a single scalar σ (sNES, Wierstra et al. 2014).

    For antithetic pairs the ``+ε`` and ``-ε`` samples share ``‖ε‖²``, so the
    pair contribution collapses to ``(u_pos + u_neg) · (‖ε‖² − d)``::

        ∇_{log σ} J ≈ (1/(2d)) · Σ_pair (u_pos + u_neg) · (‖ε‖² − d)

    Sign is the *maximisation* convention: positive → increase σ.
    """
    d = _total_dimension(eps)
    sq_norms = _pair_sq_norms(eps)
    return jnp.sum((u_pos + u_neg) * (sq_norms - d)) / (2.0 * d)


def nes_es_step(
    state: ESState,
    eps: PyTree,
    fitnesses: Array,
) -> tuple[PyTree, ESState]:
    """One NES outer step.

    Args:
        state: current ESState carrying ``log_sigma`` and ``eta_sigma``.
        eps: unit-std perturbations, leading axis = half-pop (one per
            antithetic pair); each leaf is broadcast across +ε / −ε.
        fitnesses: full-population fitness vector laid out
            ``[pos_0, neg_0, pos_1, neg_1, ...]`` (lower = better).

    Returns:
        ``(grad_mean, new_state)``. ``grad_mean`` is the
        minimisation-convention gradient pytree (matches ``eps`` structure
        without the leading axis), suitable for ``optax.sgd`` which does
        ``params -= lr · grad``. ``new_state`` advances ``log_sigma`` by
        ``η_σ · ∇_{log σ} J`` (NES maximisation update on ``J = −L``).
    """
    sigma = jnp.exp(state.log_sigma)
    n = fitnesses.shape[0]
    u = _nes_log_utilities(fitnesses)
    u_pos = u[0::2]
    u_neg = u[1::2]

    # Mean-parameter gradient: matches the existing _make_es_training_loss_fn
    # convention so the analytic / ES interchange remains a drop-in.
    w_mean = (
        sigma * (u_neg - u_pos) / n
    )  # using optax.sgd, multiplying by sigma^2 to find correct update
    grad_mean = jax.tree.map(lambda e: jnp.tensordot(w_mean, e, axes=1), eps)

    g_log_sigma = nes_log_sigma_gradient(eps, u_pos, u_neg)
    new_log_sigma = state.log_sigma + state.eta_sigma * g_log_sigma
    return grad_mean, ESState(log_sigma=new_log_sigma, eta_sigma=state.eta_sigma)


def adaptation_sampling_update(
    eps: PyTree,
    fitnesses: Array,
    eta_sigma: Array,
    eta_sigma_init: Array,
    c: float = 1.5,
    rho: float = 0.5,
    step: float = 0.1,
    eta_sigma_max: Array | float = 1.0,
) -> Array:
    """Wierstra et al. (2014) §6.2 adaptation sampling for ``η_σ``.

    Importance-reweight the *current* population under the hypothetical
    search distribution ``N(0, (cσ)² I)`` and compare to the unweighted
    distribution via a weighted Mann–Whitney U statistic. If the
    hypothetical (larger) σ would beat the current one, ``η_σ`` grows
    (clamped at ``eta_sigma_max``); otherwise it decays toward
    ``eta_sigma_init``.

    Args:
        eps: unit-std perturbations, leading axis = half-pop (antithetic).
        fitnesses: full-population fitness vector
            ``[pos_0, neg_0, pos_1, neg_1, ...]`` (lower = better).
        eta_sigma: current σ learning rate.
        eta_sigma_init: baseline rate to decay toward in the "no signal"
            (and "shrink") cases.
        c: hypothetical-σ multiplier (>1).
        rho: U-statistic threshold above which η_σ grows.
        step: multiplicative step size for the η_σ update.
        eta_sigma_max: upper clamp on ``η_σ``.

    Notes on the importance weights: since each antithetic pair shares
    ``‖ε‖²``, the log importance weight collapses to
    ``-d log c + ‖ε‖² · (1 − 1/c²) / 2`` and is shared between ``+ε`` and
    ``-ε``. We use ``logsumexp`` to keep the weighted-mean rank stable for
    large ``d``.
    """
    d = _total_dimension(eps)
    sq_norms = _pair_sq_norms(eps)  # (half_pop,)

    log_w_pair = -d * jnp.log(c) + sq_norms * (1.0 - 1.0 / c**2) / 2.0
    log_w = jnp.repeat(log_w_pair, 2)  # (population_size,)

    # Ranks: 0 = lowest fitness (best). Use argsort-of-argsort for ties-
    # arbitrary but consistent ordering.
    order = jnp.argsort(fitnesses)
    n = fitnesses.shape[0]
    ranks = jnp.empty(n, dtype=jnp.int32).at[order].set(jnp.arange(n))

    # Weighted "probability of being a good sample" — Mann-Whitney U mass
    # on the lower-fitness side. U_hyp = Σ_i w_i · (n - 1 - r_i) / ((n - 1) · W)
    # so U_hyp ∈ [0, 1]; 0.5 = no preference, > 0.5 = hypothetical wins.
    log_w_norm = log_w - jax.scipy.special.logsumexp(log_w)
    w_norm = jnp.exp(log_w_norm)
    goodness = (n - 1 - ranks).astype(jnp.float32) / (n - 1)
    u_hyp = jnp.sum(w_norm * goodness)

    grow = (1.0 + step) * eta_sigma
    decay = (1.0 - step) * eta_sigma + step * eta_sigma_init
    new_eta = jnp.where(u_hyp > rho, grow, decay)
    return jnp.minimum(new_eta, jnp.asarray(eta_sigma_max))
