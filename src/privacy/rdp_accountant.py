"""From-scratch Renyi-DP accountant for the schedule outer loop (ADR-0007).

The sampler is **fixed-size without replacement** under **replace-one**
(substitution) adjacency, so the per-step Gaussian sensitivity is ``2C`` (drop
one contribution of norm <= C, add another). Under the decoupled DP-PSAC
schedule the noise scale is ``C * sigma_mult``, so ``C`` cancels from the
privacy cost and the base (un-subsampled) Gaussian RDP is a function of the
noise weight ``w = 1 / sigma_mult`` only:

    func(alpha) = alpha / (2 * s**2)  with  s = sigma_mult / 2 = 1 / (2w)
                = 2 * alpha * w**2.

Subsampling amplification is the without-replacement bound of Wang-Balle-
Kasiviswanathan (Theorem 9 of arXiv:1808.00087), which for a base mechanism with
``func(inf) = inf`` (the Gaussian) reduces to a closed-form log-sum-exp over
integer moment orders. The order ``alpha`` is a *static* Python int (the binding
order ``alpha*`` is selected between outer steps), so the moment sum unrolls at
trace time and the binomial coefficients are compile-time constants.
"""

import math

import jax.nn
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array


def base_gaussian_rdp(alpha: int, w: Array) -> Array:
    """Un-subsampled Gaussian RDP ``func(alpha) = 2 * alpha * w**2`` (2C sensitivity)."""
    return 2.0 * alpha * w**2


def subsampled_gaussian_rdp_step(alpha: int, w: Array, p: float) -> Array:
    """Amplified per-step RDP at integer order ``alpha`` for one DP-SGD step.

    Fixed-size-WOR + replace-one subsampled Gaussian, closed form (WBK Thm 9).

    Args:
        alpha: Integer Renyi order (>= 2), static.
        w: Noise weight(s) ``1 / sigma_mult``; scalar or array. Output matches shape.
        p: Subsampling ratio ``m / n``.

    Returns:
        The per-step RDP value ``rho_step(alpha; w)``, same shape as ``w``.
    """
    if alpha < 2:
        raise ValueError(f"alpha must be an integer >= 2, got {alpha}")

    w = jnp.asarray(w)
    log_p = math.log(p)

    def func(a: int) -> Array:
        return base_gaussian_rdp(a, w)

    def cgf(x: int) -> Array:
        # cgf(x) = x * func(x + 1)
        return x * func(x + 1)

    # moments_two: the j = 2 term. func(inf) = inf collapses the second branch's
    #   min(log2, 2*(eps_inf + log(1 - exp(-eps_inf)))) to log2.
    func2 = func(2)
    log_1m_exp_func2 = jnp.log(-jnp.expm1(-func2))  # log(1 - exp(-func2)), stable
    moments_two = (
        2.0 * log_p
        + math.log(math.comb(alpha, 2))
        + jnp.minimum(
            math.log(4.0) + func2 + log_1m_exp_func2,
            func2 + math.log(2.0),
        )
    )

    # moment_bound(j) for j = 3..alpha; the eps_inf = inf case fixes the leading
    # min(j*(eps_inf + log(1 - exp(-eps_inf))), log2) to log2.
    def moment_bound(j: int) -> Array:
        return math.log(2.0) + cgf(j - 1) + j * log_p + math.log(math.comb(alpha, j))

    # log(1 + sum_j exp(term_j)) with the "1" folded into softplus so the
    # small-value regime uses a log1p path (plain log(1 + tiny) loses float32
    # precision) while large sums stay overflow-safe.
    terms = [moments_two] + [moment_bound(j) for j in range(3, alpha + 1)]
    log_moment_sum = jax.nn.softplus(logsumexp(jnp.stack(terms, axis=0), axis=0))

    # Un-amplified cap: (alpha - 1) * func(alpha).
    cgf_bound = jnp.minimum((alpha - 1) * func(alpha), log_moment_sum)
    return cgf_bound / (alpha - 1)


def rdp_total(alpha: int, weights: Array, p: float) -> Array:
    """Total RDP at integer order ``alpha`` for a schedule of noise weights.

    RDP composes additively across the T independent DP-SGD steps:
    ``rho_total(alpha) = sum_i rho_step(alpha; w_i)``.

    Args:
        alpha: Integer Renyi order (>= 2), static.
        weights: Per-step noise weights ``1 / sigma_mult``, shape (T,).
        p: Subsampling ratio ``m / n``.

    Returns:
        Scalar total RDP ``rho_total(alpha)``.
    """
    return jnp.sum(subsampled_gaussian_rdp_step(alpha, weights, p))


def rdp_to_epsilon(alphas: list[int], weights: Array, p: float, delta: float) -> Array:
    """Convert the schedule's RDP curve to ``eps(delta)`` via the tightest order.

    ``eps(delta) = min_alpha [ rho_total(alpha) + log(1 / delta) / (alpha - 1) ]``
    over the candidate integer orders.

    Args:
        alphas: Candidate integer Renyi orders (each >= 2), static.
        weights: Per-step noise weights ``1 / sigma_mult``, shape (T,).
        p: Subsampling ratio ``m / n``.
        delta: Target delta.

    Returns:
        Scalar ``eps(delta)``.
    """
    log_inv_delta = math.log(1.0 / delta)
    eps_per_alpha = jnp.stack([rdp_total(a, weights, p) + log_inv_delta / (a - 1) for a in alphas])
    return jnp.min(eps_per_alpha)


def select_optimal_alpha(alphas: list[int], weights: Array, p: float, delta: float) -> int:
    """Pick the binding integer order ``alpha*`` for the current schedule.

    ``alpha* = argmin_alpha [ rho_total(alpha) + log(1 / delta) / (alpha - 1) ]``.
    Called between outer steps (in Python) so the projection can then use a single
    fixed order, keeping the feasible surface smooth (the ``min_alpha`` conversion
    has kinks where the binding order switches). Returns a plain Python ``int``.

    Args:
        alphas: Candidate integer Renyi orders (each >= 2), static.
        weights: Per-step noise weights ``1 / sigma_mult``, shape (T,).
        p: Subsampling ratio ``m / n``.
        delta: Target delta.

    Returns:
        The candidate order minimising ``eps(delta)``.
    """
    log_inv_delta = math.log(1.0 / delta)
    eps_per_alpha = [float(rdp_total(a, weights, p) + log_inv_delta / (a - 1)) for a in alphas]
    return alphas[int(min(range(len(alphas)), key=lambda i: eps_per_alpha[i]))]
