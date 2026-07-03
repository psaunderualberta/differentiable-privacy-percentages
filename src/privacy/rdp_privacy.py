"""RDP privacy budget object for the schedule outer loop (ADR-0007/0008).

``RDPPrivacyParameters`` holds the target ``(eps, delta)``, the subsampling ratio
``p``, the step count ``T``, a small candidate set of integer Renyi orders, and
the currently-binding order ``alpha*`` (adaptive between outer steps, selected in
the non-JIT loop and stored as a *static* field so the accountant's moment sum
unrolls at trace time).

The learnable variable is the accountant's noise weight ``w = 1 / sigma_mult``.
The budget constraint at the binding order is

    g(w) = rho_total(alpha*; w) - c(alpha*),   c(alpha*) = eps - log(1/delta)/(alpha* - 1),

so ``g = 0`` is the on-budget manifold. ``c(alpha*)`` is the RDP budget that, at
order ``alpha*``, converts to exactly ``eps`` at ``delta`` (ADR-0008).
"""

import math

import equinox as eqx
import jax.lax as jlax
import jax.numpy as jnp
from jaxtyping import Array

from privacy.rdp_accountant import rdp_to_epsilon, rdp_total, select_optimal_alpha

DEFAULT_ALPHAS: tuple[int, ...] = (2, 3, 4, 6, 8, 16, 32, 64)


class RDPPrivacyParameters(eqx.Module):
    eps: float = eqx.field(static=True)
    delta: float = eqx.field(static=True)
    p: float = eqx.field(static=True)
    T: int = eqx.field(static=True)
    alphas: tuple[int, ...] = eqx.field(static=True)
    alpha_star: int = eqx.field(static=True)

    def __init__(
        self,
        eps: float,
        delta: float,
        p: float,
        T: int,
        alphas: tuple[int, ...] = DEFAULT_ALPHAS,
        alpha_star: int | None = None,
    ):
        self.eps = eps
        self.delta = delta
        self.p = p
        self.T = T
        self.alphas = tuple(alphas)
        self.alpha_star = self.alphas[0] if alpha_star is None else alpha_star

    def with_alpha_star(self, weights: Array) -> "RDPPrivacyParameters":
        """Return a copy with ``alpha*`` re-selected as the binding order for ``weights``.

        Called in the non-JIT outer loop between steps (``select_optimal_alpha``
        returns a Python int), so the following projection uses a single fixed
        order and the feasible surface stays smooth (ADR-0007).
        """
        astar = select_optimal_alpha(list(self.alphas), weights, self.p, self.delta)
        return RDPPrivacyParameters(
            self.eps, self.delta, self.p, self.T, alphas=self.alphas, alpha_star=astar
        )

    def rho_budget(self) -> float:
        """RDP budget ``c(alpha*) = eps - log(1/delta)/(alpha* - 1)`` at the binding order."""
        return self.eps - math.log(1.0 / self.delta) / (self.alpha_star - 1)

    @eqx.filter_jit
    def project_scale(self, sigma_mults: Array) -> Array:
        """Equality scaling retraction: scale ``sigma_mults`` by a common factor
        ``s`` so the schedule lands exactly on the budget manifold ``g = 0``.

        The retraction targets ``eps`` itself via the full min-over-alpha
        conversion (``rdp_to_epsilon``): it is never differentiated, so the
        conversion's kinks are harmless, and it is always feasible by scaling
        (unlike a single fixed order, whose budget ``c(alpha)`` can be negative
        off-manifold). ``eps(delta)`` is monotone decreasing in ``s`` (more noise
        ⇒ smaller eps), so the on-budget scale is found by bisection. Scaling
        ``sigma_mults`` (which the BSpline base schedule represents linearly) by a
        scalar is exact in the BSpline family (ADR-0008).
        """
        sigma_mults = jnp.asarray(sigma_mults)

        def residual(s: Array) -> Array:
            eps = rdp_to_epsilon(list(self.alphas), 1.0 / (s * sigma_mults), self.p, self.delta)
            return eps - self.eps

        # residual decreasing in s; bracket [lo, hi] with residual(lo) > 0 > residual(hi).
        lo = jnp.asarray(1e-4)
        hi = jnp.asarray(1e4)

        def body(_, lo_hi):
            lo, hi = lo_hi
            mid = 0.5 * (lo + hi)
            over = residual(mid) > 0  # still under-noised ⇒ need a larger scale
            return (jnp.where(over, mid, lo), jnp.where(over, hi, mid))

        lo, hi = jlax.fori_loop(0, 60, body, (lo, hi))
        return 0.5 * (lo + hi) * sigma_mults

    def constraint(self, weights: Array) -> Array:
        """Signed budget residual ``g(w) = rho_total(alpha*; w) - c(alpha*)``.

        Positive ⇒ over budget (too little noise). Zero ⇒ on the budget manifold.
        Differentiable w.r.t. ``weights`` (the manifold normal is its gradient).
        """
        return rdp_total(self.alpha_star, weights, self.p) - self.rho_budget()
