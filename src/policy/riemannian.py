"""Riemannian tangent-projection gradient transform for the schedule outer loop.

The schedule outer loop is optimization on the RDP budget manifold
``M = {g(θ) = 0}`` with ``g = constraint_value`` (ADR-0008). This custom
``optax.GradientTransformation`` removes the component of the gradient along the
manifold normal ``n = ∇_θ g(θ)``, yielding the Riemannian (tangent) gradient

    ξ = v − ⟨n, v⟩ / ⟨n, n⟩ · n,

so fixed points satisfy ``∇L ∥ ∇g`` (KKT) rather than the biased ``∇L ∥ ray`` of
a plain scaling projection. The schedule's ``project()`` retraction returns to
the manifold afterwards.

Per the no-momentum design (momentum lives only in the inner DP-SGD loop), the
transform is **stateless** — no momentum buffer, no projection vector transport.
The normal is nonzero only on the noise leaves, so the decoupled clip leaves
(free Euclidean params) pass through untouched. Composes in the existing chain
as ``chain(clip_by_global_norm, zero_nans, riemannian_tangent_projection,
scale_by_learning_rate)``.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import PyTree


def _tree_vdot(a: PyTree, b: PyTree) -> jnp.ndarray:
    """Sum of elementwise inner products over matching inexact-array leaves."""
    la = jax.tree.leaves(eqx.filter(a, eqx.is_inexact_array))
    lb = jax.tree.leaves(eqx.filter(b, eqx.is_inexact_array))
    return sum(jnp.vdot(x, y) for x, y in zip(la, lb))


def riemannian_tangent_projection() -> optax.GradientTransformation:
    """Stateless transform projecting the gradient onto the budget manifold's tangent space.

    ``params`` (the schedule) must expose ``constraint_value()``; the normal is
    ``∇_θ constraint_value`` by autodiff of the closed-form constraint.
    """

    def init_fn(params: PyTree) -> optax.EmptyState:
        del params
        return optax.EmptyState()

    def update_fn(updates: PyTree, state: optax.EmptyState, params: PyTree = None):
        if params is None:
            raise ValueError("riemannian_tangent_projection requires params (the schedule).")

        normal = eqx.filter_grad(lambda s: s.constraint_value())(params)
        nn = _tree_vdot(normal, normal)
        coeff = _tree_vdot(normal, updates) / jnp.where(nn > 0, nn, 1.0)
        tangent = jax.tree.map(
            lambda v, n: v - coeff * n,
            updates,
            normal,
        )
        return tangent, state

    return optax.GradientTransformation(init_fn, update_fn)
