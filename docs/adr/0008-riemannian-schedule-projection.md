---
status: accepted
---

# Riemannian gradient + scaling retraction for the schedule outer loop

The schedule outer loop is projected gradient descent: `optimizer.update` →
`apply_updates` → `project()` (`main.py`), where `project()` is post-step and
never autodiff'd. Under the exact-projection accountant (ADR-0007) the
nearest-point projection is expensive and, if replaced by a cheap *scaling*
projection, biases the learned schedule shape (fixed points satisfy `∇L ∥ ray`,
not `∇L ∥ ∇g`, so KKT is missed). We therefore reframe the outer loop as
optimization on the budget manifold `M = {g = B}`: **tangent-project the gradient
(Riemannian gradient), then apply a cheap scaling retraction.** Fixed points then
satisfy `∇L ∥ ∇g` = KKT, recovered at scaling-retraction cost.

## Decisions

- **σ-only manifold via the decoupled (DP-PSAC) schedule.** The primary
  methodology is `DecoupledSigmaAndClipSchedule`: noise scale = `C·(1/w)`, so the
  clip `C` cancels from the per-step privacy cost (`μ_step = 2/σ_mult` under
  replace-one) and the constraint depends on the `w`/σ-side **only**. Clips are
  free utility (learning-rate-like) parameters optimized by plain Euclidean SGD;
  the manifold, tangent projection, and retraction touch the noise leaves only.
- **Manifold normal by autodiff of the constraint, not hand-derived.** The
  constraint `g(θ) = Σ_i ρ(α*; w_i(θ)) − c(α*)` is a cheap, closed-form,
  differentiable scalar. The normal is `n_θ = eqx.filter_grad(g)(schedule)`,
  which flows through the BSpline `softplus`+basis parameterization in one shot
  and lands in exactly the `es_filter`/`eqx.partition` differentiable leaf space
  — automatically aligning the momentum buffer partition. We never differentiate
  the *retraction*; only the constraint. KKT-in-θ (`∇_θL ∥ ∇_θg`) is the correct
  stationarity condition because θ (control points), not the raw w-vector, is
  what is optimized.
- **Equality scaling retraction; no interior/slack branch.** The budget always
  binds (spending more budget is always a utility win), so the retraction drives
  to the boundary via 1-D bisection on a common noise scale `s` — scaling σ_mult
  up when over-budget, down when under. Exact in the BSpline family
  (`s·w = basis @ (s·softplus(θ))`). Every step therefore begins on the boundary,
  the tangent projection is unconditionally valid, and the constraint is treated
  as equality (no KKT multiplier-sign logic, no `where g<B` degrade-to-Euclidean).
  **The retraction targets `ε` itself via the full `min_α` conversion
  (`rdp_to_epsilon`), not a single fixed order.** A single order's budget
  `c(α) = ε − log(1/δ)/(α−1)` can be *negative* off-manifold (unreachable by any
  scaling), so bisecting on the single-α residual has no root; `ε(δ)` is always
  reachable. The retraction is never differentiated, so the `min_α` kinks are
  harmless. The single fixed `α*` is needed only for the **tangent-projection
  normal**, evaluated at the on-budget point where `c(α*) = ρ_total(α*) ≥ 0` by
  construction — hence `refresh_alpha_star()` runs on the freshly-retracted
  schedule each step.
- **No outer-loop momentum ⇒ stateless transform, no vector transport.**
  Momentum lives only in the *inner* DP-SGD loop; the Riemannian (outer) descent
  is plain tangent-projected gradient descent. The transform is therefore
  **stateless** — it only removes the normal component
  (`ξ = v − ⟨n,v⟩/⟨n,n⟩·n`), with no momentum buffer and no projection vector
  transport. The outer `optax.sgd` momentum is forced to 0 on this path.
  *(Supersedes the earlier "fixed-momentum heavy-ball via projection vector
  transport" decision — vector transport is only needed to keep a momentum
  buffer on-manifold, and there is no outer momentum buffer.)*
- **Delivered as a custom `optax.GradientTransformation`, not inlined.** It
  composes with the existing `optax.chain(zero_nans, clip_by_global_norm, …)`
  (ordering: neutralize/clip first, then tangent-project + transport + scale by
  lr), is unit-testable in isolation against a hand-checked normal on a toy
  2-control-point BSpline, and needs no signature change (`main.py` already
  passes the schedule as `params`). The retraction stays on the schedule
  (`project()`), reworked to the RDP equality scaling retraction; the transform
  owns normal + tangent projection + momentum transport.
- **Analytic path first; ES deferred.** The ES path calls `project()` inside the
  black-box eval, so it estimates descent of `L ∘ retraction` — which
  reintroduces the biased-shape failure mode unless the ES estimator is itself
  made Riemannian (tangent-project the antithetic perturbations / the estimated
  gradient). The primary experiments are analytic, so ES-Riemannian is a
  separate later problem touching `outer_loop.py`.

## Consequences

- Only `project()` (retraction) and the new transform change; `dp.py` is
  untouched — the decoupled schedule already emits `C·σ`, so DP-PSAC needs no
  inner-loop rewrite.
- The design transfers to RDP-fixed-α unchanged (swap the per-step ρ in the
  constraint scalar); it does **not** transfer to PLD, whose non-separable
  budget has no cheap ∇g (PLD stays post-hoc reporting only, per ADR-0007).
