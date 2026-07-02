---
status: accepted
---

# Privacy accounting model: fixed-α RDP over fixed-size-WOR sampling under replace-one adjacency

We are replacing the Gaussian-DP (GDP) accountant with Rényi DP because the GDP
path applies **Poisson-subsampling** amplification math (the Bu–Dong–Long–Su
central-limit formula) to a sampler that is actually **fixed-size without
replacement** (`approx_max_k` in `environments/dp.py`), and additionally reports
a CLT approximation rather than a real bound. The new accountant is written from
scratch in JAX (the pre-existing `privacy/rdp_privacy.py` binomial code is a
year old, unvalidated, and assumes Poisson — it is not trusted).

## Decisions

- **Accountant: RDP.** The privacy functional is separable across the T steps
  (`ρ_total(α) = Σ_i ρ_step(α; σ_i, C_i)`), which is what keeps the schedule
  projection cheap and gives a closed-form constraint gradient ∇g (the
  Riemannian projection work, likely a follow-up ADR).
- **One integer order α per projection; α\* adaptive between steps.** Each
  projection is onto a *single* order's constraint so the feasible surface is
  smooth (the true `ε(δ)=min_α[…]` conversion has kinks where the binding order
  switches, which would break the manifold structure the projection relies on).
  Between outer steps we recompute the binding `α* = argmin` over a small
  integer candidate set. Integer orders keep ρ_step and its derivatives
  closed-form. Fractional orders are deferred behind a measurement: only adopt
  them if the binding order is empirically found to sit between integers, at the
  cost of a numerical ρ-derivative at that order.
- **Sampler stays fixed-size without replacement.** True Poisson would require
  either a full-dataset masked pass (O(N)/inner-step, prohibitive at T≈3000
  inside the outer loop) or variable batch shapes (JIT-hostile under
  `vmap`/`scan`). A capped/truncated-Poisson buffer was considered and rejected
  as added complexity + overflow bias for no offsetting benefit.
- **Replace-one (substitution) adjacency ⇒ per-step sensitivity 2C.** This is
  the natural neighbouring relation for a fixed-size sampler (batch size is
  invariant under it) and matches the substitution-based WOR amplification
  literature. Sensitivity is 2C (drop one ‖·‖≤C contribution, add another), so
  per-step Gaussian μ-analog is 2C/σ, not C/σ.
- **Amplification: without-replacement (Wang–Balle–Kasiviswanathan 2019;
  Balle–Barthe–Gaboardi 2018), not the Poisson binomial.**
- **Validate against `autodp`** (Yu-Xiang Wang) at the actual
  (σ-schedule, C-schedule, m/n, T). `dp_accounting` and Opacus default to
  Poisson and cannot check the WOR bound.
- **PLD is post-hoc reporting only**, never a training-loop constraint — its
  budget is a non-separable FFT convolution with no cheap ∇g to project onto.

## Consequences

- Reported ε roughly doubles versus the old add/remove GDP number. This is a
  **correction, not a regression**: 2C is the true sensitivity for this sampler;
  the old number understated the real cost.
- **Comparability is preserved where it matters.** All project comparisons are
  internal (learned vs constant schedule, across the arch ladders and T-sweep,
  same accountant on both arms), so the factor-of-2 rescales both arms
  identically and cancels in the reported deltas. Old GDP ε numbers are *not*
  carried forward — everything is re-baselined under this model. For any
  external (literature, Poisson add/remove) comparison, convert via group
  privacy (~factor 2) or re-run the baseline under this accountant; nominal ε's
  across adjacency conventions are never directly comparable.
