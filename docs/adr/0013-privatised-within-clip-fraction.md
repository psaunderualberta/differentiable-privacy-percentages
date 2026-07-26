# Privatising the within-clip fraction in the adaptive-clip baseline

The `StatefulMedianGradient` baseline adapted its clip threshold from the **within-clip
fraction** `b̄` — the share of the batch whose per-example gradient norm fell at or below
`C_t` — and released that statistic in the clear. `b̄` is a function of the private data,
so the baseline was **not** the (ε,δ)-DP mechanism its reported budget claimed. We
privatise it as a **second vector group of the same per-step Gaussian mechanism**: the
count is released with its own noise, and its cost is carved out of the existing per-step
budget as `μ_count² + μ_grad² = μ₀²`, so the *total* ε is unchanged and the accountant is
untouched. Following Andrew et al. (NeurIPS 2021), the count noise is fixed at one
twentieth of the expected batch size rather than tuned.

## Status

accepted

## Why

- **The baseline was flattering itself.** It got adaptivity for free while every learned
  schedule paid full price for its own privacy. Any accuracy gap measured against it was
  partly an artifact of comparing a DP mechanism to a not-quite-DP one — the wrong
  direction for a thesis claim, since it makes the learned schedule look better than it is.
- **Joint mechanism, not composition.** The gradient sum and the count are computed on
  the *same* batch, so releasing them as two vector groups of one Gaussian mechanism
  (McMahan et al. 2018) costs `μ₀² = μ_grad² + μ_count²` — a Pythagorean split of one
  release. Composing them as two mechanisms would pay subsampling amplification twice for
  one look at the data, inflating the accounted ε for no reason.
- **The split happens at the un-amplified `μ₀`, which is the non-obvious part.** `μ₀` is
  the *per-step, pre-amplification* GDP parameter; `compute_mu_0` inverts
  `μ = p·√(T·(exp(μ₀²)−1))` to obtain it. Because both groups are released under the
  *same* Poisson draw, amplification applies once, to the joint release. Splitting the
  budget anywhere downstream of amplification — at ε, or at the total μ — would double-count
  the sampling benefit. Splitting at `μ₀` means `gdp_privacy.py` needs **no change at all**:
  the baseline simply calibrates its gradient noise to `μ_grad = √(μ₀² − μ_count²)` instead
  of `μ₀`.
- **`r = 20` is Andrew's number, and it is the regime-independent parameterisation.**
  They set the count noise to `σ_count = L/r`; since the release is divided by that same
  expected batch size `L`, the standard error of `b̄` is exactly `1/r = 0.05` in *every*
  privacy regime. The derived budget share `ρ = μ_count²/μ₀²` then ranges over
  `[2.9e-4, 1.3e-2]` across our (ε, T) grid at `L = 250` — at most a **0.7% increase in
  gradient noise**. Honest adaptivity turns out to be nearly free, which is itself the
  reportable result.
- **`b̄` is clamped to [0,1] and `C` is floored.** The clamp is post-processing of an
  already-private release, so it is free. The floor exists only because
  `postprocess_update` divides by `C_t`; with `r = 20` the noise on `b̄` is small enough
  that the clip random-walk should never approach it, so it is a numerical backstop, not
  part of the mechanism.

## Considered and rejected

- **Sweeping `ρ` directly** (the original plan). A single `ρ` value means a *different*
  `std(b̄)` in every regime, because `μ₀` varies by ~7× across the grid — so no one value
  is meaningful and it must be tuned per condition. Worse, `ρ` interacts with `c_0` and
  `eta_c` (all three govern how fast `C` moves), so adding it as a third dimension to
  `baseline_sweep`'s 20-draw random search would have left the baseline *less* well tuned
  than before — the opposite of the goal. Deriving it from `r` removes the dimension
  entirely. `ρ` survives as a logged diagnostic.
- **Separate composed mechanisms for gradient and count.** Conceptually simpler and
  modular, but pays amplification twice for one batch access, so the same statistical
  accuracy costs strictly more ε.
- **A `{0,1}` count encoding** (sensitivity 1) instead of Andrew's `±½` (sensitivity ½).
  The `±½` shift halves the sensitivity and therefore the noise, and works with the
  *expected* divisor `L` — which is public — so it needs no knowledge of the realised
  batch occupancy.
- **Leaving the baseline non-private and disclosing the caveat in prose.** Cheapest, but
  it makes the headline comparison unusable: every reader would have to mentally discount
  the gap by an unquantified amount.

## Consequences

- **Every cached baseline must be regenerated.** `restore_from_cache` keys purely on
  `run_id` with no version or config hash, so it would silently restore pre-privatisation
  median numbers and skip the sweep. The artifact name and local path are version-bumped
  so old caches become unreachable — the Constant and DynamicDPSGD sweeps are recomputed
  needlessly, which is the price of a cache that cannot fail open.
- **`inner_step` is reordered.** Step `t` now clips and counts against the *incoming*
  `C_t` and computes `C_{t+1}` afterwards, matching the paper's Algorithm 1. Previously
  the freshly-updated threshold was applied to the batch it was derived from, which both
  leaked in a second way and made `c_0` almost meaningless. This changes the numerics of
  the existing baseline independently of privatisation.
- **`update_state` gains a `key` parameter** on the abstract base (one implementer). The
  key is `jr.fold_in(noise_key, iter_t)`, keeping the gradient-noise stream bit-identical
  to before so the reordering is the only numeric change on that path.
- **Construction fails loudly when the budget cannot absorb the count.** `μ_count = 0.5·r/L`
  grows as `L` shrinks; at `L = 32` it exceeds `μ₀` outright and `μ_grad` would be `NaN`
  thousands of steps later. Construction raises if `ρ > 0.25`, reporting `L`, `r`, `μ₀`,
  `ρ` and the minimum viable `L`. This is a real constraint on transfer targets, which set
  their own batch size.
- **The baseline is renamed** to `"Adaptive Clip (Andrew et al.)"`. The old
  `"Clip to Median Gradient Norm"` implied a directly-estimated median, which is the exact
  misconception this change exists to correct — the schedule steers a *fraction* and the
  median is only its fixed point. `transfer_reference.py` keeps mapping it to the `Median`
  slug, so nothing moves on disk.
- Vocabulary is fixed in `CONTEXT.md` under "Adaptive clipping": within-clip fraction,
  quantile target (γ), count release, count noise ratio (r), median budget fraction (ρ).
