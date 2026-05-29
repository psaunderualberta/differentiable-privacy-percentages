# Assess symbolic-regression fit with position/scale-aware metrics and a privacy-budget validity check, not correlation

When evaluating how well a PySR equation matches the learned σ/C/μ schedules, we deliberately exclude correlation scalars (R², Pearson, Spearman) as primary metrics and instead headline **NRMSE-by-mean**, complemented by **RMSLE**, **RMSE**, **extremum location/value errors**, a **residual-vs-step diagnostic**, and a **privacy-budget validity check** (predicted privacy expenditure vs target GDP μ, plus projection distance).

## Why

Correlation is the obvious thing to reach for, so a future reader will wonder why it's absent. Two reasons make it actively misleading here:

1. **Position-blind.** Pearson and Spearman are permutation-invariant over the step index — they only see the multiset of (predicted, actual) pairs, so they cannot tell whether a non-monotonic schedule's peak (many learned schedules are inverted-U) landed at the right `t/T`. Extremum-location error and the residual-vs-step plot capture this; correlation cannot.
2. **Scale-blind.** The equations predict *absolute* σ and C, so a uniform 2× miss is a real error — but correlation would report ≈1 ("great shape") for it. NRMSE and RMSLE correctly penalise it.

Spearman additionally degenerates on the flat top of an inverted-U (rank noise). R² is kept but **caveated**: it goes degenerate for near-flat schedules (variance ≈ 0), so runs with `std/mean < 0.05` are flagged and excluded from summary R².

## Considered and rejected

- **Pooled R² as the headline.** Hides per-condition variance (mnist vs cifar) — the whole concern. Replaced by per-run metric *distributions*.
- **Correlation (Pearson/Spearman) as a shape metric.** Rejected per the two reasons above; explicit extremum-location/value errors do the shape job honestly.

## Consequence

The privacy-budget validity check requires *both* the σ and clip equations (μ is derived as C/σ to match what would be deployed); it is skipped when only one is present. A μ-only equation can be checked as an optional cross-check but is not the default, since downstream tasks need σ and C separately.
