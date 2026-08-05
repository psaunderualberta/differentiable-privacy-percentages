# Curve transfer scopes its sources to the T-sweep axis at four seeds per regime-arm

ADR 0008 requires that **every** source policy be transferred, so that the transfer matrix
is read off rather than selected from. Applied to FirSweep that is 851 policies × 12 target
columns ≈ 21,000 GPU-hours, which does not fit the remaining schedule. Curve transfer is
therefore restricted to the **T-sweep axis**, to regimes carrying at least four seeds in an
arm, and to a **fixed four seeds per regime-arm** chosen by seed index — 49 regime-arms,
196 policies, 6 target columns, ≈ 1,650 GPU-hours. Both arms are kept, and the **arm
becomes part of the regime identity**.

## Status

accepted (scopes ADR 0008 for FirSweep)

## Why

**Keeping both arms requires putting the arm in the row schema first.** `_REGIME_COLUMNS`
is `(dataset, eps, T, arch_label)` and omits `optimizer`, so ADR 0011's move of the arm onto
the run silently made a "16-seed regime" eight `sgd-m0.9` runs plus eight `sgd-m0.0` ones.
ADR 0016 measured those arms' median σ differing by 8.5× at `mnist/eps=3/T=2000`, with peak
position moving from `t/T ≈ 0.45` to `≈ 0.70`. A regime pooling them reports arm separation
where it claims to report generalization consistency. `SourcePolicy` and `transfer_rows`
carried no arm field at all, so the two were also indistinguishable *on disk* — adding m0.0
sources without the schema change would have made the matrix less trustworthy, not more.
With the arm recorded and grouped on, the momentum contrast becomes a first-class row
dimension and the question "does momentum change what transfers?" is answerable.

**The seed cap is a subsample, not a selection.** ADR 0008's prohibition is specifically on
choosing a per-regime representative *by accuracy*, which would bias toward source-overfit
shapes. Taking the first four seed indices is independent of every accuracy number and
carries no such bias; it costs only precision in the spread estimate (~35% relative error
on a 4-sample standard deviation against ~25% at 8).

**The four-seed floor keeps spreads comparable.** The m0.0 arm is unevenly populated: 14 of
16 fashion-mnist regimes carry ≥4 seeds but only 3 of 16 mnist ones do, the other 13 having
no m0.0 runs at all. Admitting a regime-arm at n=2 would put a two-sample range in the same
heatmap, rendered identically to a four-sample spread, inviting exactly the comparison it
cannot support. The consequence is that the momentum contrast is a **paired comparison
within fashion-mnist across all 16 (ε, T) points**, plus three isolated mnist regimes — a
sufficient design that must be stated as such rather than presented as a half-empty grid.

**The axis scoping follows ADR 0016.** The arch axis exists only at the single point
`(eps=10, T=5000)` and is out of scope for every synthesis, so those 483 policies — 57% of
the original cost — could never gain an equation counterpart or appear in the overlay.

## Considered and rejected

- **Scope to `sgd-m0.9` alone** (128 policies, ~1,000 GPU-hours). Cheapest correct option
  and the arm with the only complete 8-seed grid, but it discards a real experimental
  contrast for a saving the schedule does not need.
- **All 8 seeds** (~3,300 GPU-hours across both arms). No deviation from ADR 0008 and
  tighter spread bars, but it consumes the entire remaining schedule with no slack for a
  failed batch.
- **Two seeds per regime-arm.** Rejected: at n=2 the reported figure is a range, not a
  consistency measure, and the chapter's headline claim leans on it.
- **Cheaper targets (T=2000) instead of fewer sources.** Rejected: halving the step count
  on eyepacs/imagenet/chexpert risks the accuracy-floor effect ADR 0007 names as the main
  threat to the whole comparison, and unlike a seed cap it degrades every cell at once.

## Consequences

- **`arm` joins the transfer row schema.** `SourcePolicy`, `transfer_rows` /
  `_TRANSFER_COLUMNS`, the assembler's regime grouping, and `_OVERLAY_KEYS` all gain it.
  Because syntheses are scoped to one arm (ADR 0016), the m0.0 rows correctly have no
  equation counterpart and simply do not overlay.
- **Generalization consistency is computed over source policies, not evaluation reps.**
  `transfer_plot.transfer_matrix` grouped on `source_id`, making the ± printed in each cell
  the standard deviation across one policy's `num_reps` evaluations — DP-SGD run-to-run
  noise, not the regime signal CONTEXT.md names. The assembler now pools by regime-arm
  (4 policies × 3 reps = 12 samples); the per-policy figure remains available as evaluation
  noise. This is a plot-side change only and re-runs nothing.
- `num_reps` drops from 8 to 3. It now only has to stabilise a cell mean, because the
  consistency estimate draws its samples from the regime's policies.
- The matrix has 49 rows rather than 62 regimes' worth, and no row at an architecture other
  than `cnn-16x32-head32`. Any architecture-transfer claim needs a later, separate run.
