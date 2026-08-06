# Keep syntheses arm-scoped on cost, not because the arms lack a shared shape

ADR 0016's arm scoping **stands**: a **synthesis** still covers one arm, and the four
FirSweep syntheses in flight are correct as submitted. Its *justification* is corrected.
On the complete T-sweep data the two arms **do** share a **universal schedule shape**
family in σ and clip; pooling them costs roughly **one extra shape degree of freedom**,
not a different shape. Scoping is retained because pooling is expensive and buys a claim
the thesis does not need — not because it is unsound.

## Status

accepted (amends the reasoning of ADR 0016; changes none of its decisions)

## Why

ADR 0016 asserts that "no single universal schedule shape covers both arms in σ, clip, or
μ". That sentence carries the whole rejection of pooling, and only the μ half of it was
ever measured. This ADR measures the σ and clip halves on the current data.

**Method.** Restrict to the T-sweep arch `cnn-16x32-head32` (32 **conditions**, both arms
present in all 32). Interpolate each run onto a 101-point `step_norm` grid, divide by its
own mean so scale is removed, and average over seeds — one normalised shape per
(condition, arm). Then ask how many shape degrees of freedom reproduce those shapes, via
PCA reconstruction with `k` components plus free per-condition coefficients.

That PCA is a *generous* stand-in for a template fit: free linear coefficients are easier
to fit than K constants entering a discovered `f` nonlinearly. It is generous to the
scoped and pooled cases equally, so the **comparison between columns** is what carries
weight here; the absolute error levels do not.

Relative RMS reconstruction error, `k` shape DOF + free per-condition coefficients:

| k | σ m0.0 | σ m0.9 | σ pooled | clip m0.0 | clip m0.9 | clip pooled |
|---|---|---|---|---|---|---|
| 0 | 0.042 | 0.092 | 0.102 | 0.042 | 0.094 | 0.108 |
| 1 | 0.031 | 0.037 | 0.042 | 0.023 | 0.033 | 0.041 |
| 2 | 0.020 | 0.017 | 0.022 | 0.005 | 0.015 | 0.018 |
| 3 | 0.011 | 0.008 | 0.016 | 0.003 | 0.009 | 0.011 |
| 4 | 0.004 | 0.005 | 0.009 | 0.002 | 0.005 | 0.007 |

**The arms share a shape family.** Pooled `k=3` (σ 0.016) matches per-arm `k=2` (0.020 /
0.017); pooled `k=4` (0.009) matches per-arm `k=3` (0.011 / 0.008). Clip behaves the same
way. One extra degree of freedom absorbs the arm — the pooled data is not bimodal in
shape, it is one family sampled at two scales. Direct cross-arm shape correlation agrees:
median 0.900 (σ) and 0.924 (clip). For calibration, the m0.9 arm's *own* condition-to-
condition shape correlation has median 0.896 — the arms differ from each other about as
much as m0.9's conditions differ among themselves, and the template already absorbs the
latter.

**But not by scale alone.** The peak of the normalised shape sits at `t/T ≈ 0.40` (σ) and
`≈ 0.48` (clip) at m0.0, against `≈ 0.34` and `≈ 0.36` at m0.9. A multiplicative factor
cannot move a peak. Collapsing both arms onto one shape plus a single global per-arm scale
leaves **14.5 %** (σ) and **15.8 %** (clip) relative RMS — an order of magnitude worse than
2–3 DOF achieve within an arm.

**The scale findings reproduce exactly.** Median σ ratio m0.9/m0.0 is 0.107 (≈9.3×, stable
across all 32 conditions at 0.097–0.141); median μ ratio is 0.999, as the privacy
projection requires. ADR 0016 is right that the arms differ ~an order of magnitude in σ
scale and not at all in μ scale.

**What does not reproduce is ADR 0016's μ *shape* claim.** It reports the μ peak at
`t/T ≈ 0.70` (m0.0) against `≈ 0.45` (m0.9) with shape correlation 0.65–0.88. On the
current data the μ peaks are 0.67 and 0.61 with correlation median 0.954 (min 0.634). The
m0.0 figure matches; the m0.9 figure and the correlation do not. The most likely cause is
that the measurement predates the m0.0 MNIST re-fetch that ADR 0016 itself lists as
outstanding — at the time that arm held 19 of 32 conditions and 20 MNIST runs against 111
Fashion-MNIST, so the comparison was made against a degraded and dataset-skewed arm. The
re-fetch has since landed (m0.0 now carries 128 MNIST and 111 Fashion-MNIST runs, 239
total against m0.9's 255).

**Scoping still stands, on cost.** Adding `optimizer` to `CONDITION_KEYS` takes the
condition count 32 → 64 and so doubles the per-candidate constant optimisation — the exact
cost ADR 0006 already flags as where template fits get slow. Against a fixed 2h45m search
budget per job, that halves the candidates explored per unit wall-clock, and a pooled fit
needs *more* search than a scoped one (one extra DOF, twice the constants). It is entirely
possible to support the pooled shape in the data and still fit it worse in practice. Set
against that, the gain is a strengthened **compression claim** the thesis does not require:
the reportable result is per-arm, and the arms' relationship is already established
separately as a scale finding.

## Considered and rejected

- **Add `"optimizer"` to `CONDITION_KEYS` now.** Rejected on cost and timing, *not* on
  soundness — the data supports it, which is the correction this ADR records. It would
  require killing four syntheses ~19 h into a ~3-day search, doubles the condition count
  as above, and invalidates every existing `constants.csv` and `category_map.json`, with
  `transfer_equation.py` and `symbolic_regression_eval.py` to re-check. Revisit for a
  future round, where the measured price is one extra shape DOF.
- **One extra global per-arm scale parameter, keeping 32 conditions.** Rejected on
  evidence. It is the most parsimonious reading of the stable 9.3× ratio, but the ratio's
  stability is a fact about *scale* and the arms also differ in *peak position*, which no
  scale can correct. Measured ceiling 14.5 % / 15.8 % relative RMS, against 0.5–2 % for
  2–3 DOF within an arm.
- **Rely on the cross-arm shape correlation alone to decide.** Rejected as
  uninterpretable without a baseline: the median cross-arm correlation (0.900) looks
  alarming until set beside m0.9's own cross-condition correlation (0.896), which the
  template handles. The DOF comparison is the measurement that discriminates.

## Consequences

- **No change to the running syntheses, the code, or ADR 0016's decisions.** Only its
  stated reason for rejecting pooling changes, and CONTEXT.md's **Synthesis** entry is
  reworded to match.
- **The "no shared shape across arms" statement must not be repeated** in the thesis or in
  future ADRs. The defensible statements are: the arms differ ~9.3× in σ scale and not at
  all in μ scale; and their normalised shapes belong to one family separated by about one
  degree of freedom.
- **A cross-arm pooled synthesis is now a costed option, not a closed one.** Anyone
  revisiting it needs `optimizer` in `CONDITION_KEYS`, 64 conditions, and a search budget
  raised to cover twice the constants plus one more shape DOF.
- **The arms fit unequal run populations** (239 m0.0 against 255 m0.9; 16 more m0.0 runs
  fail the `include_diverged_training` / `include_nonfinite_schedules` filters). Harmless
  while scoped, since each fit is self-contained; it becomes a weighting question the day
  pooling is adopted.
- **ADR 0016's μ shape figures are superseded** by the numbers above. Anything else derived
  from that pre-re-fetch m0.0 arm deserves the same re-check.
