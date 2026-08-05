# Scope each FirSweep synthesis to one arm and the T-sweep axis, and weight runs equally in step_norm

A symbolic-regression **synthesis** over FirSweep covers **one arm** (`--optimizers`) and
**one axis** — the T-sweep (`--arch_labels cnn-16x32-head32`). FirSweep is the first sweep
to carry both arms in a single results directory, and the **condition**
`(dataset, eps, T, arch_label)` that indexes the **per-condition constants** does not name
the arm, so an unscoped synthesis silently merges them. Runs are additionally sampled at a
fixed number of points per run rather than a fixed inner-step stride, so every run
contributes equally over `step_norm`.

## Status

accepted (scopes ADR 0006 for FirSweep)

## Why

**The arm scoping is a correctness requirement, not a presentation choice.** FirSweep's
110 `(dataset, eps, T, arch, arm)` cells collapse to 62 conditions under `CONDITION_KEYS`,
and **48 of those collide across arms**. The collision is not benign: at
`mnist/eps=3/T=2000/cnn-16x32-head32` the median σ is 4.70 (m0.0) against 0.556 (m0.9).
One constant vector would be fitted to a target whose two modes differ by ~8.5×. Earlier
sweeps were safe only because the arms lived in separate W&B projects, hence separate
cache directories; ADR 0011 moved the arm onto the run and removed that accident.

Pooling the arms is not rescued by fitting μ = C/σ instead. μ *is* arm-invariant in scale
(ratio 1.01, as the privacy projection requires), but its **shape** is not: the peak sits
at `t/T ≈ 0.70` at m0.0 against `≈ 0.45` at m0.9, with normalised shape correlation
0.65–0.88. No single universal schedule shape covers both arms in σ, clip, or μ.

**The axis scoping buys clean data and a coherent question.** The two axes are structurally
disjoint: the T-sweep is 16 `(eps, T)` points at one architecture; the arch axis is ~30
architectures at the single point `(eps=10, T=5000)`. Pooling them asks one shape to
explain variation over `(eps, T)` and over architecture at once, with both absorbed into
the same K constants — an architecture-driven shape change would be indistinguishable from
an ε/T-driven one.

The T-sweep is also the only clean subset. Across its 386 runs there are zero non-finite
schedule values, zero **diverged runs** (minimum accuracy 82.7 %, against a chance level of
10 %), one **truncated run** (at step 978 of 1000), and no run whose peak σ exceeds 3× its
cell median. The arch axis carries all 10 divergences and 116 of the sweep's 119
truncations. Scoping to the T-sweep therefore needs **no run-filtering machinery at all**;
the stock `include_nonfinite_schedules` / `include_diverged_training` defaults are no-ops
on it.

**Equal weighting per run** matters because the fit is over `step_norm ∈ [0,1]` while the
old rule sampled every 100th *inner* step. T=2000 then contributes 20 points across that
interval and T=7000 contributes 70, so T=7000 supplied ~42 % of all rows and T=2000
~12–15 %. The shared shape would be fitted mainly to the largest T — precisely the
low-amplitude end of the peak-height-versus-T trend the synthesis exists to characterise.
The extra rows are not extra information either: adjacent inner steps on a smooth B-spline
curve are near-duplicates.

## Considered and rejected

- **Add `optimizer` to `CONDITION_KEYS` and fit both arms together.** Rejected: it changes
  the category-map format for every existing synthesis, and it doubles a condition count
  that ADR 0006 already flags as the point where fits get slow (110 conditions × K
  constants re-optimised per candidate). It would also assert a shared shape the μ
  comparison above shows does not exist.
- **Pool both axes per arm.** Rejected: 48/62 conditions, imports every diverged and
  truncated run, and confounds the architecture and `(eps, T)` explanations as described.
- **Exclude truncated runs to salvage the arch axis.** Rejected as the way to make pooling
  safe: truncation is systematic per rung (all 8 seeds of `mlp-512` stop near step 135 in
  both arms), so a `final_outer_step == 1000` filter deletes whole rungs — structured
  survivorship in exactly the comparison the arch axis exists for. Scoping the axis out is
  honest; filtering it down is not.
- **Detect divergence by a σ-magnitude outlier test.** Rejected: at a 3×-cell-median
  threshold it flags the 10 genuinely diverged runs *and* three healthy runs at 83.9 / 86.0
  / 96.9 % accuracy whose only fault is the boundary spike near `t/T = 0`.
- **Trim the unreliable `step_norm` ends** (across-seed CV reaches 0.60 at `t/T = 0` and
  0.23 at `0.98`, against 0.009 in the plateau). Rejected: `evaluate_equation_shape`
  evaluates the closed form on `linspace(0, 1, target_T)`, so anything trimmed becomes
  extrapolation at transfer time, where an unconstrained symbolic form can return a
  negative σ — an invalid noise multiplier rather than merely an inaccurate one. With all
  seeds retained the noisy ends are heteroscedastic but unbiased.

## Consequences

- **Four syntheses per full FirSweep analysis**: 2 arms × {σ, clip}. Both targets are
  required — ADR 0001's privacy-budget validity check needs both, and so does
  `transfer_equation.py`. They must be submitted as separate jobs, since `main()` applies
  the search timeout per target (see ADR 0002's rejection of the multi-target loop).
- **No cross-arm shape claim is available.** The reportable statement is "within each arm,
  every learned schedule is the universal shape with K knobs", never one law for both.
  The arms' relationship is a separate, already-established finding: the joint (σ, C) scale
  differs by the momentum gain while μ is preserved.
- **No arch-invariance claim is available** from these syntheses. Establishing one needs a
  separate arch-axis synthesis, which must first confront the truncation problem this ADR
  scopes around.
- **Sampling density is now a fit-defining field.** `points_per_run` (default 50) *replaces*
  ADR 0005's `datapoint_frequency` rather than coexisting with it — one sampling knob, no
  precedence rule — and takes its place in `IDENTITY_FIELDS`, or two differently-sampled
  syntheses share a slug and corrupt each other's run directory. The stride is taken from
  each run's own observed step span, so a truncated run still contributes its full share.
- **The operator set is likewise identity-tracked.** It was hardcoded and narrower than
  PySR's own default: `["*", "/"]` cannot express an additive offset, so a per-condition
  constant could only rescale the universal shape, never shift its peak — the modulation
  the FirSweep shape plots show most clearly. `binary_operators` defaults to
  `("+", "-", "*", "/")`; PySR's GP `denoise` is off, since it smooths over every column of
  X including the arbitrary `category` index.
- **The T-sweep grid fixes where equation transfer can land.** Per-condition constants are
  undefined off-grid, so transfer targets must be run at `eps ∈ {3, 5, 8, 10}` and
  `T ∈ {2000, 3000, 5000, 7000}`.
- The m0.0 arm is unusable until the 108 MNIST T-sweep runs lost to a W&B download failure
  are re-fetched: without them that arm holds 19 of 32 conditions and only 20 MNIST runs
  against 111 Fashion-MNIST. Because `cache_dir` is hashed by basename only, the re-fetch
  must land **before** the first synthesis, or the same slug will cover two different
  datasets.
