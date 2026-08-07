# Transferred schedules are tuned on the target, not borrowed verbatim

The `curve` and `equation` producers now mean **tuned** transfer: a transferred schedule's
free parameters are searched on the target the same way the native references already are.
This supersedes the direct-transfer reading of ADR 0008 for both producers and reuses
ADR 0019's candidate/selector split as the search machinery. `reference` cells are
unaffected — a native reference was always tuned on its target, which is the asymmetry
this ADR removes.

## Status

accepted (supersedes the direct-transfer reading of ADR 0008; extends ADR 0019; keeps
ADR 0018's and ADR 0021's arm scoping unchanged)

## Why

Today's matrix compares a **tuned** baseline against an **untuned** transferred policy. A
reference gets a 20-candidate random search on the target (ADR 0019); a transferred curve
or equation gets its source's numbers with nothing adapted except `seat_on_budget`'s single
budget-binding scalar. Any deficit the matrix reports therefore confounds "this shape does
not transfer" with "nobody tuned it", and that is the one confound the reference column
exists to exclude.

## What is actually free

`seat_on_budget` solves `sum_i exp(1/(c*s_i)^2) = bound` for one scalar `c` on the
multiplier `s = sigma/clip`, and returns `(seat(s)*f_clip, f_clip)`. Two consequences fix
the search space, and both are exact rather than approximate:

1. **A multiplicative prefactor on the sigma curve is a no-op.** Scaling `f_sigma` by `b`
   sends `s -> b*s` and `c -> c/b`, leaving `c*s` invariant — and the invariance survives
   `project_inverse_sigmas`, which only ever sees `c*s`. So the seated schedule depends on
   `f_clip` *exactly* and on `f_sigma` only *up to a positive constant*. This is not a
   tunable direction, it is a symmetry of the seating, and it is what
   `transfer_tuning.seats_identically` tests.

2. **Scaling both curves together is privacy-neutral.** `sigma -> a*sigma`, `clip -> a*clip`
   preserves the ratio, so the GDP spend is bit-identical and a schedule seated on the
   boundary stays on it. Training is *not* neutral to it: under DP-PSAC the clip is a
   scale, not a ceiling (`C/(||g|| + 1/(||g||+1))` is unbounded above), so with the inner
   SGD update this knob is equivalent to scaling the inner learning rate.

Consequence (2) is the correction to the premise this work started from: the joint scale is
available to **any** transferred schedule, a resampled curve included. Curve transfer is not
knob-less. What an equation uniquely adds is *shape* tunability, not tunability as such.

The knob is also the right one on its merits: it compensates for the **target's**
gradient-norm regime — the one thing a source shape cannot know — rather than for anything
about the source.

## Decision

Two nested stages, both scored through ADR 0019's candidate/selector machinery, keyed by a
`sweep_id` rather than a reference name.

**Stage A — the joint `(sigma, clip)` scale.** Curve *and* equation. Ten points,
ratio-2, `2^-6 .. 2^3`, with `1.0` a grid point exactly so "tuning helped" stays separable
from "tuning moved things".

The endpoints are anchored on the reference's own search rather than guessed:
`Baseline.candidate_schedules` draws a constant clip from `U(0.1, 5.0)` and seats sigma to
the budget, which is precisely this knob, and matching that support is what makes the two
arms comparable. Because the scale is *relative* to the transferred curve, the support it
reaches depends on the level the source learned — and the arms are nearly a decade apart:
the median mean-clip over FirSweep's 959 runs is **5.1 at `sgd-m0.0` and 0.63 at
`sgd-m0.9`**. Spanning `U(0.1, 5.0)` from *either* level is what sets the endpoints, and it
is why the grid is asymmetric about 1 — learned clip levels sit above the reference's box
more often than below it, so the headroom that matters is downward.

**Stage B — the template's per-condition shape constants.** Equation only. Each constant is
swept across its own observed range over the trained conditions, widened by 50% of that
range's span, **one at a time**.

Per-constant empirical ranges rather than a relative box, because the K constants are
heterogeneous in role — exponent, rate, offset — so a uniform percentage of each *value*
means something different for each and nothing at all for one sitting near zero. The
observed range is also, by construction, where the fitted template is known to produce
sensible shapes.

One-at-a-time rather than a joint sample: at ~12 points in 3-D with a per-score noise at or
above the 0.36pp cell-level `sigma_eval`, nothing could fit an interaction model anyway, and
OAT yields per-constant sensitivity curves, which is the more useful result.

**Screening, CPU-side, before anything is submitted.** `_TemplatePredictor.predict`
evaluates under `errstate(all="ignore")`, so an out-of-range constant returns silent NaN/Inf
and each unscreened bad candidate costs a GPU-hour to discover. Two filters: *validity*
(finite and strictly positive on the target grid — seating computes `exp(1/s^2)`) and
*degeneracy* (`seats_identically` against the candidates already kept). The degeneracy test
is the algebraic one above rather than pushing each perturbation through the real seater,
which keeps `transfer_tuning` jax-free — so the launcher can size its arrays from the same
screened list the producer indexes into — and avoids inventing a "did it move enough"
threshold on top of the bisection's own tolerance.

On the real FirSweep syntheses the screen drops **56%** of stage-B candidates at `sgd-m0.0`
(only `sigma.p2`, `sigma.p3`, `clip.p2` are identifiable) and **36%** at `sgd-m0.9`.

**Stage A is tuned per `(target x arm)` and shared across every source.** This is the single
largest cost lever and it is principled, not just cheap: the scale tracks the target, not the
source. `transfer_launch.SOURCE_SCOPED_SWEEPS` is the one place that decides it, so a caller
cannot widen or narrow a pool by accident. Stage B is per-condition, because a template
constant's winner means nothing for another condition.

**Candidates are scored at reduced reps; the winner is re-evaluated at `num_reps`**, on the
Baseline's own key — disjoint from the scoring draws, exactly as ADR 0019 requires, so the
reported number is not the draw that selected it.

**The winning knobs are recorded on the cell rows** (`tuned_scale`, `tuned_constants`). A
cell's accuracy is now the accuracy of a tuned schedule, so without them the number is
unreproducible; and "which scale did each target prefer" is a result in its own right. A
cell whose rows disagree is labelled `mixed` rather than resolved, since stage-A sharing
makes agreement an invariant. Cells written before this ADR read as untuned, which keeps
the existing `reference` cells readable.

## Cost

Measured with `transfer_launch`'s own enumeration on FirSweep, at the two targets currently
on disk (chexpert, imagenet at eps=10, T=5000), scope `cnn-16x32-head32` / 4 seeds:

| | today | tuned adds |
|---|---|---|
| curve | 248 policies -> 496 cells | 40 stage-A scoring tasks |
| equation | 8 cells | 40 stage-A + 108 stage-B scoring tasks |
| reference | 12 cells (240 candidate tasks) | unchanged |

At ~1.4 GPU-h a task that is **~263 GPU-h added against ~706 GPU-h** for today's curve and
equation stages; scoring at one rep instead of three brings it to ~88 GPU-h. The curve
stage grows by **8%**, not the 3-4x a per-cell scale sweep would have cost — entirely
because of stage-A sharing.

## Consequences

- Existing `curve` and `equation` cells are superseded and must be re-run. `reference`
  cells are not.
- The equation stage becomes three-phase (scale scoring -> shape scoring -> selector) and
  the curve stage two-phase, so `plan_jobs` needs the dependency-ordered insertion it
  already does for `reference`.
- A tuned deficit is a stronger claim than an untuned one: if a tuned transferred schedule
  still trails a tuned reference, "nobody tuned it" is no longer available as an
  explanation.

## Alternatives rejected

**Land tuning as a separate `equation_tuned` producer.** Rejected: it would leave two
columns whose difference is a tuning protocol rather than a schedule, and the untuned column
answers no question the reference column does not already answer better.

**Sweep the sigma equation's prefactor.** Rejected as an exact no-op — see consequence (1).

**A joint 4-D search over scale and constants.** Rejected: at ~20 candidates the joint box
is far too sparse to resolve differences at `sigma_eval ~ 0.36pp`, so the nesting is what
makes the search resolvable at all.

**Buying precision with more seeds instead.** Rejected: reps stop helping past `n_reps ~ 2`
and the curve stage already runs 3 — transfer precision is not compute-limited.
