# Drop EyePACS as a transfer target: no schedule-resolving power under the surrogate regime

EyePACS is removed from the transfer evaluation's target set. It returns its 73.982%
majority-class rate under *every* schedule it has been given — including with the
Gaussian mechanism switched off entirely — so no cell in its column can measure a
difference between schedules. The remaining targets are **CheXpert** and
**ImageNet-32**, and the compute this frees is spent widening the budget axis rather
than banked — so the matrix stays 6 **target regimes**, now 2 datasets × 3 budget points
(ε=10, T ∈ {2000, 5000, 7000}) rather than 3 datasets × 2.

This supersedes the EyePACS bullet of ADR 0007 ("used as-is ... the one target that is
natively a from-scratch small-CNN task") and its Consequences prescription to validate
the pipeline on EyePACS first.

## Status

accepted (supersedes the EyePACS target in ADR 0007)

## The criterion: schedule-resolving power

A target regime is admissible iff differently-shaped schedules, run natively on it at
its own budget, separate by more than evaluation noise. This is the same criterion
ADR 0007 already used to reject full 1000-class ImageNet — *"from-scratch private
accuracy is a floor, so per-curve transfer differences would be unresolvable noise"* —
applied to EyePACS and now **measured rather than predicted**.

Two things about how it is measured matter enough to state:

- It is read off the **transfer references** (Constant / DynamicDPSGD /
  StatefulMedianGradient), which are native to the target and contain **no transferred
  policy whatsoever**.
- It is **not** the gap between a non-private control and a DP run. That gap confounds
  the privacy mechanism with hyperparameter search, and measuring it that way produced a
  wrong answer here — see "A measurement that had to be discarded" below.

### What the surviving targets score

**CheXpert** (ε=10, T=5000, 8 seeds per reference), against a 60.072% floor:

| reference | mean | sd |
|---|---|---|
| Constant | 69.620 | 0.355 |
| Dynamic-DPSGD | 70.713 | 0.462 |
| StatefulMedianGradient | 70.615 | 0.201 |

Constant vs Dynamic-DPSGD: **−1.093 pp, p = 1.4e-4, Cohen's d = −2.65**. Schedule shape
moves CheXpert by roughly 1 pp against a seed sd of ~0.35 pp. It resolves schedules.

**ImageNet-32**, against 1.0% uniform chance / 1.125% majority, same protocol
(ε=10, T=5000, 8 seeds):

| reference | mean | sd |
|---|---|---|
| Constant | 6.538 | 1.011 |
| StatefulMedianGradient | 13.575 | 1.122 |
| Dynamic-DPSGD | 14.675 | 0.632 |

Constant vs Dynamic-DPSGD: **+8.138 pp, p = 2.9e-10, Cohen's d = 9.65**. Every schedule
is far clear of the floor, so ADR 0007's 64×64 escalation trigger is **not** fired; it
appeared to fire only on cells corrupted by the `seat_on_budget` units bug (fixed in
`f44d39a`).

**EyePACS** scores zero range: every arm returns 73.982%, which *is* the floor.

### The criterion has two parts, and both targets now satisfy both

Being scrupulous about this matters, because the temptation is to admit a target on
evidence of the wrong kind:

1. **Headroom above the floor** (necessary). Can *any* schedule beat the majority rate?
   This is measurable from a single native schedule. CheXpert clears it (69.6% vs 60.07%),
   ImageNet-32 clears it decisively (6.5–14.7% vs 1.125%), and **EyePACS fails it
   outright** — its best result at any learning rate, with or without the privacy
   mechanism, at 1.9× capacity, is exactly the floor.
2. **References actually separate** (sufficient). Do differently-shaped schedules give
   *different* answers? This needs the full reference set, and both surviving targets now
   have one at (ε=10, T=5000):

   | target | gap | pooled σ_eval | gap / σ_eval | ANOVA |
   |---|---|---|---|---|
   | CheXpert | 1.093 pp | 0.356 pp | 3.07 | F = 23.1, p = 4.9e-06 |
   | ImageNet-32 | 8.138 pp | 0.945 pp | **8.61** | F = 174.5, p = 8.3e-14 |

   Note this is one budget point out of the three the grid spans; the other four target
   regimes are handled by a pre-commitment in Consequences rather than by a second gate.

**The EyePACS decision turns only on part 1**, which is measured, unambiguous, and
independent of anything the other two do. A target with zero headroom cannot have
separation, so no further measurement could rescue it.

Earlier drafts of this ADR cited a *transferred curve* (16.0%) as ImageNet-32 evidence;
that is exactly the contamination the criterion forbids, and it has been removed. The
14.5% figure those drafts used came from the `curve_ab.py` diagnostic — criterion-pure
for headroom, but not a reference cell. It is superseded above by
`producer="reference"` cells.

### ImageNet-32 is the better instrument, but read the separation structurally

ImageNet-32 dominates CheXpert on every summary statistic, and it is the only target that
resolves *all three* reference pairs. But the two targets have the same qualitative
structure, and the aggregate gap hides it — the separation is almost entirely
**Constant vs the adaptive family**, not *within* that family:

| pair | CheXpert | ImageNet-32 |
|---|---|---|
| Constant vs Median | d = 3.45, p = 2.5e-05 | d = 6.59, p = 3.2e-09 |
| Constant vs Dynamic-DPSGD | d = 2.65, p = 1.4e-04 | d = 9.65, p = 2.9e-10 |
| Median vs Dynamic-DPSGD | d = 0.28, **ns** (p = 0.59) | d = 1.21, p = 0.034 |

This matters because a transferred curve is far more likely to resemble an adaptive
schedule than a constant one, so the resolving power that actually bears on *ranking
transferred policies against each other* is the last row, not the first. On that row
ImageNet-32 still wins — d = 1.21 versus 0.28, and it is the only one that reaches
significance — but the margin is ~4×, not the ~8.6× the aggregate suggests. Cells that
differ only in curve shape should be expected to separate weakly on CheXpert.

One asymmetry worth recording: on ImageNet-32 the val-loss ordering agrees with accuracy
(Constant 4.311 > Median 3.941 > Dynamic 3.769, against ln 100 = 4.605). On CheXpert it
**inverts** — Constant has the *lowest* loss (0.9309 vs 0.9390 / 0.9403) while having the
lowest accuracy. CheXpert's ~1 pp accuracy gap is therefore not loss-driven, which is
consistent with a threshold/calibration effect on an imbalanced near-binary task and is a
second reason to treat its separation as the weaker of the two.

## The evidence for EyePACS

Non-private controls (σ = 0 exactly, so the Gaussian mechanism is removed rather than
loosened), T=5000, C=1, SGD momentum 0.9:

| arm | lr | val | test |
|---|---|---|---|
| non-private | 0.3 | 73.982% | 75.360% |
| non-private | 0.1 | 73.982% | 75.360% |
| non-private | 0.03 | 73.982% | 75.360% |

73.982% *is* the evaluation split's majority-class rate, to three decimals. The matched
DP ε=10 cells return 73.96 / 73.96 / 73.98%. Train loss falls 1.659 → ~0.75–0.78 against
a **class-prior entropy of 0.873**, i.e. the model barely outperforms the marginal label
distribution. Removing DP entirely changes nothing.

Full method, numbers and reproduction commands:
`results/diagnostics/2026-08-05-target-floors/FINDINGS.md`.

### The floor is not an artifact of the surrogate's capacity

The obvious rescue for EyePACS is that the MNIST-derived conv block is simply too small,
so a second non-private control was run on a deliberately larger architecture — `deep3`,
three stride-2 blocks, **466,661 parameters against the surrogate's 241,909 (1.9×)** —
chosen so the 256×256 input is downsampled properly rather than crushed by a single
8×8/stride-2 stem.

| arch | params | lr | val | test | train loss |
|---|---|---|---|---|---|
| `cnn-16x32-head32` (surrogate) | 241,909 | 0.3 / 0.1 / 0.03 | 73.982% | 75.360% | 1.659 → 0.75–0.78 |
| `deep3` | 466,661 | 0.3 / 0.1 / 0.03 | 73.982% | 75.360% | 1.611 → 1.21–1.30 |

The larger network returns **the identical majority rate, to three decimals, on both
splits, at every learning rate** — six independent runs across a 1.9× capacity range and
a decade of learning rate, all returning one number. Capacity is not the binding
constraint.

One honest caveat: `deep3`'s train loss (1.21–1.30) is still *above* the 0.873 class-prior
entropy at 5,000 steps, so on its own it could be dismissed as undertrained. The
surrogate is what closes that gap — it reaches 0.75–0.78, i.e. it fits the training set
*beyond* the marginal label distribution, and still returns exactly the majority rate on
held-out data. Between them: the small net learns something on train that generalises to
nothing, and the large net does not even get that far within the step budget. Neither
produces a target on which schedules can be distinguished.

### What is and is not being claimed

The verdict is scoped to the regime, and the regime is the one ADR 0007 fixed: **from-scratch
small-CNN DP-SGD**. EyePACS has no schedule-resolving power *there*. It is emphatically
**not** a claim that EyePACS is unlearnable — it is a real benchmark on which pretrained
backbones do substantially better than chance, and nothing here contradicts that. The two
controls widen the scope from "under one surrogate architecture" to "across a 1.9× capacity
range and a decade of learning rate, with and without the privacy mechanism", which is
enough to stop treating the architecture as the suspect, and not enough to say anything
about EyePACS under a paradigm this project never runs.

### Why the same surrogate architecture does not floor CheXpert

EyePACS and CheXpert share a conv block — `channels=(16,32)`, k=8/4, s=2, pool=2,
head `(32,)` — inherited from MNIST. So the obvious objection to a *regime-scoped*
verdict is that if the architecture is what floored EyePACS, CheXpert should be floored
by it too, and no CheXpert control was run to rule that out.

The objection is answered by measurement rather than by a control, and the asymmetry is
large:

| regime | input | values/example | vs the block's design point |
|---|---|---|---|
| MNIST (design point) | 1×28×28 | 784 | 1× |
| CheXpert | 1×64×64 | 4,096 | 5.2× |
| ImageNet-32 | 3×32×32 | 3,072 | 3.9× |
| **EyePACS** | **3×256×256** | **196,608** | **251×** |

EyePACS asks a block sized for 784 inputs to absorb 251× that, through a stem whose
first kernel is 8×8 stride 2 — it reaches the classifier having discarded almost
everything. CheXpert's 5.2× and ImageNet-32's 3.9× are ordinary transfers of an
architecture; EyePACS' 251× is not.

More decisively: **CheXpert does not need a control, because it passes the criterion
directly.** Its three references separate by 1.09 pp at p = 1.4e-4 against a seed sd of
~0.35 pp. Resolving power is defined as measurable separation between differently-shaped
schedules, and CheXpert exhibits it. A non-private control could only have bounded what
CheXpert *might* achieve; the references show what it *does* discriminate, which is the
quantity the transfer matrix actually reads. Running a control to reassure ourselves
about a target already demonstrated to resolve schedules would be the same
non-private-vs-DP-gap instrument this ADR discards below.

## Why this is not a garden-of-forking-paths removal

EyePACS was a pre-registered target and it is being removed after results existed, so the
objection has to be answered rather than avoided.

**The criterion is evaluated on a stage that contains no transferred policy.** The
non-private control is a native training run; the transfer references are native
schedules run at the target's own budget. Neither involves a source curve, a distilled
equation, or any output of the mechanism whose performance is in question. The decision
is therefore *structurally incapable* of being influenced by whether transfer works —
it would have come out identically had the transfer stage never been run.

That is a stronger claim than "we didn't look", and it is checkable: the decision inputs
are `producer="reference"` cells and `check2_control.py`, and neither reads a source
policy.

Recorded as a separate ADR, rather than an edit to ADR 0007, precisely so the sequence
stays legible: pre-registered there, removed here, on a stated criterion, on a stated
date.

## A measurement that had to be discarded

The first version of this argument used the **non-private-vs-DP gap** as the criterion,
which on CheXpert gave non-private 66.80% vs DP ε=10 67.08% — a −0.28 pp "cost of
privacy" implying CheXpert was floored too.

That was wrong. All three CheXpert references (69.6–70.7%) beat the *non-private*
control, because the reference sweep searches 20 candidates while the control was a
single untuned point. The control was never a ceiling. **The −0.28 pp figure should not
be cited**, and the non-private-vs-DP gap is not a resolving-power instrument.

The EyePACS conclusion does not inherit this flaw: under-tuning produces scatter, and
EyePACS returned the majority rate *exactly*, at every learning rate, with the mechanism
off.

## Considered and rejected

- **Running EyePACS transfer references before deciding.** ~63 SLURM tasks (20 candidates
  + selector, × 3 references). Rejected: `Baseline.candidate_schedules` searches σ and
  clip only — **not** learning rate — and under DP-PSAC the update is ∝ C·lr while the
  noise std is ∝ C, so clip and lr are the same axis. The control already swept a decade
  of it. This buys a wider search on an axis already covered, for a predictable answer.
- **Keeping EyePACS as a documented floored target.** Rejected as a *target regime*: 248
  policies × its regimes, all of them cells that cannot differ, is 93% of the curve stage
  spent measuring nothing. But see Consequences — the already-computed reference cells
  are kept.
- **Replacing EyePACS with a fourth target.** Rejected for now: it re-opens the
  surrogate-regime design work ADR 0007 settled, and CheXpert + ImageNet-32 span
  medical/natural, binary/100-class, and 5.2×/3.9× MNIST input scale. Revisit only if
  the two-*dataset* matrix proves too thin to support the generalisation claim. Note the
  six target regimes do not make it a six-target matrix.

## Consequences

- **The freed compute is spent on a third budget, not banked — and it is far more than
  one column's worth.** The matrix was 3 datasets × 2 budgets = 6 columns; dropping
  EyePACS leaves 4, and the columns are restored to **6** by widening the budget axis
  from two points to three. Cell count is therefore unchanged at **1,488** (248 policies
  × 6 columns), but *cost* is not, because cells are not fungible: a curve task is priced
  by input resolution, and at 1.24 GPU-h against ImageNet-32's 0.06 and CheXpert's 0.03,
  **EyePACS alone was 93% of the curve stage**. The stage goes from ≈660 GPU-h to ≈63 —
  a 10.6× reduction while measuring the same number of cells across a wider budget axis.
  The one-column-per-column intuition is the thing to distrust here.
- **There is consequently substantial unspent headroom**, and it is deliberately left
  unspent for now. Widening further (more T values, more seeds per cell, a restored ε
  axis) is cheap in a way it never was while EyePACS set the price, but each option is a
  separate design question and none of them is blocked by this ADR.
- **The target grid is a T-spread at fixed ε: ε=10, T ∈ {2000, 5000, 7000}.** ε is held
  constant because the source sweep found it nearly inert — across its 3.3× ε span the
  seated σ moves under 9% — so a second ε would have bought a column that differs from
  its neighbour by less than seed noise. T is the axis schedule *shape* lives on, and
  2000→7000 is a 3.5× span. All six (ε, T) pairs are on the source condition grid
  (ε ∈ {3,5,8,10} × T ∈ {2000,3000,5000,7000}), which the equation stage requires: it is
  on-grid only, and `check_on_grid` aborts rather than degrades.
- **The launcher defaults were stale and are corrected here.** They read
  `target_eps=(1.0, 8.0)`, `target_T=(200, 5000)` — a 12-column cross-product in which
  **ε=1 and T=200 are both off the condition grid**, so a relaunch that inherited them
  would have hard-failed the equation stage on 9 of 12 columns while silently running
  curve and reference on all 12. The defaults now encode the grid above, and the
  dry-run pricing model — which had assumed "every planned target shares T=5000" —
  now scales curve cost linearly in T.
- **No EyePACS reference cells were ever produced, and none are needed.** An earlier draft
  of this ADR said completed EyePACS reference cells would be retained as a negative
  control; that was wrong — the reference sweep for EyePACS was considered and rejected
  (above), so it never ran, and no `producer="reference"` cell exists for it at any budget
  point. The retained negative control is the *control* evidence — `check2_control` and
  `arch_control` JSON under `results/diagnostics/2026-08-05-target-floors/` — which is what
  the headroom half of the criterion is read from anyway.
- **The relaunch ships the curve and reference producers only.** The equation producer is
  blocked on arm-scoped syntheses: the FirSweep SR outputs carry 32 conditions keyed
  `(dataset, eps, T, arch_label)` with no `arm`, so they cannot yield the per-regime-arm
  predictions ADR 0018/0019's schema needs, and both of their `hall_of_fame.csv` files were
  truncated to 0 bytes by an interrupted PySR rotation (the equations survive in the
  adjacent `.bak`, and no `equations.csv` was ever persisted — see `36be7e6`). Replacement
  arm-scoped syntheses are running; equation cells are added when they land. The core
  transfer claim rests on curve-vs-reference and does not wait for them. Note the chosen
  budget points are all on the source condition grid, so nothing about this deferral
  forecloses the equation stage later.
- **The ImageNet-32 gate was a judgement call, not a pre-registered threshold**, and the
  two halves of the criterion differ in kind here. EyePACS failed **headroom**, which needs
  no threshold — it returned the majority rate *exactly*, at every learning rate, with the
  mechanism on and off, at 1.9× capacity. ImageNet-32 was judged on **separation**, where
  no numeric bar was fixed in advance; a candidate rule (max−min reference gap ≥ 3× pooled
  per-seed sd, plus p < 0.01, which CheXpert clears at 3.07σ and p = 1.4e-4) was considered
  and deliberately not adopted. What limited the latitude is the direction of the
  incentive: dropping ImageNet-32 would collapse the matrix to a single dataset, so the
  pressure ran toward *admitting* it, and a judgement call biased toward inclusion is the
  safe orientation for this particular decision.

  **In the event the latitude was not needed.** ImageNet-32 came in at 8.61σ with
  p = 8.3e-14 — it would have cleared the unadopted rule by a factor of ~3 on the gap term
  and ~12 orders of magnitude on the p term, so admitting it required no discretion at the
  margin. The gate is recorded as **passed** rather than waived.
- **ImageNet-32's curve stage was gated on its reference separation, and the gate has now
  been cleared.** Its reference sweeps ran first and had to show separation before its
  curve cells were committed. The compute argument was nearly neutral — the references are
  ~10 GPU-h against ~42 for the curve cells — so the gate cost a queue round-trip, and what
  it bought is the integrity of the criterion: launching in parallel would have meant ruling
  on ImageNet-32's admissibility with its transfer results already in hand. The instrument
  would still have been technically uncontaminated, but the sequence would no longer
  *demonstrate* that, and demonstrability is the entire point of the forking-paths section
  above. Both targets' curve cells are now unblocked.
- **The four budget points that have no reference stage yet are pre-committed, here and
  now, to being reported whatever they show.** Separation was measured at (ε=10, T=5000)
  only, and the criterion above is worded per *target regime*, so on a literal reading
  T=2000 and T=7000 are unqualified on both datasets. Their reference cells come out of
  the same DAG submission as their curve cells, which means admissibility evidence and
  transfer results arrive together — the arrangement the ImageNet-32 gate above exists to
  avoid.

  Gating them the same way is rejected. It buys a second queue round-trip on a question
  that is nearly determined: headroom is a property of the *dataset*, not of the budget
  point, and it is settled for both survivors; T is a knob on the budget, not a change of
  instrument. What replaces the gate is cheaper and strictly stronger — a commitment made
  before the cells exist that **all six regimes are reported, and per-regime separation is
  an interpretive diagnostic attached to a column, never a filter that removes one.**

  Concretely: a column whose references come back tightly bunched is reported as a column
  whose references do not separate, and a column that lands on its floor is reported as
  floored. Both are findings about the target at that budget, and both stay in. The
  EyePACS-style remedy — removal on resolving-power grounds — is available only *before*
  a regime's curve cells are launched, which for these four is now, and it is not being
  exercised. Nothing observed after this point can shrink the matrix. That is what makes
  running the two stages concurrently safe; without the commitment, launching them
  together would leave a drop decision live with the results already in hand.
- **ADR 0007's validation order is reversed and reassigned**: validate the pipeline
  end-to-end on **CheXpert first, then ImageNet-32**. Both now have a complete post-fix
  reference stage at (ε=10, T=5000) and both demonstrate separation, so the tie-break is
  cost: CheXpert measures at 0.03 GPU-h per curve task against ImageNet-32's 0.06. A null
  result on either during future validation is informative. ImageNet-32 stays later as the
  target ADR 0007 calls riskiest for a floor effect — a ranking the reference stage has
  since inverted, since it is ImageNet-32 that separates more strongly. The original
  ordering put EyePACS first while EyePACS was by a wide margin the *most* expensive target
  to run — 196,608 input values per example.
- Targets are CLI arguments (`--target`, `--target_eps`, `--target_T`, fanned out by
  `expand_targets` in `src/transfer_launch.py`), so the drop is a launch-flag change.
  Remaining `eyepacs` mentions in `src/transfer_*.py` are docstring examples and test
  fixtures using it as an arbitrary name; neither needs changing.
- The generalisation claim now rests on two target **datasets** (six target regimes,
  but a regime is not an independent target — the three budget points within a dataset
  share its data, surrogate and floor). This is thin, and is the
  first thing to revisit if the matrix under-delivers.
