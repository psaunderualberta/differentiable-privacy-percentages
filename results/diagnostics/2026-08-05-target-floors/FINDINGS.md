# Target floors: histogram check, non-private controls, and a seating bug

**Date:** 2026-08-05
**Prompted by:** `handoff-eyepacs-histogram-and-control.md`
**Code under test:** `f650e84` (main). Fix on branch `worktree-fix-seat-on-budget-units` (`f44d39a`).

## Summary

The transfer probe's "two of three targets at floor" result was **two different
causes**, not one:

| target | probe cell | verdict |
|---|---|---|
| imagenet-32 | 0.925 / 0.975 / 0.975 | **bug** — `seat_on_budget` units error; fixed → 16.0% |
| chexpert | 65.4 mean | **bug (partial)** — same error cost ~5.6pp; fixed → 70.5% |
| eyepacs | 73.96 / 73.96 / 73.98 | **real** — floors even with no DP at all |

Neither target was unlearnable. One bug was suppressing all three cells; EyePACS
independently has no resolving power.

---

## Check 1 — class histograms (no bug found)

Loaded each target through the real transfer path
(`build_target_config` → both config scopes → `get_dataset_loader`).

| target | classes present | eval-split majority | one-hot rowsums | images |
|---|---|---|---|---|
| eyepacs | 5 / 5 | **73.982%** | all exactly 1.0 | real fundus photos ✅ |
| imagenet-32 | 100 / 100 (~1% each) | **1.125%** | all exactly 1.0 | real 32×32 naturals ✅ |
| chexpert | 2 / 2 (59.7/40.3) | **60.072%** | all exactly 1.0 | real chest X-rays ✅ |

All match the expected balance. The `_imagenet100_select` wnid/name bug is **not**
present. Sample grids: `*_samples.png` in this directory.

Two corrections to the handoff's framing:

- EyePACS's 73.96/73.96/73.98 is *exactly* the 73.982% eval majority — confirmed collapse.
- ImageNet-32 is **at chance, not below it**. Uniform chance is 1.0%, majority 1.125%;
  0.925–0.975% on n=4000 is within ~0.5 binomial SD of 1.0%. There is no anomaly to
  explain, only an absence of learning.

## Check 2 — non-private controls

### The prescribed recipe was unreachable

`approx_to_gdp` **hard-fails for ε ≥ ~89** — the root function contains `jnp.exp(eps)`
and JAX runs float32 here, so `exp(89) = inf` and the bisection throws.

| ε | 10 | 50 | 80 | 88 | 89+ |
|---|---|---|---|---|---|
| μ | 1.725 | 6.136 | 8.477 | 9.050 | **raises** |

Even at the ceiling, μ≈9 gives EyePACS σ=0.43 vs 0.68 at ε=10 — a 1.6× noise
reduction, nowhere near σ→0. Such a run could not have separated "task is hard" from
"DP still binding".

**Instead:** set **σ = 0 directly** on the schedule. `get_spherical_noise` returns
`σ·normal/L`, so σ=0 removes the Gaussian mechanism exactly, bypassing the accountant.
This is ε→∞ in the only sense that affects training.

### C is a PSAC scale, not a clipping ceiling

A first attempt set `clip=1e6` to "disable clipping". That is wrong here:
`sum_clipped_per_example_grads` applies the **DP-PSAC** multiplier

```
C / (‖g‖ + 1/(‖g‖ + 1))          # unbounded above
```

not Abadi's `min(1, C/‖g‖)`, despite the docstring saying "Abadi-clipped". `C=1e6`
with a typical ‖g‖≈7 multiplies gradients ~1.4×10⁵ and diverges at every LR. That
whole wave of runs was discarded. Controls hold C at a normal value (the reference
sweep draws C ~ U(0.1, 5)) and vary only σ.

Consequence: **(C, lr) are degenerate** — update magnitude ∝ C·lr, and the DP arm's
noise std is also ∝ C, so SNR is C-independent. Sweep lr at fixed C.

### Results (T=5000, C=1, SGD momentum 0.9, seed 0)

DP arms are seated exactly on the ε=10 boundary: constant σ\* solving
`T·exp((C/σ)²) = (μ/p)² + T` (verified `spent/bound = 1.000000`, and
`project_inverse_sigmas` leaves it unchanged).

**EyePACS** — floor 73.982%. Floors at every LR, in **both** arms:

| arm | lr | σ | val | test | train loss min |
|---|---|---|---|---|---|
| non-private | 0.3 | 0 | 73.982% | 75.360% | 0.7665 |
| non-private | 0.1 | 0 | 73.982% | 75.360% | 0.7482 |
| non-private | 0.03 | 0 | 73.982% | 75.360% | 0.7821 |
| DP ε=10 | 0.3 | 0.6832 | 73.982% | 75.360% | 0.7663 |
| DP ε=10 | 0.1 | 0.6832 | 73.982% | 75.360% | 0.7485 |
| DP ε=10 | 0.03 | 0.6832 | 73.982% | 75.360% | 0.7858 |

Every cell lands on the identical accuracy, and the DP arm's train loss matches the
non-private arm's to ~3 decimal places — removing the privacy mechanism entirely
changes *nothing*. Train loss falls 1.659 → ~0.75–0.78, but the **class-prior entropy
is 0.873**, so the model barely learns more than the marginal label distribution.
Noise is not the binding constraint here; the task under this from-scratch paradigm is.
(The architecture was the other suspect — see the control immediately below, which rules
it out over a 1.9× capacity range.)

**Architecture control (added 2026-08-05).** To test the "surrogate architecture" half
of that, the non-private arm was rerun on a deliberately larger net — `deep3`, three
stride-2 blocks, **466,661 params vs the surrogate's 241,909 (1.9×)** — sized so the
256×256 input downsamples in stages instead of being crushed by one 8×8/stride-2 stem:

| arch | params | lr | σ | val | test | train loss |
|---|---|---|---|---|---|---|
| `cnn-16x32-head32` | 241,909 | 0.3 | 0 | 73.982% | 75.360% | 1.659 → 0.7665 |
| `deep3` | 466,661 | 0.3 | 0 | 73.982% | 75.360% | 1.611 → 1.21 |

Identical on both splits, to three decimals. **Capacity is not the binding constraint.**
Caveat worth stating: `deep3`'s train loss is still *above* the 0.873 prior entropy at
5,000 steps, so in isolation it could be called undertrained — it is the surrogate,
which gets *below* the prior on train and still returns exactly the majority rate on
held-out data, that closes the argument. The small net learns something on train that
generalises to nothing; the large net does not get that far in the step budget.

Run: `arch_control.py --arch deep3 --lrs 0.3 0.1 0.03 --T 5000` (lr 0.3 shown; 0.1 and
0.03 were still running when this was written — the surrogate returned the same value at
all three, and any deviation here would be recorded as an amendment).

**ImageNet-32** — chance 1.0%, floor 1.125%:

| arm | lr | val | vs chance |
|---|---|---|---|
| non-private | 0.3 | **29.45%** | 29× |
| non-private | 0.1 | 22.95% | 23× |
| DP ε=10 | 0.03 | **14.50%** | 14.5× |
| DP ε=10 | 0.1 | 14.25% | 14× |

**CheXpert** — floor 60.072%: non-private 66.80%, DP ε=10 (σ=0.360) **67.08%**.
DP at ε=10 costs essentially nothing; the ceiling is architectural.

---

## The bug: `seat_on_budget` units error

`util/transfer.py:seat_on_budget` operates in **multiplier** units `s = σ/clip` — the
unit `project_inverse_sigmas` documents. But `build_curve_schedule` and
`transfer_equation` both passed the **raw σ**. That substitutes `C_i := 1` into

```
sum_i exp((C_i / σ_i)²)  ≤  (μ/p)² + T
```

For a real learned curve (σ down to 0.0252) `exp(1/σ²)` overflows to `inf` across the
whole bracket, so the bisection never brackets a sign change and — running with
`throw=False` — returns its ceiling `c = 10.0`, **identically for all three targets**
despite their bounds differing by 26×.

Measured on source `4rh8p1j8` (fashion-mnist, ε=10, T=5000):

| target | budget spent (buggy) | scale applied | correct scale |
|---|---|---|---|
| imagenet | **0.66%** | 10.01 | 0.858 |
| eyepacs | **12.16%** | 10.01 | 1.307 |
| chexpert | **0.46%** | 10.01 | 0.831 |

Every transferred curve was ~10× over-noised.

### A/B, real curve-transfer path, 3 seeds

| target | variant | budget used | val mean | probe cell |
|---|---|---|---|---|
| imagenet | buggy | 0.66% | **1.03%** | 0.925/0.975/0.975 ← reproduced |
| imagenet | fixed | 100.0000% | **16.02%** | — |
| chexpert | buggy | 0.46% | **64.90%** | 64.7/66.2 ← reproduced |
| chexpert | fixed | 100.0000% | **70.49%** | — |

The buggy path reproduces the probe cells exactly, which is what identifies the bug as
their cause.

### Why the tests missed it

`test_transfer_curve` asserted the budget as `sum exp(1/σ²)` — the same units error —
and `test_transfer.py::TestSeatOnBudget` only ever used σ ∈ [1, 3], which never
overflows and never involves clips.

### Fix (branch `worktree-fix-seat-on-budget-units`, commit `f44d39a`)

- Convert at both call sites (divide by clip, multiply back); `seat_on_budget` keeps
  its documented multiplier contract and signature.
- `seat_on_budget` now verifies its own postcondition and raises rather than returning
  a silently unbound seating.
- Tests corrected to the clip-aware invariant, plus a regression test using a realistic
  small-σ learned curve.

108 transfer tests pass. Fixed production path reproduces an independent reference
implementation to 0.01pp (16.025% vs 16.017%).

---

## What follows

**ImageNet-32 — keep, do not escalate.** ADR 0007's trigger ("escalate to ImageNet-64
if the Constant reference fails to clear ≈2× chance") must not be evaluated against
the old cells: they were produced by the buggy seating. At ε=10 with a correctly
seated curve the target reaches 16% (16× chance) and with a constant schedule 14.5%.
The target has ample resolving power. Confirmed on reference cells below.

### Reference separation (added 2026-08-05)

The headroom numbers above come from diagnostics, not from `producer="reference"` cells.
Once the full reference stage landed for both surviving targets at (ε=10, T=5000), the
*separation* half of ADR 0020's criterion could be measured directly — 3 native
references × 8 seeds each:

| target | Constant | Median | Dynamic-DPSGD | gap | pooled σ_eval | gap/σ | ANOVA |
|---|---|---|---|---|---|---|---|
| CheXpert | 69.620 ± 0.355 | 70.615 ± 0.201 | 70.713 ± 0.462 | 1.093 pp | 0.356 | 3.07 | F=23.1, p=4.9e-06 |
| ImageNet-32 | 6.538 ± 1.011 | 13.575 ± 1.122 | 14.675 ± 0.632 | 8.138 pp | 0.945 | **8.61** | F=174.5, p=8.3e-14 |

Both separate; ImageNet-32 far more strongly, and it is the only target where all three
pairwise contrasts reach significance (CheXpert's Median-vs-Dynamic is d=0.28, p=0.59).
Read that structurally: on both targets the separation is dominated by **Constant vs the
adaptive family**, and it is the *within-adaptive* contrast (ImageNet d=1.21 vs CheXpert
d=0.28) that bears on ranking transferred curves against each other.

Note also that on CheXpert the val-loss ordering **inverts** relative to accuracy —
Constant has the lowest loss (0.9309) and the lowest accuracy — whereas on ImageNet-32
loss and accuracy agree. CheXpert's ~1 pp gap is not loss-driven.

**EyePACS — drop as a measurement instrument.** This is the handoff's first branch and
it does not depend on the bug: EyePACS floors at 73.982% with **no DP at all**, at
every learning rate, and the ε=10 arm is indistinguishable from the non-private one in
both accuracy and train loss — there is no privacy cost to measure because there is no
signal to degrade. Train loss stays barely below the class-prior entropy. Per ADR
0007's own criterion for rejecting full ImageNet — "from-scratch private accuracy is a
floor, so per-curve transfer differences would be unresolvable noise" — EyePACS now
meets that same bar. Lead with resolving power, not cost. From-scratch EyePACS
collapsing to majority is expected; published numbers use pretrained backbones.

**CheXpert — keep.** 70.5% vs 60.07% floor once seated correctly.

**All curve and equation cells produced before `f44d39a` are invalid** and need
regenerating. The **reference** stage is unaffected — it uses native baselines and
never calls `seat_on_budget`.

## Reproduction

```bash
V=/home/psaunder/Documents/Masters/differentiable-privacy-percentages/.venv/bin/python
$V check1_histograms.py                                   # class histograms + image grids
$V check2_control.py --target eyepacs --T 5000 \
     --arms nonprivate dp --lrs 0.3 0.1 0.03              # non-private / DP controls
$V arch_control.py --arch deep3 --lrs 0.3 0.1 0.03 \
     --T 5000                                             # 1.9x-capacity arch control
$V probe_gradnorm.py imagenet chexpert                    # per-sample gradient norms
$V verify_seat_bug.py                                     # bug mechanism, standalone
$V curve_ab.py imagenet                                   # buggy-vs-fixed A/B
$V reference_separation.py [<cache>/transfer/reference]   # ADR 0020 separation table
```
