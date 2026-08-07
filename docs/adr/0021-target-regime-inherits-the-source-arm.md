# The target regime inherits the source policy's arm

Every transfer evaluation to date ran its target at inner SGD momentum **0.9**, whatever
arm the source policy was learned in, because `build_target_config` never sets an
optimizer and `EnvConfig` takes the `SGDConfig` default (`momentum=0.9`). A policy
learned in the `sgd-m0.0` arm was therefore transferred onto a target running `m=0.9`.
From now on the target inherits the source's arm, and the **transfer references are run
once per target momentum** (36 reference cells, not 18) so each arm is judged against a
like-for-like baseline.

## Status

accepted (amends the target-config construction assumed by ADR 0008; invalidates the
`sgd-m0.0` half of the batch under `cache/transfer/psaunder__FirSweep/`)

## What was wrong

`util/transfer.py:build_target_config` builds `EnvConfig(eps=…, delta=…, batch_size=…,
num_training_steps=…)` and leaves `optimizer` to its default factory. Both producers —
`transfer_curve.py:179` and `transfer_reference.py:140` — go through it, so:

| source arm | target inner momentum | relationship |
|---|---|---|
| `sgd-m0.9` | 0.9 | matched |
| `sgd-m0.0` | 0.9 | **mismatched** |
| references (arm `""`) | 0.9 | matched by construction |

The arm is consequently **perfectly confounded with source/target optimizer match**, and
the confound runs in one direction only: all three references sit on the matched side, so
an `sgd-m0.0` row loses to its baseline partly by construction.

This is not a small effect. On the affected batch the arm explains 91–93% of the variance
in ImageNet-32 transfer accuracy and 57–74% on CheXpert at T ∈ {5000, 7000}, against
≤ 10% for source dataset, source ε and source T *combined*. The two arms' ImageNet ranges
do not overlap: `sgd-m0.9` spans 11.7–17.2%, `sgd-m0.0` spans 1.0–7.3%, and **0 of 180
`sgd-m0.0` regime-cells beat any reference at any target regime**.

One thing the bug did *not* corrupt: because the target was identical across arms, the
observed arm effect is a genuine property of the transferred schedule shape rather than a
target-side artifact. What cannot be recovered from this batch is *which* property — the
data cannot separate "schedules learned under momentum transfer better" from "schedules
transfer when the source and target optimizers agree."

## Considered and rejected

- **The full 2×2 cross** — every source arm × target momentum ∈ {0.0, 0.9}. This is the
  scientifically complete design and the only one that separates the two readings above,
  because it supplies both off-diagonal cells. Rejected on scope, not on cost (~3,000 curve
  cells against ~1,500, both affordable now that EyePACS is gone — ADR 0020): inner
  momentum is a nuisance dimension, and CONTEXT.md already defines an **arm** as "a
  condition under which the *entire* axis matrix is replicated", i.e. something replicated
  across and never pooled, precisely so it need not be reasoned about as a factor. Spending
  half the transfer stage on a 2×2 interaction in a variable the thesis does not claim
  anything about is the wrong allocation.

  The cost of this rejection is explicit: **the optimizer-mismatch question is given up,
  not answered.** Nothing in the new batch will say what happens when a schedule meets an
  optimizer it was not learned under. If that becomes interesting, it is a separate
  experiment and needs its own ADR.
- **Pin the target at one momentum and drop the mismatched sources.** Halves the batch and
  yields one clean claim with no arm dimension at all. Rejected: it discards ~120 learned
  policies to answer strictly less, and it silently changes what the FirSweep arms were
  replicated *for*.

## Consequences

- **The `sgd-m0.0` half of the current batch is invalid and must be re-run**: 723 of the
  1,489 curve cells (≈120 policies × 6 target regimes, plus 3 stragglers carrying no arm).
  The `sgd-m0.9` cells were already matched and their numbers are unchanged by this
  decision — every conclusion resting on them survives the re-run.
- **References double to 36 cells** (3 mechanisms × 6 target regimes × 2 target momenta),
  and each candidate sweep re-runs at the new momentum. A reference is native to its target,
  so a reference tuned at `m=0.9` is not a baseline for an `m=0.0` target regime; reusing the
  existing 18 would reintroduce the same one-sided confound this ADR removes.
- **The two arms' matrices are never compared to each other.** They now differ in the target
  configuration, not only the source, so their accuracies are not on a common scale. This is
  the existing CONTEXT.md rule ("analyses and plots are split per arm, never pooled across
  them") acquiring a second, harder reason: pooling was previously a
  confounding problem, and is now a units problem. ADR 0022 fixes the plot layout that
  follows from it.
- **`source_arm` becomes a property of the whole transfer, not just its source.** The row
  schema is unchanged — the arm is still read off the source's `optimizer` column
  (ADR 0011) — but its meaning widens, and any future producer that builds a target config
  must thread it through rather than relying on the `EnvConfig` default.
- **The default is left in place rather than removed.** `SGDConfig`'s `momentum=0.9` is the
  right default for `main.py`; the fault was a target config that silently accepted a
  default for a field the caller is obliged to specify. The fix belongs in
  `build_target_config`'s signature, which should require the arm rather than permit its
  omission.

## Amendment (2026-08-07): equation cells carry the arm too

`ARM_IN_CELL_NAME` originally held `reference` alone, on the reasoning that a curve cell's
`source_id` (a W&B run id) and an equation cell's (a condition slug) each identify one arm
already. That is true of the run id and **false of the condition slug**: a *condition* is
`(dataset, eps, T, arch)` and ADR 0016 scopes the arm to the *synthesis*, not the
condition, so the two arm-scoped fits distil the same 32 conditions under the same
category indices. Both arms' equation cells therefore resolved to one filename — and the
skip filter, finding it, would have reported "already done" and silently never run the
second arm's 12 cells. Exactly the failure this ADR fixed for references, in the one
producer whose `source_id` was assumed to be safe.

`equation` joins `reference` in `ARM_IN_CELL_NAME`. The arm is *not* added to the producer
command line: `transfer_equation` already derives it from its `--eval_dir` manifest, so
the launcher reads the same file through the same `transfer_launch.synthesis_arm` (moved
there from `transfer_equation` for that reason) rather than passing a second, forgeable
copy. One equation launch therefore covers one arm; both arms means two launches, one per
eval dir.

No cells are renamed by this: the equation stage had not yet run when it was found.
