# Differentiable Privacy Percentages

Research project that learns DP-SGD noise (σ) and clipping (C) schedules via a
gradient-based outer loop. This glossary fixes the vocabulary of the **structured
architecture experiments** (`create_experiments.py`, `experiments/architectures.py`,
`compile_results_fetch.py`) that probe how network shape affects the learned schedule.

## Language

### Experiment structure

**Axis**:
A top-level experimental sweep direction. There are two: the **T-sweep** (vary the
number of inner training steps, architecture fixed at the dataset default) and the
**arch** axis (fix T, vary network shape across ladders).
_Avoid_: dimension, sweep-type.

**Ladder**:
An ordered family of network architectures that holds everything constant except one
shape property, so changes in the learned schedule can be attributed to that property.
Each ladder has a name (e.g. `mlp-depth`) and is generated from knobs, not enumerated by hand.
_Avoid_: sweep, group, series.

**Anchor**:
The single architecture shared across multiple ladders, serving as their common
reference point — here the width-128 / depth-1 MLP, where the width and depth ladders meet.
The param-matched ladder's target parameter count is defined by the anchor.
_Avoid_: baseline, pivot, origin.

**Rung**:
A single architecture within a ladder — one position on the ordered family. Per-ladder
plots use a categorical x-axis over rungs, ordered by the ladder definition (`LADDERS`)
and ticked by `arch_label`. The anchor is a rung shared by several ladders.
_Avoid_: point, step, level.

**Width ladder**:
A ladder that fixes depth (number of layers) and varies layer width.
_Avoid_: size sweep.

**Depth ladder**:
A ladder that fixes per-layer width and varies depth (number of layers). The
**fixed-width** variant lets total parameters grow with depth; the **param-matched**
variant shrinks width as depth grows so total parameters stay ≈ the anchor's.
_Avoid_: layer sweep.

**Param-matched**:
Describes a ladder whose member architectures are tuned to hold total parameter count
approximately constant (equal to the anchor), isolating depth from parameter budget.
_Avoid_: size-controlled, normalized.

**Same-conv block**:
The redesigned CNN convolutional block used only by the CNN depth ladder: 3×3 kernel,
padding 1, stride 1 (spatially size-preserving), with a MaxPool(2) carrying all
downsampling. Lets conv layers stack to depth 4 on 28×28 inputs, unlike the default
aggressive-downsampling block.
_Avoid_: standard block, conv layer.

### Run metadata

**Ladder tag**:
A W&B run tag of the form `ladder:<name>` (e.g. `ladder:mlp-depth`). A single run carries
one tag per ladder it belongs to, so the deduplicated anchor run carries several.
Downstream tooling discovers ladder membership generically from the `ladder:` prefix.
_Avoid_: axis tag (reserved for the coarser T-sweep / arch distinction), label.

### Result plots

**Per-ladder plot**:
A figure showing a single ladder, with its rungs on a categorical x-axis. Lives under
`plots/<optimizer>/ladders/<ladder-name>/`. Each ladder gets its own scalar plots
(main, deltas, table) and schedule-shape-by-rung plots.
_Avoid_: arch plot, sweep plot.

**Overlay**:
The cross-ladder figure under `plots/<optimizer>/ladders/overall/`. It answers the
**robustness** question — does Learned beat Constant at *every* architecture — not a
scaling question (the T-sweep owns scaling). It is a **forest plot**: rungs on a
categorical y-axis, grouped into ladder blocks, with Δacc (or paired absolute acc) on
x. There is no continuous parameter-count axis, because the ladders vary different knobs
and are not comparable across a shared param axis (see ADR 0002). Two variants:
`arch_forest_delta` (Learned − Constant Δ, dashed line at 0, shared x across datasets)
and `arch_forest_abs` (paired Constant/Learned absolute acc, independent x per dataset).
Each rung shows its 3 seeds as individual dots plus a mean marker and a min–max bar — no
box plot, since n=3 is too few for a five-number summary.
_Avoid_: combined plot, lumped plot, arch-sweep plot, param-count overlay.

### Policy transfer

**Policy transfer** (a.k.a. **generalization**):
Instantiating a policy learned on a *source* dataset onto a different *target*
dataset's DP-SGD run — matched to the target's own privacy budget — and measuring
downstream accuracy. Distinct from re-running the outer loop on the target.
_Avoid_: porting, migration, domain adaptation.

**Source dataset / Target dataset**:
The source is where a policy was learned; the target is the held-out dataset the
policy is transferred to. Each target name denotes a **surrogate**, not the
canonical dataset (see ADR 0007): **EyePACS** (used as-is), **CheXpert** = a binary
Pleural-Effusion probe (not the multi-label dataset), **ImageNet** = the 100-class
ImageNet-100 subset at 32×32 (not 1000-class full-resolution).

**Transferred object**:
What actually gets instantiated on the target. Two independently-runnable kinds,
compared only when both exist:
- **Equation transfer**: evaluate the SR-distilled universal shape `f(step_norm)`
  on the target's step grid. The template's per-condition constants are *not* a
  function of ε/T, so equation transfer runs only at a target `(ε, T)` that
  **exactly matches** a trained source condition, borrowing that condition's
  fitted constants; every source condition at that `(ε, T)` is transferred (read
  off, not selected), and the σ shape is seated on the target budget the same way
  curve transfer is. Off-grid `(ε, T)` is not equation-transferable without the
  deferred stage-2 constant regression.
- **Curve transfer**: resample a source run's raw length-T schedule onto the
  target's T. Under DP-PSAC the noise-multiplier (σ) curve alone carries the
  privacy budget and is projected onto the target's; the clip curve is a
  privacy-neutral per-step learning-rate multiplier that is linearly
  interpolated and carried across as-is. Run as a source-policy × target-dataset
  sweep on SLURM.
_Avoid_: policy porting.

**Regime**:
The tuple characterizing where/how a policy was trained or evaluated — privacy
budget (ε, δ), inner step count T, network architecture, and dataset. Used to
annotate each transfer-matrix cell with which source curve was trained in the
regime closest to the target's, separating "transfers because the regime matched"
from "transfers despite regime mismatch."

**Transfer matrix**:
The full descriptive source-policy × target-dataset grid of matched-privacy
downstream accuracies. Read off, not selected from; no per-target winner is picked
by target accuracy. **Every** source policy is transferred — there is no best-of-regime
selection step (selecting on a source accuracy number would bias toward source-overfit
shapes that transfer worst). Rows are grouped by regime; the spread of transfer
accuracies *within* a regime is itself a reported signal — the **generalization
consistency** of that regime's learned shape.
_Avoid_: transfer grid, results table.

**Source policy**:
The row unit of the transfer matrix: one learned run's final length-T σ/clip
schedule, identified by its W&B `run_id`. Distinct from a regime, which groups the
several source policies (one per seed) that share a `(dataset, ε, T, arch)`.
_Avoid_: source run (reserve for the W&B object), representative.

**Best / median / worst transferred policy**:
A descriptive triple summarising a target column of the transfer matrix — the source
policies at the max, median, and min of per-policy mean target accuracy *across all
regimes*. Purely diagnostic: its purpose is **shape inspection** (plotting the three
actual σ/clip curves and their source regimes, the direct analog of inspecting the
best/median/worst SR equation), never a "best transfer" headline — that still requires a
held-out target split (ADR 0008). Chosen per target column, not per cell, because within
a single regime the seeds share a near-duplicate shape.
_Avoid_: winner, selected policy.

**Transfer reference** (a.k.a. **baseline**):
The three schedules a transferred cell is judged against, each run natively on the
target at the target budget. They are distinct and must not be conflated:
- **Constant** — best flat σ/clip found by a sweep.
- **DynamicDPSGD** (arXiv:2111.00173) — a *prescribed, closed-form* dynamic
  schedule; deterministic given its params, **not** data-adaptive at runtime.
- **StatefulMedianGradient** (NeurIPS 2021) — a *runtime-adaptive* schedule that
  sets σ/C from per-step gradient-median statistics.
