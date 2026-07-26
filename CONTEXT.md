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
**arch** axis (fix T, vary network shape across ladders). A run belongs to exactly one.
_Avoid_: dimension, sweep-type, arm.

**Arm**:
A condition under which the *entire* axis matrix is replicated, so every arm contains a
full copy of both axes — currently the private network's inner SGD momentum (0.9 vs 0.0).
Distinct from an axis, which a run belongs to one of; a run belongs to one axis *and* one
arm. Analyses and plots are split per arm, never pooled across them. Previously separated
by W&B project (`MomentumSweep` / `NoMomentumSweep`); now carried on the run itself.
_Avoid_: axis, condition (reserved for the symbolic-regression sense), variant, branch.

**Inner / outer momentum**:
Two unrelated settings that the word "momentum" alone does not distinguish. **Inner**
momentum belongs to the private network's DP-SGD optimizer (`OPTIMIZERS`); **outer**
momentum belongs to the schedule optimizer (`ScheduleOptimizerConfig.momentum`), which
runs as plain GD. The **arm** varies the *inner* one. Legacy project names beginning
`NoMomentum` refer to the *outer* one and mean the opposite of what they appear to.
_Avoid_: bare "momentum".

**Ladder**:
An ordered family of network architectures that holds everything constant except one
shape property, so changes in the learned schedule can be attributed to that property.
Each ladder has a name (e.g. `mlp-depth`) and is generated from knobs, not enumerated by hand.
_Avoid_: sweep, group, series.

**Rung**:
A single architecture within a ladder — one position on the ordered family. Per-ladder
plots use a categorical x-axis over rungs, ordered by the ladder definition (`LADDERS`)
and ticked by `arch_label`. A rung may in principle belong to more than one ladder, in
which case it is run once and carries a `ladder:` tag per membership; no current ladder
pair shares one.
_Avoid_: point, step, level.

**Width ladder**:
A ladder that fixes depth (number of layers) and varies layer width.
_Avoid_: size sweep.

**Depth ladder**:
A ladder that fixes per-layer width (or channel count) and varies depth (number of
layers). Total parameters are *not* held constant along it, so a difference between rungs
is attributable to architecture but not specifically to depth — see **Overlay** for why
that is sufficient here. On the CNN depth ladder the parameter count in fact *falls*
with depth and is dominated by the flatten→head layer (99.6% of the depth-1 rung, 34%
of depth-4), because each block halves the feature map the head reads from. The shallow
rungs are therefore closer to "an MLP on pooled conv features" than to shallow CNNs, and
should be described that way rather than as a clean depth contrast.
_Avoid_: layer sweep.

**Same-conv block**:
The redesigned CNN convolutional block used only by the CNN depth ladder: 3×3 kernel,
padding 1, stride 1 (spatially size-preserving), with a **halving** MaxPool(2) — pool
stride equal to pool kernel — carrying all downsampling. Each block therefore halves
each spatial dimension, capping the ladder at depth 4 on 28×28 and depth 5 on 32×32.
_Avoid_: standard block, conv layer.

**Halving pool**:
A MaxPool whose stride equals its kernel size, so each spatial dimension is divided by
the kernel. Named explicitly because the framework default is stride 1, which pools
without downsampling and silently makes every ladder's geometry differ from its
intended design (see ADR 0010).
_Avoid_: pool, downsample, stride-2 pool.

### Run metadata

**Ladder tag**:
A W&B run tag of the form `ladder:<name>` (e.g. `ladder:mlp-depth`). A single run carries
one tag per ladder it belongs to, so a rung shared by several ladders is run once and
carries several. Downstream tooling discovers ladder membership generically from the
`ladder:` prefix.
_Avoid_: axis tag (reserved for the coarser T-sweep / arch distinction), label.

### Result plots

**Per-ladder plot**:
A figure showing a single ladder, with its rungs on a categorical x-axis. Lives under
`plots/<optimizer>/ladders/<ladder-name>/`. Each ladder gets its own scalar plots
(main, deltas, table) and schedule-shape-by-rung plots.
_Avoid_: arch plot, sweep plot.

**Read-off step**:
The outer step at which a cell's accuracy is taken — the minimum step common to that
cell's seeds, not each run's own final step. Named because runs within a rung reach
different steps, so "final accuracy" is not one quantity; reporting the read-off step
is what makes a rung's seeds comparable (see ADR 0014). Distinct from the *budget*
(`num_outer_steps`), which is the step count a run was asked for rather than the one
its slowest seed reached.
_Avoid_: final step, last step, convergence point.

**Overlay**:
The cross-ladder figure under `plots/<optimizer>/ladders/overall/`. It answers the
**robustness** question — does Learned beat Constant at *every* architecture — not a
scaling question (the T-sweep owns scaling). It is a **forest plot**: rungs on a
categorical y-axis, grouped into ladder blocks, with Δacc (or paired absolute acc) on
x. There is no continuous parameter-count axis, because the ladders vary different knobs
and are not comparable across a shared param axis (see ADR 0002). Two variants:
`arch_forest_delta` (Learned − Constant Δ, dashed line at 0, shared x across datasets)
and `arch_forest_abs` (paired Constant/Learned absolute acc, independent x per dataset).
Each rung shows its seeds as individual dots plus a mean marker and a min–max bar — the
raw seed spread, not a box summary, so single-seed collapses stay visible rather than
being absorbed into a quartile.

Because it is a robustness claim, no ladder holds parameter count constant: the reportable
statement is "Learned beats Constant at every architecture", never "deeper/wider networks
benefit more" — the latter would need a parameter-controlled ladder to separate the knob
from the parameter count it moves, and none is run.
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

Always qualify which end is meant. A **source regime** groups the several source
policies (one per seed) that were trained under one such tuple; a **target regime**
is one column of the transfer matrix — a target dataset paired with the budget it is
evaluated under. The two are independent: transferring a source policy onto a target
whose (ε, T) differs from its own is the cross-regime question the matrix exists to
answer. **Condition** is the symbolic-regression name for the same tuple (the key the
template's constants are indexed by), not a separate concept.
_Avoid_: using "regime" for the three Constant / DynamicDPSGD / StatefulMedianGradient
schedules — those are **transfer references**.

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
  sets σ/clip from per-step gradient-median statistics.
