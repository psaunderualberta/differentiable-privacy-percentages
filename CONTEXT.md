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

**Truncated run**:
A run whose outer loop stopped before the requested budget — it reached a smaller
**read-off step** than asked for, but everything it did reach is valid. Truncation is a
wall-clock property, not an outcome: it is systematic per rung (the expensive rungs
truncate, every seed of them together), so excluding truncated runs deletes whole rungs
rather than removing noise. Because the outer loop converges early, a truncated run's
schedule is normally a converged learned schedule and is safe to analyse.
_Avoid_: incomplete run, failed run, diverged run.

**Diverged run**:
A run whose outer loop lost its footing — the schedule blows up and downstream accuracy
lands at or near chance. An *outcome* property, and the one that makes a run unfit to
analyse. Independent of truncation: a run can diverge having reached the full budget, and
a truncated run is usually fine. Detected by chance-level accuracy; **not** by a
non-finite check (a diverged run's σ and accuracy are finite) nor by a σ-magnitude
outlier test (which also flags healthy runs with a boundary spike).
_Avoid_: crashed run, NaN run, truncated run, unstable run.

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

### Symbolic regression

**Synthesis**:
One symbolic-regression fit, identified by the problem it solves — which runs, which
filters, which search space — paired with a single target (σ or clip). Two invocations
describing the same problem are the same synthesis and may continue each other's search;
different problems must never share one. A synthesis is scoped to a single **arm**, since
a **condition** does not name one and the arms' schedules differ by roughly an order of
magnitude in scale.
_Avoid_: run (reserved for the W&B object), fit, job, regression.

**Universal schedule shape**:
The single form over training progress `t/T` that a synthesis fits across every run it
covers — the transferable part of the result. Everything a condition does differently is
carried by its **per-condition constants**, not by a different shape.
_Avoid_: curve, law, equation (reserve for the concrete fitted expression), template.

**Per-condition constant**:
One of the K free values fitted per **condition** that modulate the universal schedule
shape. They are a property of the condition, so the several seeds of a condition are
pooled into one constant vector rather than getting their own. They have no value for an
unseen condition, which is what confines **equation transfer** to a target `(ε, T)` that
exactly matches a trained one.
_Avoid_: parameter (overloaded with network parameters), coefficient, knob.

**Compression claim / Generalization claim**:
The two distinct things a synthesis can be said to establish, validated by different
evidence and not interchangeable. The **compression claim** — every learned schedule in
this sweep is the universal shape with K knobs — is descriptive and is settled by fit
quality on the runs fitted. The **generalization claim** — the distilled equation is a
useful schedule on data it was not fitted to — is settled only by downstream **policy
transfer** accuracy. A synthesis can satisfy either without the other.
_Avoid_: calling in-sample fit quality "generalisation", accuracy, validation.

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

The **arm** is part of a source regime's identity even though `_REGIME_COLUMNS` omits
it: the two arms' schedule shapes differ enough that pooling them turns generalization
consistency into a measure of arm separation instead.
_Avoid_: using "regime" for the three Constant / DynamicDPSGD / StatefulMedianGradient
schedules — those are **transfer references**.

**Transfer matrix**:
The full descriptive source × target-dataset grid of matched-privacy downstream
accuracies. Read off, not selected from; no per-target winner is picked by target
accuracy. **Every** source policy in scope is transferred — there is no best-of-regime
selection step (selecting on a source accuracy number would bias toward source-overfit
shapes that transfer worst). Its **row unit is the source regime-arm**, not the source
policy: a cell pools the regime's policies so its ± is that regime's **generalization
consistency**. `transfer_plot.policy_matrix` renders the same cells split by policy,
where the ± is evaluation noise instead — a diagnostic companion, not the matrix.

A native reference has no regime (its source provenance mirrors its target), so its
row unit is the reference itself; `source_label` carries whichever applies.

Which policies are *in scope* is ADR 0018: the T-sweep arch axis, both arms, regime-arms
carrying ≥4 seeds, capped at the four lowest seed indices. The cap is a subsample —
independent of every accuracy number — not the selection ADR 0008 prohibits.
_Avoid_: transfer grid, results table.

**Generalization consistency**:
The spread of transfer accuracies across the *source policies* of one regime-arm at one
target — how reproducibly that regime's learned shape transfers. Distinct from
**evaluation noise**, the spread across the evaluation reps of a *single* policy,
which measures only DP-SGD's own run-to-run variance. Both are standard deviations
over accuracy, so always name which one a bar or ± figure shows.
_Avoid_: spread (unqualified), variance, error bar.

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
- **StatefulMedianGradient** (Andrew et al., NeurIPS 2021) — a *runtime-adaptive*
  schedule that drives C toward the quantile target using the **within-clip fraction**,
  and sets σ = C/μ_grad.

**Candidate**:
One point in a transfer reference's 20-point random hyperparameter search, and the
unit its sweep is split into for SLURM (ADR 0019): each candidate is scored on its own
task at `SWEEP_SCORING_ITERATIONS` inner trainings, writing a **candidate record** —
an intermediate artifact under `<cache_root>/transfer_candidates/`, deliberately
outside the `transfer/` tree so it can never be read as a matrix cell. A **selector**
task then picks the winner by mean scored accuracy and runs the final evaluation, and
its output is the only `producer="reference"` cell.

Candidate enumeration is a pure function of (reference, key, index), so a candidate
scored in isolation is the one the monolithic sweep would have evaluated at that
position. Scoring and final evaluation use disjoint keys, so a winner's reported
number is never the draw that selected it.
_Avoid_: trial, sweep run, sample (reserve "sample" for minibatch sampling).

**Target regime**:
One **column** of the transfer matrix, and the unit `expand_targets` yields: the triple
(target dataset, ε, T). `plot_matrix` states it directly — "rows are source regime-arms,
columns are target regimes". Six of them is *not* six targets: the three budget points
inside a dataset share its data, surrogate architecture and accuracy floor, so they are
one dataset's worth of generalisation evidence at three budgets.
_Avoid_: "column" as a count of anything but target regimes — it collides with
**budget point**.

**Budget point**:
An (ε, T) pair, independent of dataset. The current grid is three of them: ε=10 with
T ∈ {2000, 5000, 7000}. Distinct from a **target regime**, which pairs a budget point
with a dataset. "Three columns" is the error this pair of terms exists to prevent —
three budget points across two datasets is *six* columns.

**Schedule-resolving power**:
Whether a target regime can measure a schedule at all: do differently-shaped schedules
run natively on it, at its own budget, separate by more than **evaluation noise**? It
is read off the three **transfer references**, which is the whole instrument — the
references contain no transferred policy, so a target admitted or rejected on this
criterion is judged without touching the transfer claim. A target that fails it returns
the same accuracy whatever schedule it is given, and every cell in its column is
measuring nothing (ADR 0007).

It has two parts, and they are not interchangeable. **Headroom** — does any schedule beat
the target's majority-class floor? — is *necessary*, and one native schedule measures it.
**Separation** — do differently-shaped schedules give different answers? — is *sufficient*,
and needs the full reference set. A target with no headroom cannot have separation, so
failing the first part settles the question on its own; passing it settles nothing.

It is a property of the **target regime**, not the dataset: the surrogate architecture
is part of the regime, so "no resolving power" always means "under this regime", never
"this dataset is unlearnable". Do **not** measure it as the gap between a non-private
control and a DP run — those two differ in hyperparameter tuning as well as in privacy,
so the difference confounds the mechanism with the search. Do not measure it with a
**transferred curve** either: that is the very thing the target is being qualified to
judge (ADR 0020).
_Avoid_: resolving power (unqualified), signal, sensitivity, dynamic range.

### Adaptive clipping

**Within-clip fraction**:
The per-step statistic b̄ — the fraction of the batch whose per-example gradient norm
falls at or below the current clip threshold C_t. The quantity the adaptive-clipping
schedule privatises and steers; the gradient *median* is the fixed point it converges
to, never a quantity that is estimated directly.
_Avoid_: median estimate, median gradient norm, clipped count.

**Quantile target (γ)**:
The value the within-clip fraction is driven toward by the geometric update
C ← C·exp(−η_C(b̄ − γ)). γ = 0.5 makes the fixed point the median norm.
_Avoid_: target quantile fraction, gamma.

**Count release**:
The privatised release of the within-clip fraction: the ±½-encoded indicator sum plus
Gaussian noise, divided by the *expected* batch size L. Jointly Gaussian with the
gradient release over the same batch, so the two share one per-step budget.
_Avoid_: quantile mechanism, b-bar noise.

**Count noise ratio (r)**:
The knob controlling how much of the per-step budget the count release consumes,
expressed the way Andrew et al. express it: the count's noise is one *r*-th of the
expected batch size. Because the release is divided by that same expected batch size,
r fixes the *standard error of the within-clip fraction* (1/r) independently of the
privacy regime — the property that makes a single default value meaningful across the
whole (ε, T) grid. Andrew's default is r = 20.
_Avoid_: sigma_b, count noise, quantile noise.

**Median budget fraction (ρ)**:
The share of the per-step GDP budget spent on the count release rather than the
gradient release: μ_count² = ρ·μ₀² and μ_grad² = (1−ρ)·μ₀². **Derived, not chosen** —
it follows from the count noise ratio and the regime. Reported as the diagnostic that
says how much privacy honest adaptivity cost; the thing that makes the adaptive
baseline genuinely (ε,δ)-DP at an unchanged total budget.
_Avoid_: privacy split, epsilon share, noise fraction.
