# Transfer evaluation: two independent shape-producers over a shared eval core, joined only when both exist

Policy-transfer evaluation is built as **one evaluation core, multiple producers, one
assembler**. The core is the shared downstream for every cell: produce a schedule
**shape** → resample/evaluate it at the target's T → `project()` onto the *target's* GDP
budget → run forward-only DP-SGD for N seeds → record top-1 accuracy. Two producers feed
it independently:

- **Curve transfer** — resample a source run's raw μ-weight curve to the target T by
  linear interpolation over normalized step, then project. Runs as a source × target
  sweep on SLURM.
- **Equation transfer** — evaluate the SR-distilled universal shape `f(step_norm)` on the
  target's step grid, then seat on the target budget. Because the template's per-condition
  constants are indexed by `(dataset, ε, T, arch)` and are *not* a function of ε/T, this
  runs only at a target `(ε, T)` that exactly matches a trained source condition (borrowing
  that condition's constants). See the equation-transfer consequence below.

Each producer, plus the three native references (Constant, DynamicDPSGD,
StatefulMedianGradient), writes result records under a shared schema — `(path, source_id,
target, regime, per-seed accuracies)` — to a parquet store under `cache/transfer/`. A
`transfer_plot.py` assembler reads whatever is present, builds the descriptive
source × target matrix (annotated with the nearest-`(ε, T)` source per column), and draws
the curve-vs-equation overlay for a cell **only when both records exist**, rendering each
alone otherwise. The unit of a SLURM job is one cell (seeds evaluated **sequentially**
inside, reusing `Baseline`'s existing per-rep `for` loop — chosen over a vmap so the eval
harness has no chance of OOM on the large targets, at the cost of wall-clock).

## Status

accepted

## Why

Two hard requirements drove the shape. **Independence** — equation transfer depends on the
slow SR pipeline, so curve transfer must run and be usable without it; making the two
paths *producers of a shared schema* means neither blocks the other. **Compare-only-when-
both-exist** — falls out for free as a presence-check join in the assembler, rather than
being an orchestration special-case. Per-cell jobs give embarrassing parallelism across
the available Compute Canada fan-out, and the fetch/plot split mirrors the existing
`compile_results_fetch` / `compile_results_plot` separation.

## Considered and rejected

- **One monolithic job looping all cells.** Simplest code, but serial (wastes the compute)
  and *couples the two paths* — curve transfer could not run before the equation pipeline
  finished, violating the independence requirement.
- **A W&B run per cell** (reusing `compile_results_fetch` verbatim). Rejected: spins up
  hundreds of heavyweight tracked runs for one-shot evaluations; a parquet store under the
  existing `cache/` convention is lighter and sufficient.

## Consequences

- The two ADRs share the "universal shape, re-instantiated elsewhere" premise, but this
  ADR does **not** realise ADR 0006's stage-2 generalisation. Read-back is now known to be
  fully possible cross-process — `symbolic_regression_eval._TemplatePredictor` reconstructs
  the closed form (shape `f` + inlined per-condition constants) from `equations.csv` +
  `category_map.json` in pure numpy, so the pickled-Julia-closure limitation blocks only
  warm-start, not evaluation. **But** the template constants are indexed by discrete
  condition `(dataset, ε, T, arch)` and are not a function of ε/T, so the closed form is
  undefined at an unseen `(ε, T)`. Equation transfer therefore runs **only at a target
  `(ε, T)` that exactly matches a trained source condition**, borrowing that condition's
  constants; every source condition present at that `(ε, T)` is transferred (read off, not
  selected), instantiating the single `selected` Pareto equation `f` with each condition's
  constant vector. The within-`(ε, T)` spread across conditions is the equation analog of
  generalization consistency. Genuinely off-grid equation transfer still needs the deferred
  stage-2 regression `p[condition] ~ g(ε, T, arch)` composed into `f` (ADR 0006); the
  pooled-scalar fit (which *does* take ε/T as inputs) was rejected there as not a clean
  transferable law. Because `f` is closed-form over `step_norm`, the producer *evaluates* it
  on the target step grid rather than resampling a length-T array, but otherwise feeds the
  identical `seat_on_budget` + eval core as curve transfer.
- Matched privacy is exact by construction, but requires an explicit **scale-to-boundary**
  step, not just `project()`. Under DP-PSAC the σ curve alone carries the budget; the
  resampled σ shape is scaled so `∑ᵢ exp(1/σᵢ)` *binds* the target's `(μ/p)² + T` exactly
  (a monotonic 1D root-find), because `project_inverse_sigmas` only enforces the inequality
  and passes an already-feasible curve through untouched — which would let a stricter
  source regime's absolute noise level leak in. Only the dimensionless shape crosses the
  target boundary; its magnitude is 100% set by the target ε.
- The descriptive matrix is **read off, not selected from** — at *both* ends. No per-target
  winner is picked by target accuracy, and no per-regime source *representative* is picked by
  source accuracy: **every** source policy (all seeds) is transferred. Selecting a source by
  its accuracy would bias toward source-overfit shapes that transfer worst, so instead the
  spread of transfer accuracies within a regime is reported as its **generalization
  consistency**. A best/median/worst-transferred-policy triple per target column is
  permitted for *shape inspection* only; any "best transfer" headline still requires a
  held-out target split.
- The result schema is the integration contract between producers and assembler; adding a
  new producer (e.g. a fourth reference) is just another writer of the same schema.
