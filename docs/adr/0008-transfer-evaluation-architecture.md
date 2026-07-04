# Transfer evaluation: two independent shape-producers over a shared eval core, joined only when both exist

Policy-transfer evaluation is built as **one evaluation core, multiple producers, one
assembler**. The core is the shared downstream for every cell: produce a schedule
**shape** → resample/evaluate it at the target's T → `project()` onto the *target's* GDP
budget → run forward-only DP-SGD for N seeds → record top-1 accuracy. Two producers feed
it independently:

- **Curve transfer** — resample a source run's raw μ-weight curve to the target T by
  linear interpolation over normalized step, then project. Runs as a source × target
  sweep on SLURM.
- **Equation transfer** — evaluate the SR-distilled closed-form σ/C at the target's T,
  then project.

Each producer, plus the three native references (Constant, DynamicDPSGD,
StatefulMedianGradient), writes result records under a shared schema — `(path, source_id,
target, regime, per-seed accuracies)` — to a parquet store under `cache/transfer/`. A
`transfer_plot.py` assembler reads whatever is present, builds the descriptive
source × target matrix (annotated with the nearest-`(ε, T)` source per column), and draws
the curve-vs-equation overlay for a cell **only when both records exist**, rendering each
alone otherwise. The unit of a SLURM job is one cell (seeds vmapped inside).

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

- This realizes the **stage-2 generalisation** deferred in ADR 0006: equation transfer is
  how the fitted template shape is finally exercised on unseen conditions. The two ADRs
  share the "universal shape, re-instantiated elsewhere" premise.
- Matched privacy is exact by construction, but requires an explicit **scale-to-boundary**
  step, not just `project()`. Under DP-PSAC the σ curve alone carries the budget; the
  resampled σ shape is scaled so `∑ᵢ exp(1/σᵢ)` *binds* the target's `(μ/p)² + T` exactly
  (a monotonic 1D root-find), because `project_inverse_sigmas` only enforces the inequality
  and passes an already-feasible curve through untouched — which would let a stricter
  source regime's absolute noise level leak in. Only the dimensionless shape crosses the
  target boundary; its magnitude is 100% set by the target ε.
- The descriptive matrix is **read off, not selected from**; no per-target winner is picked
  by target accuracy, avoiding test-set selection bias. A "best transfer" headline, if
  wanted, must come from a held-out target split.
- The result schema is the integration contract between producers and assembler; adding a
  new producer (e.g. a fourth reference) is just another writer of the same schema.
