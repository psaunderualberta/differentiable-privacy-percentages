# Fit a universal schedule shape with per-condition free constants, not a pooled global equation

Symbolic regression now defaults to a PySR `TemplateExpressionSpec` that learns one
**universal schedule shape** `f(step_norm)` shared across all runs, modulated by a small
set of free **per-condition constants** `p1, p2, … pK` (default K=3, configurable via
`n_template_params`). The category indexing the constants is the **condition**
`(dataset, eps, T, arch_label)` — the 3 replicate seeds of a condition collapse into one
constant vector. This replaces the previous pooled fit, which expressed σ/C directly as a
single global function of the existing features (`eps, T, step_norm, arch_param_count,
in_mlp_*`) and was neither expressive enough nor a clean, transferable schedule law.

## Status

accepted — template mode is the default (`template_mode=True`); the pooled scalar fit
remains available via `--template_mode False`.

## Why

Within a single run the only varying input is `step_norm` (t/T); every other feature is
constant. A pooled global equation must therefore contort one functional form to cover
all conditions at once. Splitting the model into a shared shape plus a few per-condition
constants lets `f` capture the transferable shape over training progress while the
constants absorb each condition's scale/offset/curvature. The constants double as
interpretable **schedule hyperparameters**: the result reads as "every learned schedule
is `f(t/T)` with K knobs."

## Considered and rejected

- **Category = raw timestep.** Up to 3000 free constants per slot — pure memorisation of
  the schedule, defeating both expressiveness-sharing and generalisation. (Binned phases
  would be coherent but answer a different question — a law over conditions, not over
  time.)
- **Category = run_id (per seed).** Triples the constant count and fits seed noise; the
  constants should be a property of the condition, so seeds are pooled as repeated
  observations instead.
- **Run-level features as inputs to `f`.** Redundant with the per-condition constants
  (they are constant within a condition), so they double-count and muddy interpretation.
  They are reserved for the deferred stage-2 regression.

## Consequences

- **Generalisation is deferred to a (future) stage-2.** Per-condition constants have no
  value for an unseen condition. The closed-form generalising schedule would come from a
  second regression `p[condition] ~ g(eps, T, arch)` composed into `f`. This work only
  **persists the fitted constants** (`constants.csv`) for that future effort — gated on
  verifying PySR exposes the fitted template parameters for read-back.
- **`sr_identity.py` must list the new fit-defining fields.** `template_mode` and
  `n_template_params` change the fit, so they are added to `IDENTITY_FIELDS` /
  `IDENTITY_FLAG_DEFAULTS` (and the launcher's `SRSlurmConfig`); omitting them would let a
  template synthesis and a scalar synthesis with identical filters share a slug / run
  directory and corrupt each other's PySR `warm_start` state (see ADR 0005).
- **The evaluator must reconstruct the category index.** `symbolic_regression.py`
  persists a `category_map.json` (condition → 1-indexed integer); the in-sample evaluator
  rebuilds the `category` column on `features_full` before `model.predict`.
- **Slower fits.** ~62 conditions × K constants are re-optimised by BFGS for every
  candidate equation; scope a synthesis down with the existing `--datasets` /
  `--arch_labels` filters when this bites.
