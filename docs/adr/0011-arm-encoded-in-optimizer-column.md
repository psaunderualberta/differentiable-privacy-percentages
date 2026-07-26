# Arm encoded in the optimizer column

## Context

The private network's inner SGD momentum is being varied as an **arm** — a condition under
which the entire axis matrix (T-sweep and arch) is replicated. Analyses and plots must
never pool across arms.

Historically the two arms were separated by W&B *project* (`MomentumSweep` vs
`NoMomentumSweep`), which kept them apart by accident of storage rather than by any
property recorded on the run. Both projects' `optimizer` column reads `sgd` for all 1,872
and 1,856 rows respectively. Merging them into one project — which is what running both
arms from a single `OPTIMIZERS` list does — makes them indistinguishable downstream:

- `create_experiments._opt_tag` returns `type(opt).__name__.removesuffix("Config").lower()`,
  so `SGDConfig(momentum=0.9)` and `SGDConfig(momentum=0.0)` both tag `"sgd"`. That tag
  feeds `_group_label`, so both arms land in one W&B group.
- `compile_results_fetch.resolve_optimizer` keys off the config dict's `_type` alone, so
  momentum is discarded before the `optimizer` column is written.
- `compile_results_plot` then groups on that column and writes `out/<optimizer>/`, so both
  arms would overwrite one plot tree with the arm difference invisible in every figure.

The plotting layer is already generic over the column — it iterates
`sorted(scalars["optimizer"].unique())` and exposes an `--optimizers` filter. It needs no
change; only the two places that *collapse* the distinction do.

## Decision

Encode the arm as a suffix on the existing `optimizer` column: `sgd-m0.9` / `sgd-m0.0`.
Fix `_opt_tag` and `resolve_optimizer` to derive the suffix from
`env.optimizer.momentum.value`.

The suffix applies to SGD only. `AdamConfig` / `AdamWConfig` carry no momentum field and
stay bare `adam` / `adamw`. Both must remain in `resolve_optimizer` regardless of not being
in the new sweep: it is the *fetch* path, and `FixedParallelSweep` (1,816 rows) and
`ParallelSCSweep` (1,920 rows) contain Adam and AdamW runs that would fail to re-fetch.

The suffix is only well-defined when `momentum.distribution == "constant"`. A swept
`DistributionConfig` gives a per-run continuous value that cannot serve as a categorical
split, so this is asserted rather than left to be discovered in a plot.

## Consequences

- **Every cached run is relabelled.** Existing data resolves to `sgd` and lands in a
  different directory than new data. This is accepted because ADR 0010 already invalidates
  the CNN results and forces a full re-run of the arch axis — the caches are being rebuilt
  anyway, making this the cheapest moment in the project's life to break the label.
- Splitting on a hyperparameter rather than on optimizer *identity* overloads the column's
  meaning. The alternative — a separate `arm` column with the plot tree keyed on
  `(optimizer, arm)` — is more principled, but requires changing the plot layer's output
  paths, its filter flag, and every downstream reader in order to express something the
  existing column already can. Revisit if a second arm dimension is ever added, at which
  point the suffix stops scaling.
- The same defect existed in two files that had to agree without any shared code. They
  still do; `_opt_tag` and `resolve_optimizer` are independent by design (the fetch script
  stays decoupled from training code, as with `_LADDER_TAG_PREFIX`). The naming scheme is
  the contract between them.
- Fixes a latent bug beyond this arm: the commented-out `AdamConfig` / `AdamWConfig` lines
  in `OPTIMIZERS` differ only by learning rate, and would have collided the same way.
