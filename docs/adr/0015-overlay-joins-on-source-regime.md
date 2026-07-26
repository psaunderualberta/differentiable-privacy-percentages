# The curve-vs-equation overlay joins on the source regime, not the source policy

ADR 0008 specified the curve-vs-equation comparison as a presence-check join over
"the same source×target cell", and `transfer_plot.py` implemented that literally, keying
the join on `source_id`. But the two producers do not share a `source_id` namespace:
curve transfer's is a W&B run id (one learned policy), equation transfer's is a condition
slug `f"{dataset}_eps{eps:g}_T{T}_{arch}_cat{n}"`. They can never compare equal, so
`overlay_cells` returned `[]` unconditionally and ADR 0008's headline figure could not be
produced at all. We therefore join on the **source regime** —
`(source_dataset, source_eps, source_T, source_arch)` plus the target keys — and aggregate
the curve side across its seed-policies before comparing.

## Status

accepted (amends ADR 0008)

## Why

The mismatch is not a naming accident, it is a **granularity** difference, and it is
inherent. The curve producer's row unit is one seed's learned policy; the equation
producer's is a whole distilled condition, whose per-condition constants were fitted
across every run in that condition at once. A distilled condition genuinely has no
per-seed identity to recover.

The source regime is the coarsest key both producers *do* carry natively (`SourcePolicy`
records `dataset/eps/T/arch` on every row of the shared schema, for both producers), and
it is exactly the granularity at which a condition is defined. Joining there needs no new
provenance from either side.

## Considered and rejected

- **Make the equation producer emit per-policy rows** — one row per source run in the
  condition, all sharing the condition's constants. Rejected: it would fabricate a
  per-seed identity that the distilled object does not have, and it would silently
  duplicate identical accuracies under N different `source_id`s, corrupting the seed-spread
  that ADR 0008 reports as generalization consistency.
- **Join on `(target, target_eps, target_T)` only**, pooling all sources. Rejected: it
  collapses the per-regime structure that is the point of the matrix, and would compare a
  curve mean over unrelated source regimes against one condition's equation.

## Consequences

- **The figure asserts something different, and weaker.** It is no longer "this policy
  versus its distilled form" but "**this regime's policies, pooled, versus their distilled
  form**". The curve-side error bar is now a mixture of seed noise *and* across-policy
  spread within the regime, while the equation side's is seed noise alone; the two error
  bars are therefore not like-for-like and must not be read as a significance test. What
  the overlay supports is the intended qualitative claim — whether distillation preserved
  the transferable shape — not a per-policy win/loss.
- The join key `source_arch` is a *source*-side field. Equation transfer's condition
  already carries it, and curve transfer reads it from `schedules.parquet`'s `arch_label`,
  so both sides populate it without change.
- `target_arch` is deliberately **not** a join key: it is derived from the target dataset
  by `AutoNetworkConfig` (ADR 0007), so it is functionally determined by `target` and adds
  nothing but a chance of spurious mismatch on the label string.
- Reference cells still never participate: they are a third producer, and the presence
  check is over `curve` and `equation` only.
