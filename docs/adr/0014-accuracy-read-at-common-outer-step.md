# Accuracy read at a cell's common outer step

## Context

`_fetch_one_run` records `learned_acc = history[-1]["test_acc"]` — each run's **last
logged** outer step, whichever step that happens to be
(`compile_results_fetch.py:670`). The run query includes partial runs:
`filters={"state": {"$in": ["crashed", "finished"]}}` (`:755`).

Runs do not all reach the same step. Steps reached within a single arch rung on the
cached CIFAR sweep:

| rung | min | max |
|---|---:|---:|
| `mlp-128` | 879 | 1,482 |
| `mlp-256` | 25 | 824 |
| `mlp-128x128x128x128` | 198 | 1,375 |

So a rung's seeds are averaged over different amounts of training, and the forest
plot's min–max bar — which exists to keep single-seed collapses visible — renders
that difference as seed variance. A seed killed by the scheduler at step 25 and a
seed that genuinely diverged to chance produce the same dot. `n` additionally depends
on *when* the fetch runs, because `running` runs are excluded and a chained run passes
through `running` between hops.

The aggregation itself is correct: `n="count"` is per-cell and the CI uses each row's
own `n` (`compile_results_plot.py:198-202`). The defect is that the quantity being
averaged is not the same quantity across seeds.

## Decision

Read each cell's accuracy at the **minimum outer step common to its seeds**, not at
each run's own final step. Record `final_outer_step` per run in the scalars schema, and
report the read-off step and the true per-cell `n` in every figure caption (replacing
the hardcoded `"n=8 seeds"` / `"n=5 seeds"` at `compile_results_plot.py:352,412`).

`histories.parquet` already stores per-step rows, so this needs no re-fetch.

## Considered options

**Require all runs to have finished; exclude crashed ones.** Rejected: a rung where a
seed genuinely diverges is a *result*, and the arch axis exists to detect exactly that
kind of instability. Dropping the seed suppresses the finding and biases every rung
upward by conditioning on success.

**Keep final-step reads and annotate the disagreement.** Rejected as the primary
mechanism: it uses all available training, but leaves the headline numbers
incomparable and relies on a reader noticing the annotation. Recording
`final_outer_step` keeps this option available for diagnostics.

## Consequences

- **Truncation is the price.** At a rung where one seed lags, every other seed is cut
  back to it and the figure understates all of them. With job chaining fixed
  (2026-07-11) and a 1,000-step budget, cells should mostly agree — but when they do
  not, the caption must say which step was used, or the understatement is invisible.
- **Every cached scalar changes meaning.** Numbers fetched before this are final-step
  reads; after, they are common-step reads. As with ADR 0010 and 0011, this is accepted
  because the arch axis is being re-run in full and the caches rebuilt.
- **A diverged seed stays in the plot** as a low dot that means "diverged by step *k*"
  rather than one that might mean "was killed at step 25." That is the whole point.
- **The Learned side stays 1-rep while baselines are 8-rep.** The baseline artifact also
  holds an 8-rep `"Learned Schedule"` evaluation, written by the final `log_comparison`
  (`main.py:301`) — but only when the run did *not* stop for a chain hop, so it is
  absent for exactly the partial runs this ADR exists to handle. It is therefore
  recorded as a separate `learned_acc_8rep` column rather than becoming `mean_acc`,
  giving a completed-run subset on which the two reads can be checked against each
  other. Until that check is done, the forest plot's min–max bar carries inner-run
  evaluation noise on the Learned side that the baseline side has averaged away.
- Does not fix fetch-timing sensitivity of `n` — a chained run mid-hop is still absent.
  `final_outer_step` makes it *diagnosable* (an absent seed is now distinguishable from
  a short one) but the fetch still reports whatever W&B holds at that moment.
