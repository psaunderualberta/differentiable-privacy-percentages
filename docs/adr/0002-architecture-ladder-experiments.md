# Architecture ladder experiments: isolated shape sweeps via deduped ladder tags

## Context

We want to study how network *shape* affects the learned DP-SGD schedule, separating
**width** from **depth** — the previous `arch-sweep` confounded both (it mixed
`(16,)` against `(512,256)`). We replace that single sweep with five **ladders**, each
holding all shape properties constant except one: `mlp-width`, `mlp-depth` (fixed
width), `mlp-depth-pm` (param-matched to the depth-1/width-128 **anchor**), `cnn-width`,
and `cnn-depth`. Ladders are generated from knobs in `experiments/architectures.py`
(a `LADDERS` registry), not hand-enumerated, so the param-matched widths are computed
from the anchor rather than transcribed.

## Decisions and trade-offs

**Redesigned CNN block for the depth ladder.** The default CNN block downsamples
aggressively (k8/s2 + pool) and collapses 28×28 to 1×1 after only 2 conv layers, so it
cannot express a depth ladder. The `cnn-depth` ladder therefore uses a **same-conv
block** (3×3, pad 1, stride 1, pool 2) that reaches depth 4. Consequence: the CNN width
and depth ladders use *different* block geometries, so they are comparable only
*within* a ladder, not across. We accept this — the alternative (one block for both)
makes a depth ladder impossible on 28×28.

**Deduped multi-tag membership.** The `(128,)` anchor belongs to three MLP ladders. We
create each unique architecture **once** and attach one `ladder:<name>` W&B tag per
ladder it belongs to, rather than duplicating runs per ladder. This avoids ~3× anchor
reruns at the cost of a multi-valued membership representation downstream.

**Boolean membership columns downstream, discovered from the tag prefix.**
`compile_results_fetch.py` emits one boolean column per ladder (`in_mlp_depth`, …),
generated dynamically from any `ladder:`-prefixed tag. A single-valued `axis` column
could not represent the anchor's multiple memberships. The old `_T_SWEEP_T_VALUES`
fallback in `_axis()` is retired: it inferred the axis from `T`, which would mislabel
every `T=5000` ladder run as `T-sweep`.

**Breadth.** Ladders run at **ε=8 only** (2 datasets × 3 seeds), an exploratory first
look; the T-sweep keeps full ε breadth. ε∈{1,3,5} can be backfilled on the ladders later.
