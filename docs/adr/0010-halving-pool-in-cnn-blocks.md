# Halving pool in CNN blocks

## Context

`CNN.from_config` built `eqx.nn.MaxPool2d(kernel_size=conf.pool_kernel_size)` without a
stride. Equinox defaults `MaxPool2d` to **stride 1**, so the pool shrank each spatial
dimension by one pixel instead of halving it. Every CNN in the project therefore had a
geometry that differed from its intended design, silently:

- `cnn-8x16-head64` on CIFAR-10 traced as `(3,32,32) → (8,14,14) → (16,5,5) → flatten
  (400,)`, giving **29,922** parameters against the **5,474** intended.
- The *same-conv block* underpinning the `cnn-depth` ladder (ADR 0002) did essentially no
  downsampling: a depth-*d* chain of size-preserving 3×3 convs plus stride-1 pools output
  `(32−d)×(32−d)`. The ladder's rungs all sat at 1–2M parameters, dominated by a
  flatten→head layer that barely changed with depth — so it measured almost nothing about
  depth, and was the most likely cause of memory exhaustion.
- `compile_results_fetch._cnn_param_count` models the *intended* halving geometry
  (`h //= pool_k`), so recorded `arch_param_count` disagreed with the real network by ~5×,
  non-uniformly in the number of pools.

## Decision

Pass `stride=conf.pool_kernel_size` explicitly, making the pool halving. This is the
architecture the ADR 0002 ladder design, the `CONTEXT.md` glossary, and the fetch-side
parameter counter all already assumed.

## Consequences

- **Prior CNN results are invalidated for comparison.** Every existing `cnn-width` run on
  MNIST / fashion-MNIST / CIFAR-10 trained a different architecture than the one the fixed
  code builds. The arch axis must be re-run in full to stay internally consistent; results
  from before this change cannot be mixed with results after it.
- `_cnn_param_count` needs no change — it was always right about the intent. A guard
  asserting it against a built model (mirroring `assert_shapes_consistent` for
  `DATASET_SHAPES`) is added so code and reporting cannot drift apart again.
- 32×32 inputs now support a genuine depth-5 ladder (32→16→8→4→2→1), where 28×28 caps at 4.
  The capability is deliberately **not used**: at ch=16 the depth-4 and depth-5 rungs are
  12,346 and 11,594 parameters — 6% apart — because once the feature map reaches 1×1 the
  flatten layer stops shrinking, so the extra rung is one more conv over almost no spatial
  extent rather than a meaningfully deeper network. Capping the ladder at depth 4 also keeps
  it identical across all three datasets, so the overlay's forest blocks stay the same
  length per dataset. (At ch=32 and ch=64 the effect is stronger still and parameter count
  *reverses* at d=5 — 37,674→40,778 and 129,802→154,442 — which is why the ladder uses
  ch=16.)
- Per-run memory for the depth ladder drops by roughly an order of magnitude, which is what
  makes `cnn-depth` runnable at all.
