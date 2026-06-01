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
**arch** axis (fix T, vary network shape across ladders).
_Avoid_: dimension, sweep-type.

**Ladder**:
An ordered family of network architectures that holds everything constant except one
shape property, so changes in the learned schedule can be attributed to that property.
Each ladder has a name (e.g. `mlp-depth`) and is generated from knobs, not enumerated by hand.
_Avoid_: sweep, group, series.

**Anchor**:
The single architecture shared across multiple ladders, serving as their common
reference point — here the width-128 / depth-1 MLP, where the width and depth ladders meet.
The param-matched ladder's target parameter count is defined by the anchor.
_Avoid_: baseline, pivot, origin.

**Width ladder**:
A ladder that fixes depth (number of layers) and varies layer width.
_Avoid_: size sweep.

**Depth ladder**:
A ladder that fixes per-layer width and varies depth (number of layers). The
**fixed-width** variant lets total parameters grow with depth; the **param-matched**
variant shrinks width as depth grows so total parameters stay ≈ the anchor's.
_Avoid_: layer sweep.

**Param-matched**:
Describes a ladder whose member architectures are tuned to hold total parameter count
approximately constant (equal to the anchor), isolating depth from parameter budget.
_Avoid_: size-controlled, normalized.

**Same-conv block**:
The redesigned CNN convolutional block used only by the CNN depth ladder: 3×3 kernel,
padding 1, stride 1 (spatially size-preserving), with a MaxPool(2) carrying all
downsampling. Lets conv layers stack to depth 4 on 28×28 inputs, unlike the default
aggressive-downsampling block.
_Avoid_: standard block, conv layer.

### Run metadata

**Ladder tag**:
A W&B run tag of the form `ladder:<name>` (e.g. `ladder:mlp-depth`). A single run carries
one tag per ladder it belongs to, so the deduplicated anchor run carries several.
Downstream tooling discovers ladder membership generically from the `ladder:` prefix.
_Avoid_: axis tag (reserved for the coarser T-sweep / arch distinction), label.
