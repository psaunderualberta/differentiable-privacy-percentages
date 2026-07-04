# Evaluate policy transfer with from-scratch DP-SGD on surrogate target regimes, not realistic training

Policy transfer to the held-out targets (EyePACS, CheXpert, ImageNet) is measured by
running the **same from-scratch small-CNN DP-SGD** the source policies were learned in —
a single forward-only training run per cell — rather than the realistic training setup
each dataset would normally warrant. To keep that from-scratch regime feasible and
above the accuracy floor, each large target is coerced into a **surrogate regime**:

- **EyePACS** — used as-is (5-class diabetic-retinopathy grading, already wired), the one
  target that is natively a from-scratch small-CNN task.
- **ImageNet** — **ImageNet-32, 100-class subset** (64×64 fallback). Not real
  1000-class ImageNet.
- **CheXpert** — reframed as **binary single-finding** classification (one salient,
  well-populated finding) with a fixed uncertainty-label convention (U-Ones or U-Zeros).
  Not multi-label.

Every target thus becomes a top-1 classification-accuracy task under matched privacy,
apples-to-apples with the sources and with each other.

## Status

accepted

## Why

The transfer claim is about the learned schedule **shape** conferring a *relative*,
matched-privacy accuracy benefit — not about state-of-the-art absolute accuracy. That
claim is only coherent if the target is trained the *same way* the shape was learned. So
the paradigm is fixed to from-scratch DP-SGD, and the datasets bend to fit it rather than
the reverse.

## Considered and rejected

- **DP fine-tuning of a non-privately pretrained backbone** (the standard way to get real
  DP-on-ImageNet numbers). Rejected: it requires infrastructure the codebase lacks
  (pretrained weights, no per-run reinit, feature-extractor freezing) **and changes the
  question** — a schedule learned for from-scratch training dynamics need not transfer to
  fine-tuning dynamics, so a good/bad number would be uninterpretable.
- **Full 1000-class / full-resolution ImageNet.** Rejected: from-scratch private accuracy
  is a floor, so per-curve transfer differences would be unresolvable noise.
- **True multi-label CheXpert** (sigmoid + BCE + AUC). Correct for the data, but breaks the
  softmax/top-1-accuracy assumption baked through `environments/dp.py`, the baselines, and
  the plotting — a large modelling project for a dataset meant only as a transfer *probe*.
- **Dropping ImageNet/CheXpert.** Rejected: they are the point — genuinely harder,
  real-world, out-of-distribution targets. EyePACS alone is too close to the training
  datasets to make a generalisation claim.

## Consequences

- Absolute target accuracies will be low. This is expected and acceptable; all comparisons
  are *relative* and *paired within a target*, so the floor cancels.
- The surrogate choices (ImageNet class subset + resolution, CheXpert finding + U-convention)
  must be pinned and documented so the numbers are reproducible.
- ImageNet is the riskiest target for a floor effect; validate the pipeline end-to-end on
  EyePACS first, then ImageNet-32, then CheXpert.
