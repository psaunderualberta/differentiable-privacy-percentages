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

## Pinned surrogate specifications

The reproducibility pins this ADR requires, settled during the transfer-branch grilling:

- **CheXpert** — single finding **Pleural Effusion** (highest-prevalence, most salient
  competition finding), **U-Zeros** uncertainty convention (uncertain `-1` → negative,
  blank/NaN → negative), binary one-hot. **Frontal view only**, **grayscale (1 channel)**,
  resized to **64×64**. Source: **Kaggle CheXpert-v1.0-small** via the Kaggle API,
  mirroring `_eyepacs_download_and_cache`. Surrogate net reuses the EyePACS/MNIST conv
  block (`channels=(16,32)`, k=8/4, s=2, pool=2, head `(32,)`) → `arch_label`
  `cnn-16x32-head32`. The tiny official `valid.csv` (234 rows) is unused; the val/test pool
  is carved out of `train.csv` and permuted via `split_seed` (like `california`), so
  `δ ≤ 1/N` uses the reduced `N_train`.
- **ImageNet-32** — source is a **Hugging Face downsampled-ImageNet-32 mirror** (Chrabaszcz
  32×32); subset is the **published ImageNet-100 wnid list (Tian et al. 2019, CMC)** — a
  fixed, citable 100-class set. Shape `(3, 32, 32)`, 100-class. Surrogate net reuses the
  **cifar-10** default CNN (`channels=(32,64)`, 3×3, s=1, pool=2, head `(256,)`).
  **64×64 fallback trigger**: stay at 32×32; after the EyePACS-validated pipeline runs on
  ImageNet-32, escalate to ImageNet-64 @ 64×64 **only if** the Constant reference's mean
  top-1 fails to clear ≈2× chance (<2%), i.e. paired Δacc is within seed noise. The cifar
  CNN adapts to 64×64 (larger flattened head) without a config change.

All three targets are **targets only**, never sources.
