# Interpretation: FirSweep

**Artifacts:** `src/cache/results/psaunder__FirSweep/`
- `t_sweep_table.csv` (top level — **stale**, see note below)
- `plots/{sgd-m0.0,sgd-m0.9}/t_sweep_table.{csv,tex}`
- `plots/{sgd-m0.0,sgd-m0.9}/t_sweep_main.png`, `t_sweep_delta_vs_constant.png`, `t_sweep_delta_vs_dynamic.png`
- `plots/{sgd-m0.0,sgd-m0.9}/sigma_shape.png`, `clip_shape.png`
- `plots/{sgd-m0.0,sgd-m0.9}/shape_variants/{sigma,clip}_shape__T_sweep__by_T.png`
- `plots/{sgd-m0.0,sgd-m0.9}/ladders/{mlp-width,cnn-width,cnn-depth}/table.{csv,tex}`, `main.png`
- `plots/{sgd-m0.0,sgd-m0.9}/ladders/overall/arch_forest_delta.png`
- `plots/sgd-m0.9/curves/t_sweep_acc__mnist.png`
- `missing.csv`, `scalars.parquet` (used to establish n, n_reps, and read-off convention)

**Date:** 2026-08-04

---

## Global setup (applies to everything below)

- Two **optimizer arms** in parallel directories: `sgd-m0.0` (no inner-loop momentum) and `sgd-m0.9` (momentum 0.9). Everything else nominally matched.
- Two **axes**: `T-sweep` (T ∈ {2000, 3000, 5000, 7000} × ε ∈ {3, 5, 8, 10}, fixed arch `cnn-16x32-head32`, MNIST + Fashion-MNIST) and `arch` (fixed T = 5000, ε = 10, three architecture ladders, + CIFAR-10).
- Four **methods**: Learned Schedule, Dynamic-DPSGD, Adaptive Clip (Andrew et al.), Constant σ/clip. Metric is test accuracy (%), ↑ better. Direction is not stated on any figure or table.
- Nominally **8 seeds** per cell; 851 runs total, 3404 (run × method) rows in `scalars.parquet`.

### Three setup facts that are nowhere on the figures and change how everything reads

1. **`n_reps` is 1 for Learned and 8 for every baseline** (`scalars.parquet`, exact — 851 rows at `n_reps=1` for Learned, 851 each at `n_reps=8` for the other three). Every baseline number is a mean over 8 seeds × 8 evaluation replicates; every Learned number is 8 seeds × 1 replicate. **The `±` columns are not comparable across columns** — Learned's spread is inflated by construction. (`learned_acc_8rep` exists as a column but is null in all 3404 rows, so the intended fix was never populated.)

2. **The read-off step differs per figure, per arm, and — inside the arch ladders — per run.** Captions: m0.0 T-sweep "outer step 1000"; m0.9 T-sweep "outer steps 978–1000"; m0.0 cnn-depth ladder "outer steps 426–1000"; m0.9 cnn-depth ladder "outer steps **55**–1000". The ladder captions are ranges because each run is read at *its own* last completed step. So a single arch cell averages seeds read at outer step 55 with seeds read at step 1000.

3. **119 of 851 runs (14%) never reached 1000 outer steps, and 116 of those 119 are CIFAR-10 arch runs.** Median final step by rung (CIFAR, m0.0/m0.9): `mlp-512` 134/139, `cnn-16x16x16` 698/496, `cnn-16x16x16x16` 681/690, `cnn-32x64` 656/705. Only `cnn-8x16-head64` and `mlp-64` are fully trained on CIFAR in either arm. **Every CIFAR ladder number except those two rungs is built from truncated outer loops.**

### File-hygiene note

`t_sweep_table.csv` at the top level is dated Jul 31 and **does not match** `plots/sgd-m0.9/t_sweep_table.csv` (Aug 3) — 9 of 32 rows differ (e.g. FMNIST ε=8 T=2000: `84.950` stale vs `84.863` current; MNIST ε=3 T=5000 Constant: `94.821 ± 1.119` stale vs `95.060 ± 0.847` current). Everything else in the directory was regenerated Aug 3. **Delete or regenerate the top-level copy** before anyone quotes it.

---

## `plots/sgd-m0.0/` T-sweep — table, `t_sweep_main.png`, `t_sweep_delta_vs_{constant,dynamic}.png`

**Setup:** Rows = (dataset, ε, T); columns = four methods, `mean ± spread` test accuracy, bold = per-row best. Main plot: x = T (2000–7000, linear), y = test accuracy, panels ε (cols) × dataset (rows), y **shared across all 8 panels** (73–92), shaded = 95% CI. Delta plots: y = Learned − baseline, ±4 (vs Constant) / ±7.5 (vs Dynamic).

### What it shows

- **[shown]** Learned wins 18 of 19 populated rows. The margin is large at small T and vanishes at large T. Fashion-MNIST, Learned − Constant: **+2.55 / +1.36 / +0.12 / −0.04** at T = 2000/3000/5000/7000 (ε=3); the same monotone decay holds at all four ε, crossing zero at T ≈ 5000.
- **[shown]** Learned − Dynamic-DPSGD follows the identical decay: **+2.90 / +1.58 / +0.42 / +0.15** (ε=3 FMNIST), ≈ +3.2 → +0.3 at ε = 5, 8, 10.
- **[shown]** Adaptive Clip is last in all 19 rows by a wide margin: FMNIST 73.5–80.3 vs 81–85 for everyone else (−7 to −10 pts); MNIST 89.9–91.9 vs 94.5–95.9.
- **[shown]** ε is nearly inert. FMNIST T=2000: Learned goes 83.888 → 84.271 across ε = 3 → 10, i.e. **a 3.3× budget increase buys +0.38 pts**. Constant moves +0.09, Adaptive Clip +0.02.
- **[shown]** Only 19 of 32 cells are populated. MNIST has **3 cells total** (ε=3 T=2000 n=8, ε=3 T=3000 n=5, ε=10 T=7000 n=7); the ε=5 and ε=8 MNIST panels are entirely blank in `t_sweep_main.png`. FMNIST ε=10 T=5000/7000 are **n=2** (`85.250 ± 3.177`, `85.925 ± 3.494`).
- **[inferred]** The Learned advantage is a *scarce-budget allocation* effect, not a general one — assumes accuracy at large T is optimisation-limited rather than noise-limited, in which case any valid schedule reaches the same ceiling and shape has nothing left to buy. The decay is monotone across 8 independent (ε, dataset) panels, which is strong support.
- **[inferred]** This arm is clip/step-size limited, not noise limited — assumes σ is not the binding constraint at m=0.0. Supported jointly by the ε-inertness and by Adaptive Clip (which manipulates clip, not σ) being the method that collapses.
- **[not shown]** No hyperparameter table. Whether the inner learning rate, or Adaptive Clip's target quantile / clip learning rate, were tuned per arm is not displayed anywhere.

### Rigor concerns

- **The MNIST row is 3 of 16 cells and cannot support any MNIST claim in this arm.** From `missing.csv`, **108 of the 141 missing runs are m0.0 / T-sweep / MNIST, all with reason `HTTP 500: parquet: could not read footer: context canceled`** — a W&B *download* failure, not a training failure. These runs almost certainly exist and are recoverable by re-fetching. This is the highest-value fix in the batch: it is 108 runs of already-spent compute.
- **A further 17 m0.0 FMNIST T-sweep runs are `run never started`** — these are the ε=10 T=5000/7000 cells, hence n=2. Those two cells are bolded as Learned wins on the strength of two seeds with ±3.2–3.5 spread; the `t_sweep_main.png` ε=10 panel shows the CI fan opening to ±5 pts. Drop or re-run them; do not bold them.
- **The ± is not comparable across columns** (n_reps 1 vs 8). "Learned has the widest error bars" is an artefact.
- **Shared y-axis across all 8 panels of `t_sweep_main.png`** is set by Adaptive Clip's 73.5 floor, compressing the Learned/Dynamic/Constant separation into the top ~15% of each panel. The delta plots are the only readable view of the actual effect.
- **Adaptive Clip's collapse is a baseline-integrity problem, not a result.** A published method landing 10 pts below *constant* σ/clip is far more likely mis-tuned than genuinely that bad — and the m0.9 arm demonstrates it (same code, +8.5 pts). As presented it inflates the apparent margin of every other method.
- **Caption text overlaps the legend** in `t_sweep_main.png` and both delta plots — the n and read-off statement is illegible in the rendered figure.

---

## `plots/sgd-m0.9/` T-sweep — table, `t_sweep_main.png`, `t_sweep_delta_vs_{constant,dynamic}.png`

**Setup:** As above, momentum-0.9 arm. **Complete**: all 32 cells, n = 8 seeds (one cell n = 7). Main plot y-axis is **per-row** (FMNIST 83.5–86.5, MNIST 93.8–97.0) — a ~3 pt window, not the ~19 pt window of the m0.0 figure.

### What it shows

- **[shown]** The ranking **inverts**. Learned wins 4 of 32 rows (all MNIST, T ∈ {5000, 7000}, ε ∈ {3, 5}). Adaptive Clip wins 19, Dynamic-DPSGD wins 9, Constant wins 0.
- **[shown]** All four methods collapse into a ≈ 0.3–0.8 pt band. FMNIST ε=3 T=2000: 84.24 / 84.32 / 84.39 / 84.54 — a 0.31 pt total spread against ±0.17–0.53 reported error bars.
- **[shown]** **Learned − Constant is now increasing in T** (≈ 0 at T=2000 → +0.6 FMNIST / +1.4 MNIST at T=7000) — the exact opposite of the m0.0 arm's monotone decay. Every FMNIST CI in that plot includes zero at every T.
- **[shown]** **Learned − Dynamic-DPSGD is negative at every ε and every T on Fashion-MNIST** (−0.1 to −0.4) and ≈ 0 ± 0.15 on MNIST. So "Learned wins" in this arm holds only against Constant.
- **[shown]** Constant is now clearly worst, has the largest spread (MNIST ε=3 T=5000: `95.060 ± 0.847` vs Learned `96.200 ± 0.177`), and its MNIST accuracy **decreases** with T at ε=3 and ε=5.
- **[shown]** Adaptive Clip goes from last-in-every-row to best-in-19-rows on identical (ε, T, arch, dataset).
- **[inferred]** Momentum is a partial substitute for a learned schedule, not a competitor to it. Averaged over the 16 FMNIST cells where both arms are complete, the m0.9 − m0.0 accuracy gain is:

  | method | mean gain (pts) |
  |---|---|
  | Adaptive Clip | **+8.46** |
  | Dynamic-DPSGD | +1.95 |
  | Constant | +1.33 |
  | **Learned** | **+0.25** |

  and the gain is concentrated at small T (Constant: +3.39 at T=2000 → −0.32 at T=7000; Learned: +0.57 → −0.04). Read this as: **the learned schedule already extracts most of what momentum provides; the baselines need momentum to catch up.** Assumes the two arms are otherwise matched — which is exactly what is *not* documented (see below).
- **[not shown]** Whether the inner learning rate was re-tuned when momentum was turned on. With m=0.9 the effective step is ~10× larger at fixed lr, so an un-retuned lr is a live confound for the whole cross-arm comparison.

### Rigor concerns

- **The whole-arm result rests on differences smaller than the outer-loop noise** (see the training-curve artifact below). A 0.3 pt table spread is not resolvable by a read-off that samples a ±1 pt band.
- **Favourable-baseline framing.** `t_sweep_delta_vs_constant.png` is positive everywhere and looks like a clean win for Learned; `t_sweep_delta_vs_dynamic.png` — same data, same arm — is negative everywhere on FMNIST. If only the vs-Constant panel is shown, the reader is handed the flattering comparison. Show both or show neither.
- **Different y-scaling between arms** makes the two `t_sweep_main.png` figures visually non-comparable: the m0.9 figure's 3 pt window makes a 0.3 pt effect look like the m0.0 figure's 3 pt effect. Any side-by-side presentation needs a shared scale or an explicit warning.
- **Different read-off convention between arms** (single step 1000 vs mean over 978–1000). The m0.9 numbers are 23-step averages and the m0.0 numbers are single draws; the m0.9 spreads are therefore smaller partly by protocol, not by stability.

---

## `plots/sgd-m0.9/curves/t_sweep_acc__mnist.png`

**Setup:** x = outer step (0–1000), y = test accuracy, one panel per ε (cols) × T (rows). One line per seed, Learned schedule only, m0.9 arm.

### What it shows

- **[shown]** Every panel is a dense noise band with **no visible trend after outer step ≈ 50**. T=2000/ε=3 spans roughly 94.5–97.0 across the full 1000 steps; T=7000/ε=10 spans roughly 96.0–97.8. Band width is ≈ 1–2 pts throughout.
- **[shown]** The initial rise is complete within ~20–50 outer steps in all 16 panels.
- **[inferred]** **The reported between-method differences (0.1–0.5 pts) are an order of magnitude smaller than the within-run outer-step fluctuation (~1–2 pts).** Averaging over 978–1000 shrinks this, but those 23 steps are adjacent and autocorrelated, so the effective reduction is well short of √23.
- **[inferred]** ~950 of the 1000 outer steps are buying nothing measurable. If this holds on the other datasets, the outer-step budget is the obvious place to reclaim compute for more seeds — which is what the tables actually need.
- **[not shown]** The equivalent curve figures exist for FMNIST and for m0.0 but were not checked here; whether the "converged by step 50" pattern is universal is unverified.

### Rigor concerns

- **This figure undercuts the read-off protocol used by every table in the batch.** Any claim resting on a sub-0.5 pt gap needs either (a) a much wider averaging window with a stated autocorrelation argument, or (b) more seeds, or (c) a paired test across seeds at matched outer steps. None of the three is present.
- One line per seed at full opacity produces solid ink; per-seed behaviour (does *any* seed drift or diverge?) is unreadable. A per-seed rolling median would answer the question the plot is meant to answer.

---

## `plots/{sgd-m0.0,sgd-m0.9}/shape_variants/{clip,sigma}_shape__T_sweep__by_T.png`

**Setup:** x = t/T (0–1), y = clip or σ, panels ε (rows) × dataset (cols), colour = T. Thick line = seed mean, thin = per seed. The m0.0 MNIST column is mostly empty (the 108 lost runs).

### What it shows — the most robust result in the batch

- **[shown]** The learned schedule is a clean **concave arch** in both σ and clip: rises from ≈ 0 at t/T = 0, peaks, decays to ≈ 0 at t/T = 1. Per-seed lines sit essentially on top of the mean — **this shape is reproducible across all 8 seeds**, unlike any accuracy number in this batch.
- **[shown]** Peak height **decreases monotonically in T** at every (ε, dataset, arm). m0.0 FMNIST clip peak: **10.7 / 8.6 / 6.0 / 4.5** at T = 2000/3000/5000/7000 (ε=3). m0.9 FMNIST clip peak: **1.45 / 0.99 / 0.61 / 0.44** (ε=3).
- **[shown]** Peak height **increases monotonically in ε**, weakly: m0.9 FMNIST T=2000 clip peak 1.45 → 1.57 → 1.77 → 1.83 for ε = 3 → 5 → 8 → 10.
- **[shown]** **The whole schedule is ~7× smaller under momentum**, in both σ and clip, but the *ratio* is preserved: m0.0 FMNIST ε=3 T=2000 peak clip/σ ≈ 10.7/6.5 = 1.65; m0.9 ≈ 1.45/0.87 = 1.67. At ε=10: 11.4/5.2 = 2.19 vs 1.83/0.83 = 2.20. This is what the accountant requires (σ = C/μ pins C/σ = μ), so it is a **passing internal-consistency check on the privacy projection**, independently in both arms.
- **[shown]** The *temporal placement* does change with momentum. m0.0: peak at t/T ≈ 0.4–0.5, near-symmetric plateau. m0.9: peak at t/T ≈ 0.25–0.30 with a long asymmetric decay. MNIST peaks later than FMNIST in both arms.
- **[shown]** m0.0 FMNIST σ has a reproducible **shoulder / double hump** near t/T ≈ 0.12 at ε = 8 and 10 (local max ≈ 4.2, dip, then main peak ≈ 5.2). Absent in m0.9.
- **[inferred]** Peak clip scales roughly as a power of T. m0.0 ε=3: 10.7/4.5 = 2.4 against a T ratio of 3.5, i.e. ≈ T^(−0.7). m0.9 ε=3: 1.45/0.44 = 3.3, i.e. ≈ T^(−0.95), near 1/T. **This is directly checkable and is the most promising closed-form / symbolic-regression target in the batch** — assumes the B-spline parameterisation isn't itself imposing the arch, which the endpoint behaviour (forced to ≈ 0 at both ends) makes worth confirming.
- **[not shown]** No overlay of the analytic optimum, the Dynamic-DPSGD shape, or a fitted functional form. The arch is described but not compared to anything.

### Rigor concerns

- **`sigma_shape.png` and `clip_shape.png` (the aggregate, not-by-T versions) are unusable for m0.9.** A single CIFAR outlier reaching σ ≈ 330 sets the shared y-axis, flattening the Fashion-MNIST and MNIST panels to a line at zero. The by-T variants carry all the information; the aggregate versions should be dropped or given per-panel y-limits.
- **The ε colour encoding is defeated by overplotting** in the aggregate figures — ε=10 (yellow) has the most runs and covers everything else. The by-ε-row layout of the `shape_variants` figures is the right design; use it everywhere.
- **Near-t/T=0 behaviour is unreliable.** m0.9 MNIST mean curves show a non-monotone dip-then-rise below t/T ≈ 0.1, and per-seed lines fan wildly there. This looks like a B-spline boundary artefact rather than a learned feature; it should not be interpreted.
- **The clip peak values are read from figures, not tables.** The scaling exponents above are read-off estimates (±5–10%). If this becomes a claim, extract from `schedules.parquet` and fit properly.

---

## `plots/{sgd-m0.0,sgd-m0.9}/ladders/{mlp-width,cnn-width,cnn-depth}/table.csv` + `main.png` + `ladders/overall/arch_forest_delta.png`

**Setup:** Fixed ε = 10, T = 5000. Rows = ladder rung (arch), columns = four methods. Forest plot: y = rung grouped by ladder, x = Learned − Constant Δ accuracy, diamond = seed mean, dots = per seed, line = 95% CI, one panel per dataset (**shared x-axis across the three panels**).

### What it shows

- **[shown] m0.0, structure by ladder type.** On CIFAR-10 the sign of Learned − Constant depends on which ladder you walk:
  - `cnn-width`: **−2.05 / −2.05 / −1.30** (8x16 / 16x32 / 32x64) — negative at all three widths, CIs excluding zero, per-seed dots tightly clustered.
  - `cnn-depth`: **−0.14 / +1.59 / +4.02 / +2.08** (1 / 2 / 3 / 4 blocks) — positive and peaking at 3 blocks.
  - `mlp-width`: −0.16 / +0.11 / −0.71.
- **[shown] m0.0, FMNIST and MNIST.** All rungs ≥ 0 and small: FMNIST +0.19 to +1.38, largest at the deepest CNNs; MNIST ≈ 0 everywhere except mlp-128 (+0.79) and mlp-512 (+1.25).
- **[shown] m0.0 Adaptive Clip is again catastrophic** — CIFAR `cnn-16x16x16x16`: 36.19 vs 52.8–54.9 for the others; FMNIST deepest rung 77.00 vs 84.5–85.9.
- **[shown] m0.9 CIFAR Learned blows up on the deep/wide rungs**: `mlp-128` 35.850 ± 10.481, `mlp-512` 36.213 ± 10.310, `cnn-16x16x16` 46.393 ± 19.253, `cnn-16x16x16x16` 35.808 ± 14.374, against 44–60 for all three baselines.
- **[shown]** Those blowups are **bimodal, not noisy**. Per-seed Learned accuracy, m0.9 CIFAR `cnn-16x16x16` (with each run's final outer step): 59.2 @496, 58.3 @716, **8.95 @128**, 58.2 @469, **22.9 @93**, 57.3 @724, 59.2 @723. `mlp-512`: 44–45% for five seeds (all @~135), **19.2 / 22.8 / 19.6** for three seeds (also @~135).
- **[inferred]** Two distinct failure modes are being averaged into one cell: (a) runs truncated so early the schedule is effectively untrained (steps 55–128), and (b) genuine outer-loop divergence at ~step 690–708 on the deepest CNN. `mlp-512` shows (b) exists independently of (a) — all eight seeds stopped at the same step ~135, yet split 5 good / 3 collapsed. Assumes the evaluation is deterministic given the schedule, which `n_reps=1` makes only approximately true.
- **[inferred]** The m0.0 `cnn-width` negative result is **not** a truncation artefact: `cnn-8x16-head64` has zero incomplete runs in that arm and still shows −2.05 with a tight CI. Something about widening a CNN at fixed depth genuinely defeats the learned schedule on CIFAR. This is the most interesting unexplained finding in the batch.
- **[not shown]** `arch_param_count` is in `scalars.parquet` but is not used as an x-axis anywhere. The width/depth ladders confound parameter count with structure; a Δ-vs-param-count plot would separate them and is nearly free to produce.

### Rigor concerns

- **`plots/sgd-m0.9/ladders/overall/arch_forest_delta.png` is unreadable for two of its three datasets.** The shared x-axis is stretched to −50…+10 by the CIFAR outliers, so the Fashion-MNIST and MNIST panels are a vertical line at 0. The m0.0 version (±4) is fine. Give each panel its own x-limits, or clip and annotate the outliers.
- **Mean ± std over a bimodal population is the wrong summary.** `46.393 ± 19.253` describes no run that actually happened. Report median + per-seed dots, or split the modes and report the failure rate explicitly ("3 of 8 seeds collapsed") — the forest plot's per-seed dots already do this honestly and should be the primary artifact.
- **The read-off is not matched across seeds within a cell.** The m0.9 cnn-depth caption reads "outer steps 55–1000": some seeds contribute a schedule after 55 outer updates, others after 1000, and they are averaged together. The m0.0 equivalent is 426–1000. **A cell should be read at one step for all its seeds, or truncated runs should be excluded** — currently the cell mean blends trained and untrained schedules, and the two arms' ladders are read at different points of outer training, which makes the m0.0-vs-m0.9 arch comparison invalid as presented.
- **CIFAR is the dataset where the compute ran out, and it is also the dataset carrying the most interesting claims.** 116 of 119 truncated runs are CIFAR arch runs; only `cnn-8x16-head64` and `mlp-64` are fully trained. Every other CIFAR row needs "not trained to completion" attached to it, or a re-run.
- **Caption/legend overlap** again in every `main.png` — the n and read-off statement is illegible.
- **No metric-direction marker** (↑) and no caption at all in the `.tex` files (bare `tabular`, no `\caption`), so n, read-off, and direction are lost the moment a table is pasted into a document.

---

## Synthesis

### Agreements

- **The learned schedule's advantage is real but conditional on the inner optimiser being weak.** The m0.0 T-sweep (+2.5 to +2.9 vs Constant at T=2000), the m0.0 arch ladders (positive on FMNIST/MNIST, +4 on CIFAR cnn-depth), and the cross-arm momentum-gain table (Learned +0.25 vs Constant +1.33, Dynamic +1.95, Adaptive Clip +8.46) all say the same thing from different directions.
- **The advantage decays with T** in the m0.0 arm across all 8 (ε, dataset) delta panels, monotonically, crossing zero around T ≈ 5000.
- **The learned shape is a concave arch whose peak falls monotonically in T and rises weakly in ε**, reproducibly across seeds, in both arms, in both σ and clip.
- **The privacy projection is behaving.** Peak clip/σ matches to within 1% across arms at fixed ε — an independent check that the GDP constraint is enforced identically regardless of optimiser.

### Contradictions

- **The T-trend of Learned − Constant reverses between arms**: decreasing in T at m0.0 (+2.6 → 0.0), increasing in T at m0.9 (0.0 → +0.6/+1.4). Trust the m0.0 trend more — its effect sizes (2–3 pts) exceed both the error bars and the outer-step noise band, whereas the m0.9 effects (0.1–0.6 pts) sit inside both. The m0.9 "trend" may be a read-off artefact.
- **Adaptive Clip is either the worst method by 10 pts or the best method, depending only on inner momentum.** Both cannot be a property of the method. The m0.9 behaviour (competitive) is the plausible one; the m0.0 behaviour is a mis-tuning signature. Until that is resolved, **every margin computed against Adaptive Clip in the m0.0 arm is inflated**.
- **`delta_vs_constant` and `delta_vs_dynamic` disagree on whether Learned wins in the m0.9 arm** (positive everywhere vs negative everywhere on FMNIST). Both are correct; the conclusion depends entirely on baseline choice.

### Strongest supported claim

*At small inner-loop budgets (T ≈ 2000–3000) and without inner momentum, a learned σ/clip schedule beats constant σ/clip and Dynamic-DPSGD by 1.5–3 accuracy points on Fashion-MNIST, across ε ∈ {3, 5, 8, 10}, with n = 8 seeds and non-overlapping error bars; and the learned schedule is a reproducible concave arch whose peak scales down with T.* This survives every rigor check applied here. Two supporting facts belong alongside it: the advantage vanishes by T ≈ 5000–7000, and it is largely a substitute for inner-loop momentum rather than an addition to it.

### Weakest link

**The m0.9 CIFAR-10 arch rows.** They combine (a) 116 truncated runs, (b) a within-cell read-off spanning outer steps 55–1000, (c) bimodal per-seed outcomes summarised as mean ± std, and (d) `n_reps=1` on the very column that is collapsing. `46.393 ± 19.253` is not a measurement. It should be reported as a stability finding ("3 of 8 seeds collapse") or withheld.

Running it close: **the entire m0.9 T-sweep**, where the reported effects (0.1–0.5 pts) are smaller than the outer-step noise band visible in `curves/t_sweep_acc__mnist.png` (~1–2 pts). That table may be measuring nothing.

### Open questions

1. **Re-fetch the 108 lost m0.0 MNIST T-sweep runs.** They failed with `HTTP 500: parquet: could not read footer` — a W&B download error, not a training failure. That is 108 completed runs of compute currently invisible, and it is why the m0.0 arm has 3 MNIST cells instead of 16. Highest value per unit effort in this batch.
2. **Was the inner learning rate re-tuned between arms?** With m=0.9 the effective step is ~10× larger at fixed lr. If it was not re-tuned, the entire cross-arm comparison — including the momentum-gain table above — is confounded.
3. **Was Adaptive Clip tuned at all in the m0.0 arm?** A 10 pt gap below constant σ/clip, closing to a win under momentum, is a tuning signature. Until this is settled, do not report m0.0 margins against it.
4. **Fix the read-off protocol before re-plotting.** One step for all seeds in a cell; the same step (or window) in both arms; exclude runs that did not reach it. The current per-run "last available step" rule silently averages trained and untrained schedules.
5. **Is 1000 outer steps needed?** The training curves converge by ~50 and then wander over a ~1–2 pt band. Reallocating outer steps to seeds and to `n_reps` on the Learned column would directly fix the two biggest statistical weaknesses.
6. **Populate `learned_acc_8rep`.** The column exists and is null in all 3404 rows; filling it makes the `±` columns comparable and removes the "Learned's error bars are wider by construction" caveat.
7. **Why does widening a CNN at fixed depth defeat the learned schedule on CIFAR (m0.0, −1.3 to −2.1 at all three widths) while deepening it helps (up to +4.0)?** `cnn-8x16-head64` is fully trained and shows the effect cleanly, so this is not truncation. Plot Δ against `arch_param_count` (already in `scalars.parquet`) to separate parameter count from structure.
8. **Fit the peak-clip scaling law.** Read-off estimates suggest peak clip ∝ T^(−0.7) at m=0.0 and ≈ T^(−1) at m=0.9. Extract from `schedules.parquet` and fit properly — the cleanest closed-form candidate the sweep has produced.
9. **Regenerate or delete the stale top-level `t_sweep_table.csv`** (Jul 31; 9 of 32 rows disagree with the current m0.9 table).
10. **Fix the figure captions.** Caption text overlaps the legend in every `t_sweep_main.png`, delta plot, and ladder `main.png`; the `.tex` tables have no caption at all, so n, read-off window, and metric direction do not survive being pasted into a document.
