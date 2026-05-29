# Interpretation: DCSweep (SGD) — learned σ/clip schedules vs baselines

**Artifacts:**
- `arch_sweep_table.tex` / `.csv`, `t_sweep_table.tex` / `.csv`
- `arch_sweep_main.png`, `t_sweep_main.png`
- `arch_sweep_delta_vs_constant.png`, `arch_sweep_delta_vs_dynamic.png`
- `t_sweep_delta_vs_constant.png`, `t_sweep_delta_vs_dynamic.png`
- `sigma_shape.png`, `clip_shape.png`
- `shape_variants/{sigma,clip}_shape__T_sweep__by_T.png`, `shape_variants/{sigma,clip}_shape__arch_sweep__by_arch.png`
- `curves/arch_sweep_acc__mnist.png`

**Date:** 2026-05-29

**Common setup (all artifacts):** four methods compared — **Learned** (this work's optimized σ/clip schedule), **Dynamic-DPSGD**, **Median-Clip**, **Constant** (flat schedule baseline). Metric is validation accuracy (%), higher = better. Two datasets (mnist, fashion-mnist), four privacy budgets ε ∈ {1, 3, 5, 8}. The arch sweep varies network architecture (sorted by parameter count) at a fixed T (curves say T=5000); the T sweep varies inner-loop steps T ∈ {1500, 2000, 3000, 5000, 7000} at fixed architecture. δ is not shown in these artifacts (presumably fixed; verify).

---

## arch_sweep_table.tex / .csv

**Setup:** rows = (dataset, ε, arch_label); columns = the four methods. Cells are val-accuracy mean ± std; **bold** = best in row. Note many cells show `± 0.000`–`± 0.002`, several Constant/Dynamic/Median cells show **no ± at all**, and Learned stds are larger (≈0.03–0.31).

### What it shows
- [shown] **Learned wins the majority of rows at ε ≥ 3.** mnist: Learned is bold in every ε≥3 row. fashion-mnist: Learned bold in all ε≥3 rows except `mlp-512x256` (Dynamic wins ε=3,5; Learned wins ε=8) and `mlp-16` at ε=3 (Constant 83.611 vs Learned 83.567).
- [shown] **At ε = 1, Learned generally loses.** fashion-mnist ε=1: Learned loses every row (e.g. cnn-16x32 82.080 vs Constant **83.233**). mnist ε=1: Learned wins only `mlp-512x256` (92.567).
- [shown] **Median-Clip is consistently worst** in essentially every row (e.g. fashion-mnist ε=8 cnn-16x32: 81.997 vs others 85.7–87.9).
- [shown] **Learned's margin grows with ε.** mnist cnn-32x64: Learned − Constant = +0.62 (ε=1, actually Dynamic wins), +0.74 (ε=3), +0.88 (ε=5), +1.03 (ε=8). Similar monotone trend on fashion-mnist.
- [shown] **Bigger nets help Learned more.** mnist ε=8: Learned − Constant is +0.73 (cnn-16x32) but +2.11 (mlp-512x256: 96.177 vs 94.071).
- [inferred] Learned schedules pay off when there is enough budget to shape the σ/clip trajectory; under tight privacy (ε=1) a flat Constant schedule is as good or better — assumes the ε=1 losses are real and not tuning artifacts.
- [not shown] Number of seeds per cell; δ; whether the same T/architecture grid is identical across all four methods.

### Rigor concerns
- **Variance is implausibly small for baselines (`± 0.000`).** This almost certainly is *not* run-to-run seed variance — more likely eval noise on a single fixed schedule, or n is tiny. Learned stds (up to ±0.31) and baseline stds (±0.000) are not comparable quantities; confirm what the ± actually measures and over how many seeds.
- **Missing ± on many cells implies n=1 for those configs** (e.g. all mnist mlp rows at ε=5: 98.100, 93.160 with no spread). A bold winner with no variance cannot be called significant.
- **Bolded margins often << reported spread.** mnist ε=3 mlp-16: Learned 92.907 ± 0.186 vs Constant 92.567 — gap 0.34, ~1.8σ of the Learned std alone; with baseline variance unknown this is borderline. fashion-mnist ε=3 mlp-16: 83.567 vs 83.611 bolded for Constant — a 0.044 gap, inside noise.
- **Unequal architecture rows across datasets:** mnist & fashion-mnist ε=1 omit `cnn-64x128-head128` (present at ε≥3). Why is the largest net absent at ε=1? Possible survivorship/compute confound (see curves below).
- **Metric direction not labelled** in the table header (↑ implied).

---

## t_sweep_table.tex / .csv

**Setup:** rows = (dataset, ε, T); columns = the four methods; val-accuracy mean ± std, bold = row best.

### What it shows
- [shown] **Learned wins almost every ε ≥ 3 row across all T**, both datasets. fashion-mnist ε=3–8 and mnist ε=3–8: Learned bold in all 30 rows.
- [shown] **ε = 1 is the weak regime again.** fashion-mnist ε=1: Learned wins only T=1500 (82.867); Constant/Dynamic win T≥2000. mnist ε=1: Learned wins T=1500, 2000; Dynamic wins T≥3000.
- [shown] **Learned degrades as T grows under tight privacy.** fashion-mnist ε=1: Learned 82.867 (T=1500) → 81.755 (T=7000), i.e. *decreasing*; Constant stays ~82.7–83.2. mnist ε=1 Learned: 96.820 → 96.385.
- [shown] **At ε ≥ 3 Learned's lead is roughly flat-to-growing in T.** fashion-mnist ε=8: Learned − Constant ≈ +2.19 (T=1500) → +1.69 (T=7000), still large everywhere.
- [shown] Median-Clip again uniformly worst; it *improves* with T (fashion-mnist ε=1: 79.885 → 82.229) but never leads.
- [inferred] The learned schedule's advantage is a budget-shaping effect that is robust to inner-loop horizon for ε≥3, but at ε=1 more steps dilute the per-step budget to where a flat schedule is preferable — assumes the T=7000/ε=1 Learned dip is real, not an optimization failure of the outer loop.

### Rigor concerns
- Same `±0.000`/missing-± issue as the arch table — several rows (mnist ε=5 T=1500/3000/5000/7000) have **no variance at all**, yet are bolded.
- **The ε=1 reversal trend rests on small gaps** (fashion-mnist ε=1 T=7000: Learned 81.755 ± 0.265 vs Constant 82.963 — gap 1.2, comfortably outside Learned σ, so this one is probably real; but several mnist ε=1 gaps are <0.2).
- Confirm T is held fixed identically per method and that all four were re-tuned (or none were) — an untuned Constant baseline would inflate Learned's lead.

---

## arch_sweep_main.png  /  t_sweep_main.png

**Setup:** 2×4 grids. Columns = ε ∈ {1,3,5,8}; rows = fashion-mnist (top), mnist (bottom). y = val accuracy. arch_main x = architecture sorted by param count; t_main x = T (inner-loop steps). Four colored series (Learned blue, Dynamic red, Median green, Constant grey). t_sweep_main shows faint shaded bands (uncertainty); arch_main does not visibly.

### What it shows
- [shown] Visually confirms the tables: at ε=1 the blue (Learned) line sits at/below red & grey; at ε≥3 blue is on top across nearly all x.
- [shown] Green (Median-Clip) is a clear lower envelope in every panel.
- [shown] arch_main is **non-monotone in param count** — there is a consistent dip at `mlp-16` (smallest MLP) in the mnist row (drops to ~91–93 vs ~96–98 for CNNs), i.e. architecture *family* matters more than raw param count, so the "sorted by param count" x-axis mixes two confounded factors (CNN vs MLP, and size).
- [shown] t_main: at ε≥3 all methods rise with T and plateau; Learned's blue band sits above red/grey with only slight overlap. At ε=1 lines are tangled and cross.
- [inferred] Learned ≈ Dynamic ≈ Constant within noise at ε=1; the method matters only once ε≥3.

### Rigor concerns
- **arch_sweep_main shows no uncertainty bands** while t_sweep_main does — inconsistent uncertainty reporting across twin figures.
- The shared y-axis per row compresses the ε=1 panel where differences are sub-point; the eye reads "all equal" but the table shows ordered (if tiny) gaps — neither over- nor under-stated, just hard to read.
- x-axis "sorted by param count" conflates MLP/CNN architecture family with size — the dip at mlp-16 is an architecture-type effect, not a smooth size trend.

---

## arch_sweep_delta_vs_constant.png  /  _vs_dynamic.png  +  t_sweep_delta_vs_*.png

**Setup:** Learned *minus* baseline accuracy (Δ acc), 2×4 grids (ε columns, dataset rows). Horizontal line at 0 = parity. Shaded band = uncertainty (likely across seeds). Positive = Learned better.

### What it shows
- [shown] **vs Constant, arch:** mnist row is ≥0 everywhere and rises with both ε and param count (ε=8 reaches Δ≈+1 to +2 at mlp-512x256). fashion-mnist ε=1 is **negative** (Δ≈−0.5 to −1, band excludes 0); ε≥3 mostly positive with one near-zero point.
- [shown] **vs Constant, T:** fashion-mnist ε=1 crosses from ~0 at T=1500 to **−1** at T=7000 (band below 0) — Learned actively worse. mnist ε=1 stays a small +0.1–+0.4. ε≥3 strongly positive (fashion ε=5,8 Δ≈+1.5–2.2), with a gentle decline as T grows.
- [shown] **vs Dynamic:** similar shape but fashion-mnist ε=1 is even more negative (Δ down to −1 to −2) — Dynamic-DPSGD is the toughest competitor at ε=1; at ε≥3 Learned beats it by ~+1 to +2 (fashion) and +0.5–1.3 (mnist).
- [inferred] The story "Learned wins, more so at higher ε / bigger model" is genuinely supported for ε≥3; the **ε=1 deficit vs both baselines is a real, reproducible weakness**, since the shaded band excludes 0 in several ε=1 panels.

### Rigor concerns
- Band semantics undocumented (std? SEM? CI? min–max?). The whole significance read depends on this — label it.
- Where the band is invisibly thin (e.g. several mnist points) it again suggests n is small or these are eval-noise bands, not seed bands.
- These Δ plots are the *honest* artifacts of the set — they're the only ones that surface the ε=1 regression that the bolded tables visually downplay.

---

## sigma_shape.png  /  clip_shape.png (+ by_T, by_arch variants)

**Setup:** learned schedule shape. x = normalized training progress t/T ∈ [0,1]; y = σ (noise) or clip threshold. Lines colored by ε (main plots) or by T / architecture (variants). Many overlaid lines = individual runs.

### What it shows
- [shown] **σ schedule:** large spike at t/T≈0 (σ up to ~6–7 fashion, ~5 mnist), rapid decay to a low plateau (~0.3–1.5) across the middle, sometimes a small uptick near t/T≈1. Interpreted: heavy noise early, light noise later.
- [shown] **clip schedule:** rises from ~0 to a broad hump peaking around t/T≈0.25–0.4 (clip up to ~8 at fashion ε=8) then descends back toward ~0 at t/T=1. Interpreted: small clip early, generous clip mid-training, tighten again at the end.
- [shown] **by_T variants:** both σ and clip magnitudes **scale down monotonically as T grows** (clip peak fashion ε=8: ~7.5 at T=1500 → ~2 at T=7000). Consistent with a fixed privacy budget spread over more steps.
- [shown] **by_arch variants:** `mlp-512x256` (yellow) is an outlier — much larger σ spike / different clip shape than the CNNs, especially at small ε.
- [shown] σ and clip are near-mirror images (σ high where clip low), as expected from the C = σ·μ coupling in the accountant.
- [inferred] The learned policy converges to a recognizable, low-dimensional shape (early-noise / mid-clip-hump) that is stable across seeds and ε — a genuinely interpretable finding.

### Rigor concerns
- **Heavy line overplotting** in the main σ/clip figures — individual runs are indistinguishable; cannot read per-condition spread or spot outlier runs. The `by_T`/`by_arch` variants fix this and should be the ones cited.
- No mean ± band overlaid on the shape plots; the "stable shape" claim is eyeballed from a spaghetti plot.
- y-axes differ per subplot row in the variants (good for visibility, but makes cross-ε magnitude comparison misleading at a glance).

---

## curves/arch_sweep_acc__mnist.png

**Setup:** outer-loop training curves (val accuracy vs outer step, 0–1000) for the **Learned** schedule only, mnist, T=5000. Rows = architecture, columns = ε.

### What it shows
- [shown] Most runs converge early (within ~100–200 outer steps) and plateau; `mlp-16` is visibly noisier (±~1% jitter) than CNNs.
- [shown] **Several `cnn-64x128-head128` panels (ε=3,5,8) terminate early** — curves stop around outer step ~300–400 instead of 1000.
- [inferred] The largest CNN's runs were cut short (compute/time limits or divergence) — this likely explains its **absence at ε=1** in the tables and is a survivorship concern: its reported accuracies come from fewer outer steps than competitors.

### Rigor concerns
- **Truncated outer-loop runs for the largest model** mean its table cells are not directly comparable (fewer optimization steps). Confirm whether reported accuracy is at matched outer-step budget across architectures.
- Curves shown for Learned only — no equivalent convergence check that the *baselines* were run to comparable convergence.

---

## ε=1 fashion-mnist deep-dive — curves/{arch,t}_sweep_loss/acc__fashion-mnist.png

**Setup:** outer-loop training curves for the **Learned** schedule only, fashion-mnist. arch grid: rows = architecture, cols = ε, T=5000. T grid: rows = T, cols = ε, arch=cnn-16x32-head32. x = outer step (0–1000); y = val loss (↓ better) or val accuracy (↑ better). Focus here = the ε=1 (leftmost) column.

### What it shows
- [shown] **No descent phase at ε=1.** cnn-16x32 loss drops ~0.78→~0.55 within ~50 outer steps then is **flat with ±~0.03 jitter** for the remaining ~950 steps. cnn-32x64 and mlp-512x256 likewise hit their plateau almost immediately; mlp-16 just oscillates at ~0.62–0.64 with no trend.
- [shown] **Same rows at ε≥3 keep descending** for hundreds of steps to a lower loss (cnn-16x32 reaches ~0.45–0.48 at ε=8 vs ~0.55 at ε=1) — a real learning phase that is essentially absent at ε=1.
- [shown] **cnn-64x128-head128 panel is empty at ε=1** — confirms the missing table row; this config was not run here.
- [shown] **T-sweep at ε=1: more T → less jitter but no-better (worse) plateau.** Every T plateaus ~0.60–0.62 with no post-~50-step descent; T=7000 is nearly flat-line yet sits at the *worst* plateau (~0.62, acc ~80.5), matching the table's 81.755.
- [shown] **Meta-overfitting on mlp-512x256, ε=1:** val accuracy starts ~84 and **drifts down to ~82 over outer steps** — the outer loop degrades held-out accuracy as it optimizes.
- [inferred] The ε=1 deficit is an **optimization/signal problem, not a schedule-shape problem**: at ε=1 the meta-objective is noise-dominated, so the outer loop converges (stably, at large T) to a no-better-than-Constant or worse optimum and can even meta-overfit. This coherently explains both the ε=1 tie/loss vs Constant and the worsening-with-T trend in the Δ-plots.

### Rigor concerns
- These are **Learned-only** curves — there is no equivalent showing the Constant/Dynamic baselines converging under the same ε=1 budget, so "the baseline is genuinely better" vs "both are noise-limited and Constant happens to land higher" can't be separated from these plots alone.
- The "no descent at ε=1" read is from a single architecture per grid; confirm it holds for the configs actually tabulated.
- Jitter here is outer-step variance within one run, **not** seed variance — it does not substitute for the missing seed error bars in the tables.

## Synthesis

- **Agreements:** Tables, main plots, and Δ-plots all tell the same ordered story — **Median-Clip < {Constant ≈ Dynamic} < Learned** for ε ≥ 3, on both datasets, across both architecture and T sweeps. The shape plots independently and consistently show the early-noise / mid-clip-hump trajectory and clean T-scaling.
- **Contradictions:** None internal. The only tension is presentation: the bolded tables make Learned look like a near-universal winner, while the Δ-plots honestly expose that **at ε=1 Learned ties or loses** to Constant/Dynamic (and the deficit *worsens with T* on fashion-mnist). Trust the Δ-plots — they show bands excluding 0.
- **Strongest supported claim:** *For ε ≥ 3, the learned σ/clip schedule beats Constant, Dynamic-DPSGD, and Median-Clip baselines on MNIST and Fashion-MNIST, with a margin that grows in ε and in model size (up to ~+2 points), and the learned schedule converges to a stable, interpretable early-noise / mid-training-clip-hump shape that scales predictably with T.*
- **Weakest link:** The **ε=1 regime** and the **variance/seed reporting**. Several bolded wins rest on gaps smaller than the (under-specified, sometimes `±0.000` or absent) variance, and the largest CNN's truncated training curves raise a matched-budget/survivorship question. The ε=1 result is the one most likely to be challenged — and it cuts *against* the method, so it should be reported, not buried.
- **Open questions:**
  1. What does `±` measure, and over how many seeds? Why are so many cells `±0.000` or blank? Re-report with consistent seed counts and a stated dispersion (std/CI).
  2. Were all four methods given equal tuning / equal outer-step and T budgets? Is the Constant baseline tuned or naive?
  3. Why is `cnn-64x128-head128` absent at ε=1 and truncated at ε≥3 — compute limit, divergence, or design? Are its accuracies at a matched outer-step budget?
  4. What is δ here, and is it fixed across all ε / T / datasets?
  5. Is the ε=1 Learned regression (worsening with T) an outer-loop optimization failure or a genuine property of budget-shaping under tight privacy? **The fashion-mnist ε=1 curves point to the former** — flat-from-start loss, no descent phase, meta-overfitting on mlp-512x256. Confirm with a Constant-baseline curve under the same ε=1 budget and with a few seeds; consider early-stopping/regularizing the outer loop at ε=1.
  6. Standardize uncertainty: arch_sweep_main has no bands while t_sweep_main does; the shape plots are overplotted spaghetti — cite the `by_T`/`by_arch` variants instead.
