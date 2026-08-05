# Interpretation: FirSweep

**Artifacts:** `src/cache/results/psaunder__FirSweep/`
- `scalars.parquet` (3836 rows = 959 runs × 4 methods), `schedules.parquet` (4.41M rows, full T-resolution), `missing.csv`
- `plots/{sgd-m0.0,sgd-m0.9}/t_sweep_table.{csv,tex}`, `t_sweep_main.png`, `t_sweep_delta_vs_{constant,dynamic}.png`
- `plots/{sgd-m0.0,sgd-m0.9}/shape_variants/{sigma,clip}_shape__T_sweep__by_T.png`, `sigma_shape.png`, `clip_shape.png`
- `plots/{sgd-m0.0,sgd-m0.9}/ladders/{mlp-width,cnn-width,cnn-depth}/table.csv`, `ladders/overall/arch_forest_delta.png`
- `t_sweep_table.csv` (top level — **stale**, see hygiene note)

**Date:** 2026-08-04

> **Provenance.** The parquets were re-fetched 2026-08-04 21:14 after the `compile_results_fetch.py` fix; the `plots/` tree on disk was stale (Jul 31 / Aug 3). I regenerated all figures and tables with
> `uv run compile_results_plot.py --in-dir cache/results/psaunder__FirSweep` before reading anything. **Everything below refers to the regenerated artifacts.** The previous version of this file was written against the data-starved fetch and its headline conclusion does not survive; see *What changed*.

---

## What changed with the new fetch

- **[shown]** Missing runs: **141 → 33**. All 108 `HTTP 500: parquet: could not read footer` failures are gone. The 33 remaining are 30 `run never started` + 3 stopped-early (steps 49/74 of 1000).
- **[shown]** Usable runs: **851 → 959**. The recovered runs are almost entirely the m0.0 / MNIST / T-sweep block.
- **[shown]** **m0.0 MNIST went from 3 of 16 populated cells to all 16 at n=8.** The old file could support no MNIST claim in that arm; it now carries a full 4×4 (ε × T) grid.
- **[shown]** Remaining thin cells are only m0.0 fashion-MNIST ε=10: n = 7/6/2/2 at T = 2000/3000/5000/7000.

The recovered MNIST block does not overturn the m0.0 trend — it *confirms* it on a second dataset, and simultaneously exposes that the trend reverses under momentum.

---

## Global setup

- Two **optimizer arms**, parallel directories: `sgd-m0.0` (no inner-loop momentum) and `sgd-m0.9` (momentum 0.9). Inner-loop momentum only — the outer schedule optimiser is plain GD in both.
- Two **axes**: `T-sweep` (T ∈ {2000,3000,5000,7000} × ε ∈ {3,5,8,10}, fixed arch `cnn-16x32-head32`, MNIST + Fashion-MNIST) and `arch` (T = 5000, ε = 10, three ladders, + CIFAR-10).
- Four **methods**: Learned Schedule, Dynamic-DPSGD, Adaptive Clip (Andrew et al.), Constant σ/clip. Metric is test accuracy (%), ↑ better. δ = 1e-6, batch 250, 8 seeds nominal.
- Schedule type is `DecoupledSigmaAndClipScheduleConfig` (`create_experiments.py:229`). **This matters for reading the shape plots** — see the σ/clip section.

### Setup facts absent from every figure

1. **`n_reps` is 1 for Learned and 8 for all three baselines** (exact, all 959 runs). Baseline cells are means over 8 seeds × 8 eval replicates; Learned is 8 seeds × 1. **The `±` columns are not comparable across columns** — Learned's spread is inflated by construction. `learned_acc_8rep` exists as a column and is null in all 3836 rows, so the intended fix was never populated. *(Unchanged from the previous fetch.)*
2. **Read-off step varies per figure and, in the arch ladders, per run.** m0.0 T-sweep reads at outer step 1000; m0.9 T-sweep at 978–1000; arch ladders read each run at *its own* last completed step. Only the Learned column depends on outer step at all, so this is a one-sided bias (§ CIFAR).
3. **Runs were configured for 1000 outer steps**, but `create_experiments.py:114` currently reads `NUM_OUTER_STEPS = 3000`. Re-running the generator today would not reproduce this sweep.

---

## `plots/{sgd-m0.0,sgd-m0.9}/t_sweep_table.csv` + `t_sweep_main.png` + delta plots

**Setup:** Rows = (dataset, ε, T); columns = four methods, `mean ± spread` test accuracy, bold = per-row best. Main plot: x = T (linear), y = accuracy, panels ε (cols) × dataset (rows), shaded = 95% CI. Delta plots: y = Learned − baseline.

### What it shows

**m0.0 — Learned wins 25 of 32 rows; the margin decays monotonically in T and crosses zero.**

- **[shown]** Learned − Constant, fashion-MNIST ε=3: **+2.55 / +1.36 / +0.12 / −0.04** at T = 2000/3000/5000/7000. The same monotone decay holds in all 8 (ε, dataset) panels.
- **[shown]** Same on the recovered MNIST block, ε=10: **+0.87 / +0.53 / +0.10 / −0.19**. Constant takes the bold in all four MNIST T=7000 rows and in MNIST ε=3,5 at T=5000.
- **[shown]** Learned − Dynamic-DPSGD decays identically (+2.90 → +0.15 at FMNIST ε=3) but stays ≥ 0 in 31 of 32 rows.
- **[shown]** Adaptive Clip is last in all 32 rows by a wide margin: FMNIST 73.5–80.3 (vs 81–86); MNIST 89.9–94.6 (vs 94.5–96.8).
- **[shown]** ε is nearly inert. FMNIST T=2000, Learned: 83.888 → 84.271 across ε = 3 → 10 — **a 3.3× budget increase buys +0.38 pts**. Constant moves +0.09.

**m0.9 — the Learned advantage disappears. Learned wins 4 of 32 rows; Adaptive-Clip wins 18, Dynamic-DPSGD 10.**

- **[shown]** On fashion-MNIST, Learned wins **0 of 16** rows. Learned − Dynamic ranges −0.39 to +0.00; Learned − Adaptive-Clip −0.41 to +0.45.
- **[shown]** On MNIST, Learned wins 4 rows (ε=3,5 at T=5000,7000) and is within ±0.2 of the best elsewhere.
- **[shown]** **The delta-vs-Constant trend reverses sign in T between arms**: m0.0 decays with T (+2.5 → 0), m0.9 *grows* with T (MNIST ε=3: +0.22 → +1.41). The m0.9 growth is driven by **Constant degrading**, not Learned improving — Constant MNIST ε=3 falls 95.411 → 94.838 from T=2000 → 7000, with spread widening to ±0.978.
- **[shown]** m0.9 CIs are wide relative to the effect: every fashion-MNIST delta-vs-Constant band straddles zero. m0.9 MNIST bands exclude zero from T=3000 up.

**Momentum lift (m0.9 − m0.0), averaged over all 32 cells:**

| Method | Mean lift (pts) |
|---|---|
| Adaptive Clip | **+6.115** |
| Dynamic-DPSGD | +1.271 |
| Constant σ/clip | +0.553 |
| **Learned Schedule** | **+0.175** |

- **[shown]** The Learned schedule is nearly momentum-invariant; every baseline gains substantially. At FMNIST T=2000 the lift is +0.57 (Learned) vs +3.39 (Constant) vs +10.92 (Adaptive-Clip).
- **[inferred]** The learned schedule, without momentum, already recovers most of what momentum supplies — so the two are largely redundant and the advantage collapses when both are present. Assumes the two act through the same channel (effective per-step step size). Independently supported by the σ/clip decomposition below.
- **[not shown]** No hyperparameter table. Whether the inner learning rate or Adaptive-Clip's target quantile / clip-LR were tuned per arm is displayed nowhere. Adaptive-Clip's +6.1 pt momentum sensitivity is exactly what a mis-tuned baseline looks like.

### Rigor concerns

- **The `±` is not comparable across columns** (n_reps 1 vs 8). "Learned has the widest error bars" is an artefact of the evaluation protocol, not of the method. Populate `learned_acc_8rep` before any of these tables goes in a write-up.
- **m0.0 fashion-MNIST ε=10, T=5000/7000 are n=2** (`85.250 ± 3.177`, `85.925 ± 3.494`) and are still bolded as Learned wins. The ε=10 delta panel shows the CI fan opening to ±4. Drop, re-run, or un-bold — do not quote them.
- **Adaptive Clip's m0.0 collapse is a baseline-integrity problem, not a result.** A published method landing 7–10 pts below *constant* σ/clip is far likelier mis-configured than genuinely that bad, and the m0.9 arm proves the code can do better (+6.1 pts, same code). As presented it inflates the apparent margin of every other method in the m0.0 arm.
- **`t_sweep_main.png` (m0.0) shares one y-axis across all 8 panels**, set by Adaptive-Clip's 73.5 floor, compressing Learned/Dynamic/Constant into the top ~15% of each panel. The delta plots are the only readable view.
- **The two arms' main plots use different y-limits** (m0.0 ≈ 73–92; m0.9 ≈ 83.5–86.5). Flipping between the two figures will badly misjudge the arm difference. They are not visually comparable.
- **Legend text overlaps the caption** in `t_sweep_main.png` and `t_sweep_delta_*.png` for both arms — the caption is partly illegible. Rendering bug in the plot script.
- **Direction of merit (↑ better) is stated on no figure or table.**

---

## `shape_variants/{sigma,clip}_shape__T_sweep__by_T.png` + the σ/clip decomposition

**Setup:** x = t/T (0–1), y = σ (or clip), panels ε (rows) × dataset (cols), colour = T, thick line = seed mean, thin = per-seed. **These are the cleanest figures in the batch** — per-seed lines hug the mean almost everywhere.

**Critical reading note.** Under `DecoupledSigmaAndClipSchedule`, the logged `sigma` column is `get_private_noise_scales() = clip_t · s_t`, and `get_private_weights() = 1/s_t`. So the privacy-constraint variable is
**w_t = clip_logged / σ_logged**, and the GDP constraint is Σ_t exp(w_t²) = (μ/p)² + T. **The plotted σ curve is not the privacy allocation** — it is the allocation multiplied by the (privacy-free) clip envelope. Reading the σ plot as "where the budget is spent" is wrong.

### What it shows

- **[shown]** Both σ and clip trace the same "mesa": rise from near zero at t/T=0, plateau through the middle, decay to near zero at t/T=1. Peak at t/T ≈ 0.4 (m0.0), ≈ 0.25–0.3 (m0.9).
- **[shown]** Amplitude scales inversely with T. m0.0 FMNIST ε=3 clip peaks at ≈ 10.7 (T=2000) → 4.5 (T=7000); σ peaks ≈ 6.5 → 3.5.
- **[shown]** m0.9 magnitudes are ~6× smaller than m0.0 (clip peak ≈ 1.5 vs ≈ 10.7), consistent with momentum accumulating gradient magnitude.
- **[shown]** **I verified the privacy constraint directly.** Computing Σ exp(w_t²) at full resolution over all 512 T-sweep runs and comparing to (μ/p)² + T with μ = `approx_to_gdp(ε, 1e-6)`: the ratio is **1.0000–1.0037** (mean 1.0000). The projection is tight and the learned schedules are budget-valid. This closes the concern from the July decoupled-projection bug.
- **[shown]** **The learned privacy allocation is close to uniform.** Median w_t vs the uniform-allocation w_unif = √(ln(target/T)), in all 32 cells of both arms, agrees to within ~2% (e.g. m0.0 MNIST ε=10 T=2000: w_med 2.171 vs w_unif 2.144; ε=3 T=7000: 1.253 vs 1.222).
- **[shown]** Budget mass exp(w²) per decile of training (uniform = 10.0%):

  | Arm / cell | d0 | d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8 | d9 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | m0.0 ε=10 T=2000 | 1.3 | 4.4 | 9.3 | 10.5 | 13.0 | 14.0 | **14.8** | 14.3 | 11.6 | 6.8 |
  | m0.0 ε=3 T=7000 | 4.9 | 7.8 | 9.6 | 10.8 | 11.6 | 12.2 | **12.5** | 12.1 | 10.7 | 7.9 |
  | m0.9 ε=10 T=2000 | 4.0 | 7.9 | 10.6 | 12.3 | 12.9 | **13.0** | 12.8 | 11.7 | 9.4 | 5.4 |

  The policy spends **slightly less than uniform in the first ~15% and last ~10%, slightly more in the middle**. Deciles 2–8 stay within 9–15% throughout.
- **[shown]** The within-run spread of the privacy weight is small: max-decile/min-decile w ratio is 1.28–1.42 (m0.0 MNIST), 1.17–1.40 (m0.9). Meanwhile the *logged* σ and clip each swing ~30× within a run.
- **[inferred]** **Most of what the outer loop learns is a clip (step-size) envelope, not a privacy reallocation.** The privacy-relevant variable moves ~1.3×; the unconstrained clip moves ~30×, in a warmup-then-decay shape. Assumes clip enters utility principally as an effective step size (true for Abadi clipping when most per-sample gradients are clipped). **This is the most consequential inference in this document and it is not directly tested by any artifact here.**
- **[inferred]** This explains the momentum result: momentum supplies its own effective-step-size adaptation, so the learned clip envelope becomes redundant and the Learned-vs-baseline gap closes (lift +0.175 vs +0.55/+1.27/+6.12). Two independent lines of evidence agree.
- **[not shown]** No ablation isolating the two factors. Nothing here separates "learned w, constant clip" from "uniform w, learned clip".

### Rigor concerns

- **`sigma_shape.png` (m0.9) is unreadable.** A handful of CIFAR outliers reach σ ≈ 330, so the y-axis runs 0–330 and all three panels render as a flat line at zero. Same defect at lower severity in the m0.0 version (CIFAR panel dominated by one ε=10 spike). Clip to a percentile or use a log axis. The `shape_variants/*_by_T.png` figures are the usable ones.
- **The σ curves are plotted without noting that σ_logged = clip · s.** Anyone reading `sigma_shape.png` as the privacy schedule will draw the wrong conclusion. The axis label should say so, or the plot should show w_t = clip/σ directly. **A `w_t` (or budget-mass-per-decile) panel is the single most valuable figure missing from this batch.**
- The endpoint dips (σ → ~0.2 at t/T = 0 and 1) look alarming — w = 1/s there would be large and exp(w²) explosive — but the constraint check above confirms the totals hold, so these are the clip envelope going to zero, not a budget leak.

---

## `ladders/{mlp-width,cnn-width,cnn-depth}/table.csv` + `ladders/overall/arch_forest_delta.png`

**Setup:** T = 5000, ε = 10, 8 seeds, three ladders × three datasets. Forest plot: x = Learned − Constant (Δ acc), y = rungs grouped by ladder, one panel per dataset, dots = per-seed.

### What it shows

- **[shown]** On MNIST and fashion-MNIST, m0.0 Learned takes the bold in **18 of 19** rows (the exception is MNIST `cnn-16x32-head64`, where Constant wins by 0.013); margins are small and consistent (+0.2 to +1.0 vs Constant, e.g. MNIST mlp-512 94.313 vs 93.066 = +1.25).
- **[shown]** On CIFAR-10 m0.0, Learned **loses to Constant systematically in the cnn-width ladder**: 43.650 vs 45.698, 46.981 vs 49.032, 49.750 vs 51.047 (−1.3 to −2.1 pts, spreads ≤ 0.6 — not outlier-driven).
- **[shown]** But on CIFAR-10 m0.0 cnn-**depth**, Learned wins large: 60.169 vs 56.152 at `cnn-16x16x16-head64` (**+4.0**), 54.913 vs 52.837 at 4-deep. The two CNN ladders disagree in sign on the same dataset.
- **[shown]** **m0.9 CIFAR-10 Learned collapses.** `mlp-128` 35.850 ± 10.481 (Dynamic 44.772 ± 0.480); `mlp-512` 36.213 ± 10.310; `cnn-16x16x16` 46.393 ± **19.253**; `cnn-16x16x16x16` 35.808 ± 14.374.
- **[shown]** Collapse rate, quantified from `scalars.parquet`: **12 of 77 (15.6%) of m0.9 CIFAR Learned runs land below 35% accuracy. Zero of 80 m0.0 CIFAR Learned runs do, and zero of the 471 CIFAR baseline runs do** (baseline minimum across all methods/arms is 35.76).
- **[shown]** No collapse anywhere in the T-sweep: minimum Learned accuracy is 82.70 (FMNIST) / 95.20 (MNIST) in both arms.
- **[shown]** **The collapses are not simply truncation.** The three failing `mlp-128` m0.9 seeds reached outer steps 851/852/854 of 1000 — near-fully-optimised policies scoring 18.1/19.4/19.9%. (Some others are truncated *and* collapsed: `cnn-16x16x16` seed 2 stopped at step 128 → 8.95%, below chance-adjacent 10%.)
- **[inferred]** This is the known outer-loop divergence instability (`project_policy_instability_nan_grads`) surfacing specifically in the (momentum × CIFAR-10) corner. Assumes the collapsed runs are the same failure mode; not verified from logs here.
- **[not shown]** Nothing in the tables or forest plot marks which cells contain collapsed seeds. The mean ± std silently mixes converged and diverged runs.

### Rigor concerns

- **Reporting mean ± std over a bimodal population is not meaningful.** `46.393 ± 19.253` describes no run that actually happened. These cells need either the collapse rate reported alongside a converged-only mean, or a median.
- **CIFAR arch runs are truncated, and truncation is one-sided.** Median final outer step by rung (m0.0/m0.9): `mlp-512` **134/139**, `cnn-16x16x16` 698/496, `cnn-16x16x16x16` 681/690, `cnn-32x64` 656/705. Only `cnn-8x16-head64` and `mlp-64` reach 1000 in either arm. Baselines are fixed schedules and do not depend on the outer step at all — **so truncation degrades only the Learned column.** Every CIFAR Learned-vs-baseline comparison is biased against Learned by an unknown amount.
  - Mitigating evidence: m0.0 `mlp-512` CIFAR reads at step 134 and still scores 44.50 ± 0.90, so truncation alone is not catastrophic. The m0.9 collapses are something else.
- **`arch_forest_delta.png` shares one x-axis (−50 to +10) across all three dataset panels.** The scale is set by the CIFAR collapses, so the MNIST and fashion-MNIST panels — where the real, consistent ±1 pt effect lives — render as an indistinguishable vertical stripe at zero. The figure is unusable for its stated purpose ("architecture robustness") on two of three datasets. Per-panel x-limits, or a separate CIFAR panel, would fix it.
- **The cnn-width vs cnn-depth sign disagreement on CIFAR m0.0 is unexplained** and both ladders read at similar truncation levels, so truncation does not obviously account for it.
- **MNIST m0.0 `mlp-64` has 0 seeds and `mlp-128` has 5** — the mlp-width ladder is incomplete in that arm; `mlp-64` is silently absent from the m0.0 MNIST table rather than marked missing.

---

## File hygiene

- **`t_sweep_table.csv` at the directory root is stale** (Jul 31) and now differs from `plots/sgd-m0.9/t_sweep_table.csv` in 20 of 32 rows. It is not written by `compile_results_plot.py` and did not regenerate. Delete it — it is a live footgun, identical in name to the per-arm tables and wrong.

---

## Synthesis

**Agreements.**
- The T-sweep tables, the delta plots, and the momentum-lift table all say the same thing: the learned schedule's advantage is a **small-T, no-momentum** effect. It decays monotonically in T across all 8 (ε, dataset) panels and vanishes when momentum is on.
- The σ/clip decomposition and the momentum-lift table agree mechanistically: the learned policy barely reallocates privacy (w within ~2% of uniform) while learning a large clip envelope, and momentum — which supplies similar step-size adaptation — makes that envelope redundant (+0.175 pts lift vs +0.55/+1.27/+6.12 for the baselines).
- MNIST and fashion-MNIST agree in the m0.0 arm across both axes. The recovered MNIST block is confirmatory, not contradictory.

**Contradictions.**
- CIFAR-10 cnn-**width** (Learned loses to Constant by 1.3–2.1 pts, tight spreads) vs cnn-**depth** (Learned wins by up to +4.0) in the same arm, same dataset, same ε and T. Unresolved. Both ladders are truncated similarly, so truncation is not an obvious explanation. **Trust neither CIFAR ladder until the runs are trained to 1000 outer steps.**
- m0.0 says Learned > Constant at T=2000; m0.9 says Learned ≈ Constant at T=2000 but Learned > Constant at T=7000. These are consistent once you notice the m0.9 result is Constant *degrading* with T, not Learned improving — but the two arms' main plots, with their different y-limits, invite the wrong reading.

**Strongest supported claim.** *Without inner-loop momentum, a learned σ/clip schedule beats constant and Dynamic-DPSGD baselines at small step budgets, and the advantage decays to zero as T grows.* This rests on 25/32 winning rows across two datasets and four ε, a monotone decay in all 8 panels, n=8 in 28 of 32 cells, and verified-tight privacy accounting (Σexp(w²)/target ∈ [1.0000, 1.0037]).

A close second, and more interesting: *the learned schedule is nearly momentum-invariant while every baseline gains 0.55–6.1 pts from momentum* — a clean, large, consistently-signed effect across all 32 cells.

**Weakest link.** The **CIFAR-10 architecture ladders**. They are simultaneously truncated (median outer step as low as 134 of 1000), one-sidedly biased (truncation hits only the Learned column), internally contradictory (width vs depth disagree in sign), and contaminated by a 15.6% divergence rate in the m0.9 arm that is being averaged into means ± std rather than reported. Nothing about CIFAR in this batch should be relied on.

Runner-up: **Adaptive Clip in the m0.0 arm**. A 7–10 pt deficit against constant σ/clip, in a method that performs *best* in the other arm, is much more likely a configuration error than a finding. It inflates every m0.0 margin it appears next to.

**Open questions.**
1. **The ablation that decides the paper's framing:** uniform-w + learned clip, vs learned w + constant clip, vs both learned. If uniform-w + learned-clip matches full Learned, the contribution is a learned *clipping* schedule and should be framed that way — the near-uniform w profile above predicts it will.
2. Was Adaptive-Clip's target quantile / clip learning rate tuned separately for m0.0? A single shared setting tuned under momentum would explain the whole collapse.
3. Is the m0.9 CIFAR divergence the known outer-loop NaN-grad instability, and does `max_grad_norm` / `zero_nans` clipping suppress it? The 12 affected (arch, seed) pairs are identified above; checking their outer-loop traces is cheap.
4. Re-run the CIFAR arch cells to a full 1000 outer steps, then re-read. Until then the arch axis supports MNIST/FMNIST conclusions only.
5. Populate `learned_acc_8rep` (null in all 3836 rows) so the `±` columns become comparable, then re-check which of the sub-0.2-pt "wins" survive.
6. Do the 30 `run never started` seeds need re-launching, or is n=6–7 acceptable for the affected cells? (m0.0 FMNIST ε=10 T=5000/7000 at n=2 do need it.)

---

## Suggested figure fixes (all in `compile_results_plot.py`)

1. Per-panel x-limits on `arch_forest_delta.png`, or split CIFAR out — currently unusable on 2 of 3 datasets.
2. Percentile-clip or log-scale the y-axis on `sigma_shape.png` — m0.9 renders as a flat line.
3. Match y-limits across the two arms' `t_sweep_main.png`, or annotate that they differ.
4. Fix the legend/caption overlap on `t_sweep_main.png` and `t_sweep_delta_*.png`.
5. Add a **w_t = clip/σ** (or budget-mass-per-decile) shape figure — the privacy allocation is currently not plotted anywhere, and the σ plot is routinely mistaken for it.
6. Mark truncated cells (final outer step < 1000) in the ladder tables, and report collapse counts instead of folding diverged seeds into mean ± std.
7. Delete the stale root `t_sweep_table.csv`.
