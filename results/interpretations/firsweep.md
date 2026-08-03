# Interpretation: FirSweep

**Artifacts:** `src/cache/results/psaunder__FirSweep/`
- `t_sweep_table.csv` (top level; identical to `plots/sgd-m0.9/t_sweep_table.csv`)
- `plots/sgd-m0.0/t_sweep_table.csv`, `plots/sgd-m0.9/t_sweep_table.csv`
- `plots/{sgd-m0.0,sgd-m0.9}/t_sweep_main.png`, `t_sweep_delta_vs_constant.png`
- `plots/{sgd-m0.0,sgd-m0.9}/sigma_shape.png`, `clip_shape.png`
- `plots/{sgd-m0.0,sgd-m0.9}/shape_variants/{sigma,clip}_shape__T_sweep__by_T.png`
- `plots/{sgd-m0.0,sgd-m0.9}/ladders/{mlp-width,cnn-width,cnn-depth}/table.csv`
- `plots/{sgd-m0.0,sgd-m0.9}/ladders/overall/arch_forest_delta.png`
- `plots/sgd-m0.9/curves/t_sweep_acc__mnist.png`
- `missing.csv`, `scalars.parquet` (used to check n and read-off convention)

**Date:** 2026-07-31

---

## Global setup (applies to everything below)

- Two **optimizer arms**, plotted in parallel directories: `sgd-m0.0` (no inner-loop momentum) and `sgd-m0.9` (momentum 0.9). Everything else is nominally matched.
- Two **axes**: `T-sweep` (T ∈ {2000, 3000, 5000, 7000} × ε ∈ {3, 5, 8, 10}, fixed arch `cnn-16x32-head32`, MNIST + Fashion-MNIST) and `arch` (fixed T = 5000, ε = 10, arch ladders, + CIFAR-10).
- Four **methods**: Learned Schedule, Dynamic-DPSGD, Adaptive Clip (Andrew et al.), Constant σ/clip. Metric is test accuracy (%), ↑ better.
- **8 seeds** per cell nominally.

⚠ **The single most important setup fact, and it is nowhere on the figures:** from `scalars.parquet`, `n_reps = 1` for **Learned Schedule** and `n_reps = 8` for **all three baselines**. Every baseline number is a mean over 8 seeds × 8 evaluation replicates (64 draws); every Learned number is a mean over 8 seeds × 1 replicate (8 draws). The `±` columns are therefore not comparable across columns.

---

## `plots/sgd-m0.0/t_sweep_table.csv` + `t_sweep_main.png` + `t_sweep_delta_vs_constant.png`

**Setup:** Rows = (dataset, ε, T). Columns = the four methods, `mean ± spread` test accuracy. Bold = per-row best. Plot: x = T (2000–7000, linear), y = test accuracy, one panel per ε (columns) × dataset (rows), shaded = 95% CI. Caption: *"n = 1–9 seeds; read off at outer step 1000."*

### What it shows
- **[shown]** Learned wins 26 of 32 rows. The wins are concentrated at small T: at Fashion-MNIST/T=2000 the Learned–Constant gap is +2.5 to +2.9 pts across all four ε; by T=7000 it is ≈ 0.0 to −0.2.
- **[shown]** The delta plot is monotone decreasing in T in all 8 panels, crossing zero at T ≈ 5000 (Fashion-MNIST) and T ≈ 5000–5500 (MNIST). At T=7000 Constant is bolded in 5 of 8 rows.
- **[shown]** Adaptive Clip is catastrophically bad in this arm: Fashion-MNIST 73.5–80.3 vs 81–85 for everything else (−7 to −10 pts); MNIST 89.9–94.6 vs 94.5–96.8 (−2 to −5 pts). It is last in all 32 rows.
- **[shown]** ε is nearly inert. Averaging over T, the ε=3→ε=10 accuracy change is +0.44 (Learned FMNIST), +0.19 (Constant), +0.15 (Dynamic), and **−0.03 / +0.02 for Adaptive Clip** — i.e. a 3.3× privacy-budget increase buys ≲ 0.5 pt for anyone, and literally nothing for Adaptive Clip.
- **[inferred]** The Learned advantage is a *budget-allocation-under-scarcity* effect, not a general one — assumes accuracy at large T is optimisation-limited rather than noise-limited, in which case any valid schedule reaches the same ceiling and the learned shape has nothing to buy.
- **[inferred]** The ε-invariance plus the Adaptive-Clip collapse jointly suggest this arm is *clip/step-size limited*, not noise limited — assumes σ is not the binding constraint at m=0.0.
- **[not shown]** No per-arm hyperparameter table. Whether the inner learning rate was retuned for m=0.0 vs m=0.9 (and whether the Andrew-et-al. baseline's clip-quantile / learning-rate were tuned at all in this arm) is not displayed.

### Rigor concerns
- **Fashion-MNIST, ε=10, T=5000 has no `±` at all** — that cell is **n = 1 seed** (`85.000` / `84.131` / `78.094` / `84.638`). It is bolded as a Learned win. The adjacent T=7000 cell is n = 2 (`85.925 ± 3.494`). The upper-right panel of `t_sweep_main.png` and the corresponding delta panel show a CI fan opening to ±4 pts. **Those two cells carry no evidence and should be dropped or re-run**, not bolded.
- **The ± is not comparable across columns** (n_reps 1 vs 8, see global note). Learned's spread is inflated relative to baselines by construction, so "Learned ± is larger" is an artefact, not a finding.
- **Bolding vs noise:** at T=5000/7000 the Learned–Constant gap (≈ 0.0–0.2) is well inside the ±0.2–0.6 spreads. The T=7000 bolds are not distinguishable; the T=2000 bolds (+2.5 pts vs ±0.5) are.
- **Read-off convention differs between arms.** The m0.0 caption says *"read off at outer step 1000"*; the m0.9 caption says *"read off at outer steps 976–1000"*. Two arms presented side by side should not use two read-off rules — the single-step read-off is a one-sample draw from a visibly noisy band (see the training-curve artifact below).
- **Adaptive Clip's collapse is a baseline-integrity problem, not a result.** A published method landing 10 pts below *constant* σ/clip is far more likely mis-tuned than genuinely that bad. As presented it inflates the apparent margin of everything else.
- **Survivorship:** `missing.csv` drops 28 m0.0 T-sweep runs, non-uniformly (all of the Fashion-MNIST ε=10 damage is here). Which seeds vanished is not random with respect to outcome — 17 of them are *"no test-accuracy rows in run history"*, i.e. runs that failed to log, and 12 of those are Fashion-MNIST ε=10.

---

## `plots/sgd-m0.9/t_sweep_table.csv` (= top-level `t_sweep_table.csv`) + `t_sweep_main.png`

**Setup:** As above, momentum-0.9 arm. Caption: *"n = 6–8 seeds; read off at outer steps 976–1000."*

### What it shows
- **[shown]** The result **inverts**. Learned wins 4 of 32 rows (all MNIST, T ∈ {5000, 7000}, ε ∈ {3, 5}). Adaptive Clip wins 19, Dynamic-DPSGD wins 9, Constant wins 0.
- **[shown]** All four methods collapse into a ≈ 0.3–0.8 pt band. Fashion-MNIST ε=3 T=2000: 84.24 / 84.32 / 84.55 / 84.39 — a 0.31 pt total spread against ±0.17–0.53 error bars.
- **[shown]** Adaptive Clip is now *competitive-to-best* (85.0 → 86.3 on Fashion-MNIST) instead of last. Same code, same ε, same T — only inner momentum changed.
- **[shown]** Constant is now clearly *worst* and has the largest spread (e.g. MNIST ε=3 T=5000: `94.821 ± 1.119` vs `96.267 ± 0.214`), and its MNIST curve *decreases* with T in the ε=3/5 panels.
- **[shown]** ε sensitivity recovers: ε=3→10 gains are 0.49–1.05 pts (vs 0.02–0.44 at m=0.0).
- **[shown]** The absolute ceiling is higher: best Fashion-MNIST 86.34 (m0.9) vs 85.93 (m0.0); best MNIST 96.79 vs 96.82 (a wash).
- **[inferred]** Momentum supplies most of what the learned schedule was supplying at m=0.0 — assumes the two arms are otherwise matched, which the artifacts do not establish.
- **[not shown]** No paired statistical test. With four methods inside a 0.3 pt band and 6–8 seeds, per-row bolding is picking noise.

### Rigor concerns
- **Multiple comparisons / bolding is meaningless here.** 32 rows × 4 methods, each row bolds a winner, and in most rows the top-3 overlap within one standard deviation. "Adaptive Clip wins 19/32" should not be reported as a finding without a paired test across seeds.
- **The two arms disagree so violently that at least one is mis-specified.** Adaptive Clip moves from −10 pts to +0.3 pts against Constant purely from momentum. That is not a plausible property of the method; it is a symptom that one arm's baseline configuration is wrong.
- **n = 6–8 seeds is uneven across cells** (from `scalars.parquet`: MNIST ε=3 T=5000 has 6 Learned seeds, Fashion ε=10 T=5000 has 6). The caption gives the range, not the per-cell n, so a reader cannot tell which cells are thin.

---

## `plots/{sgd-m0.0,sgd-m0.9}/shape_variants/{sigma,clip}_shape__T_sweep__by_T.png`

**Setup:** Learned σ (and clip) vs normalised step t/T ∈ [0,1]. Rows = ε ∈ {3,5,8,10}, columns = {fashion-mnist, mnist}. Bold line = seed mean, thin lines = individual seeds. Colour = T.

### What it shows — this is the strongest artifact in the set
- **[shown]** A consistent **inverted-U**: σ starts near 0, rises over the first 20–40 % of training, plateaus, and decays back toward ~0 at t/T = 1. Reproduced in all 8 (ε, dataset) panels × both arms × all four T. Clip follows the same shape (σ and clip are coupled through the schedule parametrisation).
- **[shown]** σ scales **down with T** exactly as GDP composition predicts. m0.0 MNIST ε=5 peak: T=2000 → 4.7, T=3000 → 4.2, T=5000 → 3.5, T=7000 → 3.0. Ratios 4.7/3.0 = 1.57 vs √(7000/2000) = 1.87 — same direction, somewhat shallower than √T.
- **[shown]** σ scales **down with ε**: m0.0 MNIST T=2000 peak 5.2 (ε=3) → 4.7 (ε=5) → 4.3 (ε=8) → 4.2 (ε=10).
- **[shown]** **Seed agreement is excellent** in the plateau region — thin per-seed lines are visually indistinguishable from the mean for t/T > 0.25 in most panels. Disagreement is confined to t/T < 0.2.
- **[shown]** **Scale differs ~10× between arms**: m0.0 σ peaks at 3–6 and clip at 6–11; m0.9 σ peaks at 0.3–0.9 and clip at 1.0–1.8.
- **[shown]** **Skew differs between arms**: m0.0 Fashion-MNIST peaks at t/T ≈ 0.4–0.5 (roughly symmetric); m0.9 Fashion-MNIST peaks at t/T ≈ 0.25–0.30 with a long right tail.
- **[shown]** m0.9 MNIST panels have an odd boundary artefact: σ starts *high* (≈ 0.3), dips at t/T ≈ 0.05, then rises into the hump.
- **[inferred]** The 10× scale gap is the momentum gain 1/(1−0.9) = 10 — the optimiser found the same *effective* step size in both arms, with (σ, C) jointly rescaled. Assumes the decoupled schedule leaves the absolute (σ, C) scale free with only the ratio privacy-constrained (which is what `decoupled-sigma-and-clip` does). **This is a good consistency check and worth stating explicitly in the write-up.**
- **[inferred]** The early-training instability (thin lines diverging, spikes to σ = 13 and σ = 330) is the outer loop still moving before the schedule settles — assumes early t/T control points get weak gradient signal.
- **[not shown]** No overlay of what Constant / Dynamic-DPSGD / Adaptive Clip actually use, so the reader cannot see *how far* the learned shape departs from the baselines it is being compared to.
- **[not shown]** No functional form fitted. The shape is clean enough to be worth a closed form (this is the input the SR pipeline wants).

### Rigor concerns
- **`plots/sgd-m0.9/sigma_shape.png` is unreadable.** A single CIFAR-10 seed reaching σ ≈ 330 sets the y-limit, flattening all three panels to a line at zero. That figure conveys no information and should be re-rendered with a robust y-limit or on a log axis. The m0.0 version has the same problem in milder form (a σ = 13 outlier in the Fashion-MNIST panel).
- **The aggregate `sigma_shape.png` / `clip_shape.png` pool over ε, T, seed, and arch simultaneously.** The `by_T` variants are strictly more informative; the pooled versions mostly show variance.
- **Shape ≠ benefit.** These plots show the schedule is *reproducible*; they say nothing about whether it *helps*. The accuracy tables say it helps only at small T in the m=0.0 arm. Do not let the crispness of these curves carry an efficacy claim.

---

## `plots/{sgd-m0.0,sgd-m0.9}/ladders/*/table.csv` + `ladders/overall/arch_forest_delta.png`

**Setup:** Fixed T = 5000, ε = 10. Three ladders — mlp-width (64/128/512), cnn-width (8x16 / 16x32 / 32x64), cnn-depth (1–4 conv blocks) — × three datasets. Forest plot: x = Learned − Constant Δ accuracy (pts), diamond = seed mean, dots = per seed, line = 95 % CI, dashed = 0.

### What it shows
- **[shown] m0.0 arm:** Learned beats Constant in 26 of 29 ladder rows. The largest wins are on **CIFAR-10 cnn-depth**: +4.0 (16x16x16) and +2.1 (16x16x16x16) pts. It *loses* on **CIFAR-10 cnn-width**: −1.8, −2.1, −1.3 pts across the three rungs, with CIs excluding zero.
- **[shown]** Adaptive Clip is again last in every one of the 29 m0.0 rows, and degrades with depth (CIFAR 45.2 → 36.2 as depth goes 1 → 4).
- **[shown] m0.9 arm:** the CIFAR-10 panel of the forest plot needs an x-range of **−50 to +10** (vs −4 to +5 for m0.0). Individual seeds sit at −48, −37, −27 pts.
- **[shown]** Diverged m0.9 cells: `cifar-10 cnn-16x16x16 46.393 ± 19.253`, `cnn-16x16x16x16 35.808 ± 14.374`, `mlp-128 35.850 ± 10.481`, `mlp-512 36.213 ± 10.310`.
- **[shown]** From `scalars.parquet`, **11 of 464** m0.9 Learned runs land > 5 pts below their cell median. **All 11 are CIFAR-10, arch-sweep, ε = 10.** Accuracies: 8.95 (chance is 10), 17.65, 18.10, 19.15, 19.40, 19.60, 19.90, 22.30, 22.75, 22.90, 41.90. The m0.0 arm has **0 such runs out of 437**.
- **[shown]** **CIFAR-10 arch-sweep runs are systematically under-trained in the outer loop.** Median `final_outer_step` for Learned: CIFAR-10 703 (m0.0) / 725 (m0.9), minimum 124 / 55 — versus 1000 for MNIST and Fashion-MNIST. 27 % of all arch-sweep Learned runs stopped before step 1000.
- **[inferred]** The m0.9 CIFAR failures are outer-loop divergence (schedule blow-up), not inner-loop noise — supported by the σ ≈ 330 trace in `sgd-m0.9/sigma_shape.png` and by accuracy landing at chance. Assumes the σ outlier and the 8.95 % run are the same run family; not directly verified.
- **[inferred]** The mean ± std cells above are **means over a bimodal mixture** (a converged mode near 57–59 and a diverged mode near 10–23), so both the mean and the std are meaningless summaries there.
- **[not shown]** No divergence/failure-rate column. A reader of `ladders/cnn-depth/table.csv` sees `46.393 ± 19.253` and cannot tell it is "6 runs at ~59 and 2 at ~18."

### Rigor concerns
- **Under-training is confounded with dataset.** CIFAR-10 is exactly (a) the hardest dataset, (b) the one where Learned loses, and (c) the one where the outer loop ran ~30 % fewer steps. Any claim of the form "the learned schedule doesn't transfer to CIFAR" is unidentifiable from this data until the outer budget is matched.
- **Report failure rate, not just mean ± std**, for any cell containing diverged runs. Better: report median and an explicit "n diverged / n total".
- **The m0.9 CIFAR forest panel is uninformative as drawn** — the −50 range needed by the outliers compresses every real effect to invisibility. Split the outliers out or clip the axis with an annotation.
- **Ladders use a single (ε=10, T=5000) point.** m0.9 CIFAR wants ε=10 σ values that appear to be at the edge of solver stability. Whether the ladder conclusion is architecture-dependent or just ε=10-dependent is untested.
- **The mlp-width MNIST ladder is missing the `mlp-64` rung in the m0.0 arm** (the table has only mlp-128 and mlp-512). `missing.csv` shows all 8 `sgd-m0.0/dsMNIST/e10/arch-sweep/T=5000/mlp-64/seed=*` runs dropped with *"no test-accuracy rows"* — a whole rung lost, silently, to a logging failure.

---

## `plots/sgd-m0.9/curves/t_sweep_acc__mnist.png`

**Setup:** Learned-schedule test accuracy vs **outer step** (0–1000), one panel per ε × T. Single trace per panel (no band).

### What it shows
- **[shown]** Every panel rises steeply over the first ~25–50 outer steps and is then **flat for the remaining ~950 steps**, oscillating in a ±0.7–1.0 pt band with no visible trend.
- **[shown]** The oscillation amplitude (≈ ±0.8 pts) is **larger than every effect size in the accuracy tables** (0.1–0.9 pts).
- **[inferred]** The outer loop converges by step ~50 and the remaining 95 % of the outer budget buys nothing measurable — assumes this MNIST panel is representative (the Fashion-MNIST panels look the same; CIFAR-10 was not plotted this way).
- **[inferred]** Reading a single outer step therefore samples the oscillation, not the converged value. The m0.0 arm's single-step read-off adds ~±0.8 pt of pure read-off noise on top of seed noise. The m0.9 arm's 976–1000 window averages 25 samples and should be much tighter — **which is itself a systematic difference between the two arms that has nothing to do with momentum.**
- **[not shown]** No CIFAR-10 training curves, i.e. no visual on the runs that actually diverged.

### Rigor concerns
- **This artifact undermines the read-off convention used by every table above.** Adopt one rule (a trailing-window mean) for both arms and re-generate; some of the m0.0-vs-m0.9 difference may be read-off artefact.
- **The 1000-step outer budget is ~20× more than needed on MNIST/Fashion-MNIST**, while CIFAR-10 ran short of even that. Compute is being spent where it has no effect and withheld where it does.

---

## `missing.csv`

### What it shows
- **[shown]** 91 runs dropped; 900 retained (≈ 9.2 % loss).
- **[shown]** Two distinct failure modes: **51 × "missing 'sigmas'/'clips' artifact"** (W&B artifact upload failure) and **30 × "no test-accuracy / test-loss rows in run history"** (the run never logged evaluation).
- **[shown]** Losses are **highly non-uniform**: m0.0 loses 39, m0.9 loses 31 on arch; but the "no test-accuracy" mode is 28/30 in the m0.0 arm and clusters hard — 12 in `sgd-m0.0/dsFASHION-MNIST/e10/T-sweep`, 8 in `sgd-m0.0/dsMNIST/e10/arch-sweep/mlp-64` (the entire rung).
- **[inferred]** These are infrastructure failures (the SLURM job-chain / artifact-upload path), not scientific outcomes — assumes the failures are independent of the schedule being learned. **This assumption is worth checking**: if a run diverged and produced NaN σ, it could fail to upload the `sigmas` artifact, in which case dropping it is survivorship bias that hides exactly the failures reported in the ladder section.

### Rigor concerns
- **Verify the failure mode is not outcome-correlated.** Spot-check 3–4 `missing 'sigmas'` runs in W&B: if their logged test accuracy is at chance, the drop is silently improving the reported numbers.
- The Fashion-MNIST ε=10 T-sweep column of the m0.0 table is effectively destroyed by these drops (n = 1 and n = 2 cells) and should be marked as such or re-run.

---

## Synthesis

**Agreements**
- The **learned σ/clip shape is real and reproducible**: inverted-U, peaking at t/T ≈ 0.25–0.5, decaying to ~0 at the end. It holds across both momentum arms, all four ε, all four T, both MNIST-family datasets, and across seeds with visually tight agreement. It scales the right way with both T and ε.
- The **10× (σ, C) rescaling between the m=0.0 and m=0.9 arms matches the momentum gain 1/(1−0.9)** — an independent consistency check that the decoupled parametrisation and the projection are behaving as designed.
- Both the T-sweep table and the delta plot agree that in the m=0.0 arm the Learned advantage is **large at small T and vanishes by T ≈ 5000**.

**Contradictions**
- **The two arms disagree about which method wins, and about Adaptive Clip by 10 accuracy points.** At m=0.0 Adaptive Clip is last in all 61 rows; at m=0.9 it is first in 19 of 32 T-sweep rows. Trust neither until the baseline configuration is audited: a correctly-implemented Andrew-et-al. adaptive clip should not lose to *constant* σ/clip by 10 pts under any momentum setting. My read is that the **m=0.0 arm's Adaptive Clip is mis-tuned**, and correspondingly that the m=0.0 arm's Learned margins are inflated against a straw baseline.
- The shape plots show a *highly stable* learned schedule; the m0.9 CIFAR ladder shows 11 outright divergences. Both are true — stability holds on MNIST/Fashion-MNIST and breaks on CIFAR-10 at ε=10 with momentum.

**Strongest supported claim**
> The gradient-learned schedule converges to a reproducible inverted-U in σ (and clip) that scales correctly with T and ε, and — **in the no-momentum arm** — delivers up to +2.9 pts over a constant schedule at short training horizons (T = 2000), with the advantage decaying monotonically to zero by T ≈ 5000.

Everything past that sentence is currently under-supported.

**Weakest link**
The **m=0.9 arm's headline conclusion** ("Adaptive Clip ≥ Dynamic-DPSGD ≥ Learned > Constant"). It rests on 32 rows where all four methods sit inside a 0.3–0.8 pt band, with 6–8 seeds, no paired test, per-row bolding across 4 methods (guaranteeing spurious "winners"), a different read-off rule from the arm it's being compared to, and a baseline that behaves completely differently one directory over. It will not survive a paired seed-level test.

Close second: the **CIFAR-10 ladder results**, which confound dataset difficulty with a ~30 % shorter outer-loop budget and a bimodal converged/diverged run mixture summarised as `mean ± std`.

**Open questions to resolve before relying on any of this**
1. **Audit the Adaptive Clip baseline at m=0.0.** Was its clip-quantile / clip-learning-rate tuned per arm, or inherited from the momentum arm? A 10-pt loss to Constant is a bug signature.
2. **Unify the read-off convention** (trailing-window mean over the last ~25 outer steps) across both arms and re-generate every table. Some m0.0-vs-m0.9 difference may be read-off artefact.
3. **Run a paired per-seed test** (Learned − Constant per seed, then a sign test or Wilcoxon across seeds) instead of bolding row maxima. The forest plots already contain the paired data; the tables discard it.
4. **Was the inner learning rate re-tuned per momentum arm?** If not, the m=0.0 arm is running at ~10× too small an effective step and its "Learned helps" result may just be "Learned recovers a mis-set learning rate" — consistent with the near-total ε-invariance at m=0.0.
5. **Check whether `missing 'sigmas' artifact` runs are diverged runs.** If so, 51 dropped runs are survivorship bias in the same direction as the CIFAR divergences.
6. **Re-run the n = 1 and n = 2 cells** (m0.0 Fashion-MNIST ε=10, T ∈ {5000, 7000}) and the lost m0.0 MNIST `mlp-64` rung.
7. **Match the outer-loop budget on CIFAR-10** (median 703–725 vs 1000 elsewhere) before drawing any conclusion about architecture or dataset transfer.
8. **Diagnose the m0.9 CIFAR ε=10 divergences** (σ → 330). Does the projection lose its footing at large ε with small σ? This is adjacent to the known `project_inverse_sigmas` overflow regime for σ ≲ 0.11 — and m0.9 σ values live at 0.3–0.9, uncomfortably close.
9. **Overlay the baseline σ/clip schedules on the shape plots** so the reader can see how far the learned shape departs from what it beats.
10. **Re-render `sgd-m0.9/sigma_shape.png`** with a robust y-limit — as shipped it is blank.
