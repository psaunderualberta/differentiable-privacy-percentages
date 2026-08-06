# Do the momentum arms share a schedule shape? (2026-08-06)

Backs **ADR 0023**. Answers whether a template synthesis could pool `sgd-m0.0` and
`sgd-m0.9` into one fit, and if so what it would cost.

Run from `src/` (the scripts read `cache/results/psaunder__FirSweep/schedules.parquet`).
That cache is not in git, so run these from the main checkout, not from a worktree:

```bash
cd src
uv run python ../results/diagnostics/2026-08-06-sr-arm-pooling/shape_correlation.py
uv run python ../results/diagnostics/2026-08-06-sr-arm-pooling/shape_dof.py
uv run python ../results/diagnostics/2026-08-06-sr-arm-pooling/mu_and_coverage.py
```

All three restrict to the T-sweep arch `cnn-16x32-head32` (32 conditions, both arms
present in all 32), interpolate each run onto a 101-point `step_norm` grid, divide by its
own mean to remove scale, and average over seeds.

- `shape_correlation.py` — cross-arm shape correlation and peak position, with the
  within-arm cross-condition correlation as the baseline that makes it interpretable.
- `shape_dof.py` — the measurement that decides it. PCA reconstruction error with `k`
  shape components + free per-condition coefficients, per-arm against pooled. Also the
  ceiling for "one shared shape + one global per-arm scale".
- `mu_and_coverage.py` — run counts per (dataset, arm), plus the μ = clip/σ shape and the
  σ/μ scale ratios that re-check ADR 0016's own figures.

Headline: pooling costs about **one extra shape degree of freedom**, not a different
shape. A single global per-arm scale is *not* sufficient (14.5 % σ / 15.8 % clip residual)
because the arms' peaks sit at different `t/T`. Scoping is kept on cost — see ADR 0023.

The PCA is a generous stand-in for a template fit (free linear coefficients are easier
than K constants entering a discovered `f` nonlinearly). It is generous to the scoped and
pooled cases equally, so the comparison between columns carries the weight, not the
absolute error levels.
