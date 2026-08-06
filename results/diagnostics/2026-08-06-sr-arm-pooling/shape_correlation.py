import numpy as np
import pandas as pd

d = pd.read_parquet("cache/results/psaunder__FirSweep/schedules.parquet")
d = d[d.arch_label == "cnn-16x32-head32"]
print(
    "T-sweep rows:", len(d), "| conditions:", d[["dataset", "eps", "T"]].drop_duplicates().shape[0]
)

GRID = np.linspace(0.0, 1.0, 101)


def shape_profiles(target):
    """Per (condition, arm): seed-averaged curve normalised by its own mean -> shape."""
    out = {}
    for (ds, eps, T, arm), g in d.groupby(["dataset", "eps", "T", "optimizer"], sort=True):
        curves = []
        for _, r in g.groupby("run_id"):
            r = r.sort_values("step_norm")
            y = np.interp(GRID, r.step_norm.values, r[target].values)
            m = np.nanmean(y)
            if not np.isfinite(m) or m <= 0:
                continue
            curves.append(y / m)  # scale divided out, per run
        if curves:
            out[(ds, eps, T, arm)] = np.nanmean(np.stack(curves), axis=0)
    return out


for target in ("sigma", "clip"):
    prof = shape_profiles(target)
    conds = sorted({k[:3] for k in prof})
    corrs, peak0, peak9, l2 = [], [], [], []
    for c in conds:
        a = prof.get((*c, "sgd-m0.0"))
        b = prof.get((*c, "sgd-m0.9"))
        if a is None or b is None:
            continue
        corrs.append(np.corrcoef(a, b)[0, 1])
        peak0.append(GRID[np.argmax(a)])
        peak9.append(GRID[np.argmax(b)])
        # relative L2 between the two normalised shapes
        l2.append(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(((a + b) / 2) ** 2)))
    corrs, peak0, peak9, l2 = map(np.array, (corrs, peak0, peak9, l2))
    print(f"\n=== {target}: {len(corrs)} conditions present in BOTH arms ===")
    print(
        f"  shape corr   min {corrs.min():.3f}  median {np.median(corrs):.3f}  max {corrs.max():.3f}"
    )
    print(f"  rel L2 gap   min {l2.min():.3f}  median {np.median(l2):.3f}  max {l2.max():.3f}")
    print(
        f"  peak t/T  m0.0 median {np.median(peak0):.2f} (range {peak0.min():.2f}-{peak0.max():.2f})"
    )
    print(
        f"  peak t/T  m0.9 median {np.median(peak9):.2f} (range {peak9.min():.2f}-{peak9.max():.2f})"
    )

    # within-arm baseline: how much do shapes vary BETWEEN conditions of the SAME arm?
    for arm in ("sgd-m0.0", "sgd-m0.9"):
        ps = [prof[(*c, arm)] for c in conds if (*c, arm) in prof]
        wc = [
            np.corrcoef(ps[i], ps[j])[0, 1] for i in range(len(ps)) for j in range(i + 1, len(ps))
        ]
        print(
            f"  within-{arm} cross-condition corr: median {np.median(wc):.3f} min {np.min(wc):.3f}"
        )
