import numpy as np
import pandas as pd

d = pd.read_parquet("cache/results/psaunder__FirSweep/schedules.parquet")
d = d[d.arch_label == "cnn-16x32-head32"].copy()

print("=== run counts per (dataset, arm) on the T-sweep ===")
runs = d.groupby(["dataset", "optimizer"])["run_id"].nunique().unstack("optimizer")
print(runs.to_string())
print("\ntotal runs:", d.groupby("optimizer")["run_id"].nunique().to_dict())

d["mu"] = d["clip"] / d["sigma"]
GRID = np.linspace(0.0, 1.0, 101)
P = {}
for (ds, eps, T, arm), g in d.groupby(["dataset", "eps", "T", "optimizer"], sort=True):
    cs = []
    for _, r in g.groupby("run_id"):
        r = r.sort_values("step_norm")
        y = np.interp(GRID, r.step_norm.values, r["mu"].values)
        m = np.nanmean(y)
        if np.isfinite(m) and m > 0:
            cs.append(y / m)
    if cs:
        P[(ds, eps, T, arm)] = np.nanmean(np.stack(cs), axis=0)

conds = sorted({k[:3] for k in P})
corr, p0, p9 = [], [], []
for c in conds:
    a, b = P.get((*c, "sgd-m0.0")), P.get((*c, "sgd-m0.9"))
    if a is None or b is None:
        continue
    corr.append(np.corrcoef(a, b)[0, 1])
    p0.append(GRID[np.argmax(a)])
    p9.append(GRID[np.argmax(b)])
corr, p0, p9 = map(np.array, (corr, p0, p9))
print(f"\n=== mu = clip/sigma, {len(corr)} shared conditions (ADR 0016's measurement) ===")
print(f"  shape corr  min {corr.min():.3f} median {np.median(corr):.3f} max {corr.max():.3f}")
print(f"  peak t/T  m0.0 median {np.median(p0):.2f} | m0.9 median {np.median(p9):.2f}")

# mu SCALE ratio across arms (ADR 0016 claims ~1.01)
med = d.groupby(["dataset", "eps", "T", "optimizer"])["mu"].median().unstack("optimizer").dropna()
print(f"  mu scale ratio m0.9/m0.0: median {(med['sgd-m0.9'] / med['sgd-m0.0']).median():.3f}")
sig = (
    d.groupby(["dataset", "eps", "T", "optimizer"])["sigma"].median().unstack("optimizer").dropna()
)
print(f"  sigma scale ratio m0.9/m0.0: median {(sig['sgd-m0.9'] / sig['sgd-m0.0']).median():.3f}")
