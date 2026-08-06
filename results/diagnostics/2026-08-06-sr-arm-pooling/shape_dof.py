import numpy as np
import pandas as pd

d = pd.read_parquet("cache/results/psaunder__FirSweep/schedules.parquet")
d = d[d.arch_label == "cnn-16x32-head32"]
GRID = np.linspace(0.0, 1.0, 101)


def profiles(target):
    out = {}
    for (ds, eps, T, arm), g in d.groupby(["dataset", "eps", "T", "optimizer"], sort=True):
        cs = []
        for _, r in g.groupby("run_id"):
            r = r.sort_values("step_norm")
            y = np.interp(GRID, r.step_norm.values, r[target].values)
            m = np.nanmean(y)
            if np.isfinite(m) and m > 0:
                cs.append(y / m)
        if cs:
            out[(ds, eps, T, arm)] = np.nanmean(np.stack(cs), axis=0)
    return out


def recon_err(M, k):
    """Rel. RMS error reconstructing rows of M from k components + free per-row coeffs."""
    mu = M.mean(axis=0)
    X = M - mu
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    approx = mu + U[:, :k] @ np.diag(S[:k]) @ Vt[:k]
    return np.sqrt(np.mean((M - approx) ** 2)) / np.sqrt(np.mean(M**2))


for target in ("sigma", "clip"):
    P = profiles(target)
    A0 = np.stack([v for k, v in sorted(P.items()) if k[3] == "sgd-m0.0"])
    A9 = np.stack([v for k, v in sorted(P.items()) if k[3] == "sgd-m0.9"])
    POOL = np.vstack([A0, A9])
    print(
        f"\n=== {target}  (m0.0: {A0.shape[0]} conds, m0.9: {A9.shape[0]}, pooled: {POOL.shape[0]}) ==="
    )
    print(
        "  k |   m0.0    m0.9   pooled   <- rel RMS recon error, k shape DOF + free per-condition coeffs"
    )
    for k in range(0, 5):
        print(f"  {k} | {recon_err(A0, k):7.4f} {recon_err(A9, k):7.4f} {recon_err(POOL, k):7.4f}")

    # Option 3 ceiling: ONE global per-arm scale on a single shared shape.
    # Best case = each arm's mean shape, scaled optimally; residual is irreducible.
    m0, m9 = A0.mean(axis=0), A9.mean(axis=0)
    s = (m0 @ m9) / (m9 @ m9)  # optimal single scale mapping m9 -> m0
    r = np.sqrt(np.mean((m0 - s * m9) ** 2)) / np.sqrt(np.mean(m0**2))
    print(
        f"  option-3 ceiling: one shared shape + one global per-arm scale leaves {r * 100:.1f}% rel RMS"
    )
