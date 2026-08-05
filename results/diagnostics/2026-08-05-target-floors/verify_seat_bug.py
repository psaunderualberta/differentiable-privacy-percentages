"""Verify the two compounding bugs in util/transfer.py:seat_on_budget."""

import sys

sys.path.insert(0, "/home/psaunder/Documents/Masters/differentiable-privacy-percentages/src")

import numpy as np
import pandas as pd
from scipy.optimize import brentq

PARQ = "/home/psaunder/Documents/Masters/differentiable-privacy-percentages/src/cache/results/psaunder__FirSweep/schedules.parquet"

df = pd.read_parquet(PARQ)
d = df[df["run_id"] == "4rh8p1j8"].sort_values("inner_step")
sig, clip = d["sigma"].values.astype(float), d["clip"].values.astype(float)
T = len(sig)

# imagenet target numbers (from the reproduction run)
mu, p = 1.7254, 0.001953
bound = (mu / p) ** 2 + T

print("BUG 1 — residual uses exp(1/sigma^2), ignoring the clip entirely.")
print("  The GDP constraint is  sum_i exp((C_i/sigma_i)^2) <= (mu/p)^2 + T.")
print("  seat_on_budget never receives `clips`, so it substitutes C_i := 1.\n")
print(
    f"  source sigma min={sig.min():.4f}  ->  1/sigma^2 max={1 / sig.min() ** 2:.1f}"
    f"  ->  exp(...) = {np.exp(1 / sig.min() ** 2):.3g}"
)
print(
    f"  correct per-step weight (C/sigma)^2 max={np.max((clip / sig) ** 2):.3f}"
    f"  ->  exp(...) = {np.exp(np.max((clip / sig) ** 2)):.3g}\n"
)


def buggy_residual(c):
    with np.errstate(over="ignore"):
        return np.sum(np.exp(1.0 / (c * sig) ** 2)) - bound


def correct_residual(c):
    with np.errstate(over="ignore"):
        return np.sum(np.exp((clip / (c * sig)) ** 2)) - bound


print("BUG 2 — the bisection is bracketed on c in [1e-6, 10] with throw=False,")
print("  so a root outside the bracket is silently returned as the ceiling.\n")
print(f"  {'c':>8} {'buggy residual':>20} {'correct residual':>20}")
for c in [0.5, 0.84, 1.0, 2.0, 5.0, 10.0]:
    print(f"  {c:>8g} {buggy_residual(c):>20.6g} {correct_residual(c):>20.6g}")

print("\n  buggy residual is +inf across the whole bracket (exp overflow on the")
print("  smallest sigmas), so bisection never brackets a root and saturates at c=10.")

c_star = brentq(correct_residual, 1e-3, 10.0)
print(f"\nCORRECT scale factor for imagenet: c = {c_star:.6f}  (buggy code used 10.0)")
scaled = c_star * sig
spent = float(np.sum(np.exp((clip / scaled) ** 2)))
print(f"  correctly-seated sigma: mean={scaled.mean():.4f} (buggy: {10.0 * sig.mean():.4f})")
print(f"  budget spent={spent:.6g}  bound={bound:.6g}  ratio={spent / bound:.6f}")
print(
    f"  per-step mu=C/sigma mean={np.mean(clip / scaled):.4f} "
    f"(buggy: {np.mean(clip / (10.0 * sig)):.4f})"
)
print(f"\n  => the buggy path over-noises by {10.0 / c_star:.2f}x")
