"""Reference-separation analysis for the surviving transfer targets (ADR 0020, part 2).

Reads every ``producer="reference"`` cell in the transfer cache and, per target regime,
reports the between-reference spread against the within-reference (per-seed) spread.
That ratio is the *separation* half of the schedule-resolving-power criterion: whether
differently-shaped native schedules give measurably different answers on a target.

The ``accuracy`` column is already in percent; do not rescale it.

Usage (from ``src/``)::

    uv run python ../results/diagnostics/2026-08-05-target-floors/reference_separation.py

Takes the reference-cell directory as an optional argument, since a git worktree has no
cache of its own and must be pointed at the main checkout's.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REF = Path(__file__).resolve().parents[3] / "src" / "cache" / "transfer" / "reference"


def main() -> None:
    ref = Path(sys.argv[1]) if len(sys.argv) > 1 else REF
    cells = sorted(ref.glob("*.parquet"))
    if not cells:
        raise SystemExit(f"no reference cells under {ref}")
    df = pd.concat([pd.read_parquet(p) for p in cells], ignore_index=True)
    print(f"{len(cells)} cells, {len(df)} rows\n")

    for (target, eps, t_steps), g in df.groupby(["target", "target_eps", "target_T"]):
        print(f"===== {target}  eps={eps}  T={t_steps} =====")
        tab = (
            g.groupby("source_id")["accuracy"]
            .agg(["count", "mean", "std", "min", "max"])
            .sort_values("mean")
        )
        print(tab.round(3).to_string())

        by_ref = [sub["accuracy"].values for _, sub in g.groupby("source_id")]
        dof = sum(len(x) - 1 for x in by_ref)
        pooled = np.sqrt(sum((len(x) - 1) * np.var(x, ddof=1) for x in by_ref) / dof)
        gap = tab["mean"].max() - tab["mean"].min()
        print(f"\n  pooled sigma_eval = {pooled:.3f} pp  (dof={dof})")
        print(f"  max-min gap       = {gap:.3f} pp")
        print(f"  gap / sigma_eval  = {gap / pooled:.2f}")

        print("\n  pairwise (Welch t, Cohen's d):")
        for a, b in itertools.combinations(list(tab.index), 2):
            xa = g.loc[g.source_id == a, "accuracy"].values
            xb = g.loc[g.source_id == b, "accuracy"].values
            _, p = stats.ttest_ind(xa, xb, equal_var=False)
            sp = np.sqrt((np.var(xa, ddof=1) + np.var(xb, ddof=1)) / 2)
            star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
            print(
                f"    {a:>14} vs {b:<14} d={(xb.mean() - xa.mean()) / sp:+6.2f}  p={p:9.2e}  {star}"
            )

        f_stat, p_anova = stats.f_oneway(*by_ref)
        print(f"\n  one-way ANOVA: F={f_stat:.2f}  p={p_anova:.3e}\n")

    print("=== val loss (does the loss ordering agree with accuracy?) ===")
    print(df.groupby(["target", "source_id"])["loss"].agg(["mean", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
