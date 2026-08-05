"""A/B the buggy vs corrected seat_on_budget on the real curve-transfer path.

Buggy path should reproduce the probe cell (~1% on imagenet, ~74% on eyepacs).
Corrected path scales the source sigma by the c that actually solves
    sum_i exp((C_i/(c*sigma_i))^2) = (mu/p)^2 + T
i.e. the constraint the GDP accountant actually enforces, using the clips.
"""

import sys

sys.path.insert(0, "/home/psaunder/Documents/Masters/differentiable-privacy-percentages/src")

import json
import time
from pathlib import Path

import jax.random as jr
import numpy as np
import pandas as pd
from scipy.optimize import brentq

PARQ = "/home/psaunder/Documents/Masters/differentiable-privacy-percentages/src/cache/results/psaunder__FirSweep/schedules.parquet"
OUT = Path("/home/psaunder/.claude/jobs/ae8bba61/tmp/check2")
OUT.mkdir(parents=True, exist_ok=True)
FLOOR = {"eyepacs": 73.982, "imagenet": 1.125, "chexpert": 60.072}


def corrected_seat(sigmas, clips, mu, p, T):
    """The scale factor that binds the *actual* GDP boundary, clips included."""
    bound = (mu / p) ** 2 + T

    def residual(c):
        with np.errstate(over="ignore"):
            return np.sum(np.exp((clips / (c * sigmas)) ** 2)) - bound

    c = brentq(residual, 1e-3, 1e3)
    return c * sigmas, c


def main(targets, seeds=(0, 1, 2)):
    from conf.scope import RunContext, using
    from conf.singleton_conf import SingletonConfig
    from environments.dp import train_with_noise
    from environments.dp_params import DPTrainingParams
    from privacy.gdp_privacy import get_privacy_params
    from transfer_curve import build_curve_schedule, resample_curve
    from util.dataloaders import get_dataset_shapes
    from util.transfer import RawArraySchedule, TargetSpec, build_target_config

    df = pd.read_parquet(PARQ)
    d = df[df["run_id"] == "4rh8p1j8"].sort_values("inner_step")
    ssig, sclip = d["sigma"].values.astype(float), d["clip"].values.astype(float)

    rows = []
    for t in targets:
        spec = TargetSpec(name=t, eps=10.0, delta=1e-7, T=5000, arch="")
        config = build_target_config(spec, 250)
        with SingletonConfig.override(config), using(RunContext(config)):
            X_shape, *_ = get_dataset_shapes()
            pp = get_privacy_params(X_shape[0])
            env_params = DPTrainingParams.create_direct_from_config()
            T = 5000
            clips_t = resample_curve(sclip, T)
            sig_t = resample_curve(ssig, T)

            variants = {}
            buggy = build_curve_schedule(ssig, sclip, pp)
            variants["buggy"] = buggy
            fixed_sig, c = corrected_seat(sig_t, clips_t, float(pp.mu), float(pp.p), T)
            variants["fixed"] = RawArraySchedule(fixed_sig, clips_t)
            print(f"\n### {t}: corrected scale c={c:.4f} (buggy path used ~10.0)", flush=True)

            for name, sched in variants.items():
                sg = np.asarray(sched.get_private_noise_scales())
                cl = np.asarray(sched.get_private_clips())
                spent = float(np.sum(np.exp((cl / sg) ** 2)))
                bound = float((pp.mu / pp.p) ** 2) + T
                print(
                    f"  [{name}] sigma mean={sg.mean():.4f}  budget used="
                    f"{100 * spent / bound:.4f}%",
                    flush=True,
                )
                for seed in seeds:
                    k = jr.PRNGKey(seed)
                    mb, ik, nk = jr.split(k, 3)
                    t0 = time.time()
                    _, st = train_with_noise(sched, env_params, mb, ik, nk)
                    losses = np.asarray(st.losses)
                    r = {
                        "target": t,
                        "variant": name,
                        "seed": seed,
                        "sigma_mean": float(sg.mean()),
                        "budget_used_pct": 100 * spent / bound,
                        "val_accuracy": float(st.val_accuracy),
                        "test_accuracy": float(st.test_accuracy),
                        "val_loss": float(st.val_loss),
                        "train_loss_step0": float(losses[0]),
                        "train_loss_last100": float(np.nanmean(losses[-100:])),
                        "seconds": time.time() - t0,
                    }
                    rows.append(r)
                    print(
                        f"    seed={seed} val={r['val_accuracy']:.3f}% "
                        f"test={r['test_accuracy']:.3f}% loss {r['train_loss_step0']:.4g}"
                        f"->{r['train_loss_last100']:.4g} ({r['seconds']:.0f}s)",
                        flush=True,
                    )
                    (OUT / "curve_ab.json").write_text(json.dumps(rows, indent=2))

    print(f"\n{'=' * 80}\nCURVE A/B — source 4rh8p1j8 (fashion-mnist eps=10 T=5000)\n{'=' * 80}")
    print(
        f"{'target':>10} {'variant':>8} {'budget%':>9} {'val mean':>10} {'probe cell':>12} {'floor':>8}"
    )
    agg = {}
    for r in rows:
        agg.setdefault((r["target"], r["variant"]), []).append(r["val_accuracy"])
    probe = {"imagenet": "0.925/0.975", "eyepacs": "73.96/73.98", "chexpert": "64.7/66.2"}
    for (t, v), vals in agg.items():
        b = next(r["budget_used_pct"] for r in rows if r["target"] == t and r["variant"] == v)
        print(
            f"{t:>10} {v:>8} {b:>8.3f}% {np.mean(vals):>9.3f}% "
            f"{probe.get(t, ''):>12} {FLOOR.get(t, 0):>7.3f}%"
        )


if __name__ == "__main__":
    main(sys.argv[1:] or ["imagenet"])
