"""Check 2 — non-private EyePACS control.

The handoff asks for "one training run at a very large eps (sigma -> ~0)". That is
not reachable: ``approx_to_gdp`` hard-fails above eps~=88 (jnp.exp(eps) overflows
float32) and even at the ceiling mu only reaches ~9.05, which for eyepacs
(p=250/28100, T=5000) leaves a constant sigma of ~0.43 vs ~0.68 at eps=10 — a 1.6x
noise reduction, nowhere near sigma->0. A run there could not distinguish "task is
hard" from "DP still binding".

So the control is run the other way: the *real* ``train_with_noise`` path with the
DP mechanism neutralised directly — sigma = 0 (get_spherical_noise scales by
sigma/L, so this is exactly zero noise) and clip = 1e6 (no per-sample gradient is
anywhere near this, so no clipping occurs). Everything else — dataloading,
truncated-Poisson buffers, network, optimizer, eval split — is the production path.
That is a genuinely non-private SGD upper bound on the surrogate architecture.

Learning rate is swept, because a single-LR null result would not be evidence about
the task.
"""

import sys
import time

sys.path.insert(0, "/home/psaunder/Documents/Masters/differentiable-privacy-percentages/src")

import dataclasses
import json
from pathlib import Path

import jax.random as jr
import numpy as np

OUT = Path("/home/psaunder/.claude/jobs/ae8bba61/tmp/check2")
OUT.mkdir(parents=True, exist_ok=True)

NONPRIVATE_SIGMA = 0.0
DEFAULT_CLIP = 1.0
"""C is NOT an Abadi ceiling in this codebase. ``sum_clipped_per_example_grads``
applies the DP-PSAC smooth multiplier ``C / (||g|| + 1/(||g||+1))``, which is
unbounded above — a large C *amplifies* the gradient (C=1e6 scales it ~1e5x and
diverges instantly) rather than disabling clipping. C is a normalisation scale, so
the non-private control holds it at a normal value and only removes the noise.

That also makes (C, lr) degenerate: the update is C x (mean unit-ish direction),
and in the DP arm the noise std is sigma/L with sigma proportional to C, so the SNR
is C-independent too. Only the product C*lr matters, hence C is fixed and lr swept."""


def run_one(
    target_name,
    lr,
    T,
    batch_size,
    seed,
    arm="nonprivate",
    clip=DEFAULT_CLIP,
    eps=10.0,
    scan_segments=-1,
    momentum=0.9,
    microbatch=-1,
):
    """One inner training run; returns eval metrics.

    ``arm="nonprivate"`` sets sigma=0 (zero noise) at the given C. ``arm="dp"``
    seats the same C on the real eps budget via ``schedule.project()`` — exactly
    what the Constant reference does — so the two arms differ only in the noise.
    """
    from conf.config_util import DistributionConfig
    from conf.scope import RunContext, using
    from conf.singleton_conf import SingletonConfig
    from environments.dp import train_with_noise
    from environments.dp_params import DPTrainingParams
    from policy.base_schedules.constant import ConstantSchedule
    from policy.schedules.sigma_and_clip import SigmaAndClipSchedule
    from privacy.gdp_privacy import get_privacy_params
    from util.dataloaders import get_dataset_shapes
    from util.transfer import TargetSpec, build_target_config

    def const(v):
        return DistributionConfig(min=v, max=v, value=v, distribution="constant")

    spec = TargetSpec(name=target_name, eps=eps, delta=1e-7, T=T, arch="")
    config = build_target_config(spec, batch_size)
    # Same optimizer family the references use (SGD + momentum); only the LR moves.
    opt = dataclasses.replace(
        config.sweep.env.optimizer, learning_rate=const(lr), momentum=const(momentum)
    )
    env = dataclasses.replace(
        config.sweep.env, optimizer=opt, scan_segments=scan_segments, microbatch_size=microbatch
    )
    sweep = dataclasses.replace(config.sweep, env=env)
    config = dataclasses.replace(config, sweep=sweep)

    with SingletonConfig.override(config), using(RunContext(config)):
        X_shape, *_ = get_dataset_shapes()
        pp = get_privacy_params(X_shape[0])
        env_params = DPTrainingParams.create_direct_from_config()

        if arm == "nonprivate":
            # sigma = 0 is exactly "no Gaussian mechanism": get_spherical_noise
            # returns sigma * normal / L, so the update is the pure PSAC-normalised
            # mean gradient. This is eps -> infinity without touching the accountant.
            schedule = SigmaAndClipSchedule(
                ConstantSchedule(NONPRIVATE_SIGMA, T), ConstantSchedule(clip, T), pp
            )
        elif arm == "dp":
            # project() only enforces the *inequality* sum_i exp((C/s_i)^2) <= (mu/p)^2 + T,
            # so a feasible-but-slack sigma passes through untouched and silently
            # under-spends the budget. For a constant schedule the binding sigma is
            # closed-form: T*exp((C/s)^2) = (mu/p)^2 + T.
            bound = float((pp.mu / pp.p) ** 2) + T
            sigma_star = clip / float(np.sqrt(np.log(bound / T)))
            schedule = SigmaAndClipSchedule(
                ConstantSchedule(sigma_star, T), ConstantSchedule(clip, T), pp
            )
            spent = T * np.exp((clip / sigma_star) ** 2)
            assert np.isclose(spent, bound, rtol=1e-4), f"budget not bound: {spent} vs {bound}"
        else:
            raise ValueError(f"unknown arm {arm!r}")

        got_sigma = np.asarray(schedule.get_private_noise_scales())
        got_clip = np.asarray(schedule.get_private_clips())
        assert got_sigma.shape == (T,) and got_clip.shape == (T,), (got_sigma.shape, got_clip.shape)
        if arm == "nonprivate":
            assert np.allclose(got_sigma, 0.0), f"noise not removed: {got_sigma[:3]}"
            assert np.allclose(got_clip, clip), "clip not seated"
        sigma, clip = float(got_sigma[0]), float(got_clip[0])
        print(
            f"    seated: sigma={sigma:.6g}  clip={clip:.6g}  mu_step=C/sigma="
            f"{(clip / sigma) if sigma else float('inf'):.6g}",
            flush=True,
        )

        key = jr.PRNGKey(seed)
        mb_key, init_key, noise_key = jr.split(key, 3)
        t0 = time.time()
        _, stats = train_with_noise(schedule, env_params, mb_key, init_key, noise_key)
        elapsed = time.time() - t0

        losses = np.asarray(stats.losses)
        accs = np.asarray(stats.accuracies)
        np.savez(
            OUT / f"curves_{target_name}_{arm}_lr{lr:g}_T{T}_s{seed}.npz",
            losses=losses,
            accs=accs,
        )
        return {
            "target": target_name,
            "arm": arm,
            "eps": eps if arm == "dp" else float("inf"),
            "lr": lr,
            "momentum": momentum,
            "T": T,
            "seed": seed,
            "sigma": float(sigma),
            "clip": float(clip),
            "val_loss": float(stats.val_loss),
            "val_accuracy": float(stats.val_accuracy),
            "test_loss": float(stats.test_loss),
            "test_accuracy": float(stats.test_accuracy),
            "train_loss_step0": float(losses[0]),
            "train_loss_first10": [float(v) for v in losses[:10]],
            "train_loss_min": float(np.nanmin(losses)),
            "train_loss_first100": float(np.nanmean(losses[:100])),
            "train_loss_last100": float(np.nanmean(losses[-100:])),
            "train_acc_first100": float(np.nanmean(accs[:100])),
            "train_acc_last100": float(np.nanmean(accs[-100:])),
            "n_nan_losses": int(np.isnan(losses).sum()),
            "seconds": elapsed,
        }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="eyepacs")
    ap.add_argument("--T", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=250)
    ap.add_argument("--lrs", type=float, nargs="+", default=[0.1])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scan-segments", type=int, default=-1)
    ap.add_argument(
        "--microbatch",
        type=int,
        default=-1,
        help="cap per-sample-gradient memory; must divide both batch_size and buffer B",
    )
    ap.add_argument("--tag", default="control")
    ap.add_argument("--arms", nargs="+", default=["nonprivate"], choices=["nonprivate", "dp"])
    ap.add_argument("--clip", type=float, default=DEFAULT_CLIP)
    ap.add_argument("--eps", type=float, default=10.0)
    args = ap.parse_args()

    floors = {"eyepacs": 73.982, "imagenet": 1.125, "chexpert": 60.072}
    floor = floors.get(args.target)

    rows = []
    for arm in args.arms:
        for lr in args.lrs:
            print(f"\n=== {args.target} arm={arm} lr={lr} C={args.clip} T={args.T} ===", flush=True)
            r = run_one(
                args.target,
                lr,
                args.T,
                args.batch_size,
                args.seed,
                arm=arm,
                clip=args.clip,
                eps=args.eps,
                scan_segments=args.scan_segments,
                microbatch=args.microbatch,
            )
            rows.append(r)
            print(
                f"    val={r['val_accuracy']:.3f}%  test={r['test_accuracy']:.3f}%  "
                f"loss {r['train_loss_step0']:.4g} -> {r['train_loss_last100']:.4g}  "
                f"({r['seconds']:.0f}s)",
                flush=True,
            )
            path = OUT / f"{args.tag}_{args.target}_T{args.T}.json"
            path.write_text(json.dumps(rows, indent=2))

    print(
        f"\n{'=' * 88}\nCONTROL — {args.target}, T={args.T}, C={args.clip}"
        + (f"   (majority floor = {floor:.3f}%)" if floor else "")
        + f"\n{'=' * 88}"
    )
    print(
        f"{'arm':>11} {'lr':>8} {'sigma':>8} {'val_acc':>9} {'test_acc':>9} "
        f"{'trloss_0':>10} {'trloss_min':>11} {'trloss_last':>12} {'sec':>7}"
    )
    for r in rows:
        flag = ""
        if floor is not None:
            flag = (
                "  AT/BELOW FLOOR"
                if r["val_accuracy"] <= floor + 0.05
                else f"  +{r['val_accuracy'] - floor:.2f}pp"
            )
        print(
            f"{r['arm']:>11} {r['lr']:>8g} {r['sigma']:>8.4g} {r['val_accuracy']:>8.3f}% "
            f"{r['test_accuracy']:>8.3f}% {r['train_loss_step0']:>10.4g} "
            f"{r['train_loss_min']:>11.4g} {r['train_loss_last100']:>12.4g} "
            f"{r['seconds']:>7.0f}{flag}"
        )


if __name__ == "__main__":
    main()
