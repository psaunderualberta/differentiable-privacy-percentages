"""Architecture control for the EyePACS drop.

Question this answers: is EyePACS's collapse to the 73.982% class prior a property
of the *dataset* (from-scratch EyePACS is not learnable at this scale) or of the
*surrogate regime* ADR 0007 fixes (the MNIST-sized conv block is the binding
constraint)? The answer decides whether ADR 0007's amendment scopes its claim to
the regime or states it about the target.

Method: re-run `check2_control.run_one` unchanged — same non-private arm (sigma=0),
same C=1, same T=5000, same lr sweep, same seed, same dataloading and eval split —
with exactly one variable moved: `env.network` is pinned to an entry of `ARCHS`
instead of the eyepacs default (`channels=(16,32)`, k=8/4, stride 2, head (32,)).

The reported arm is `deep3` (the default): three stride-2 blocks, 466,661 params vs
the surrogate's 241,909 (1.9x), sized so 256x256 downsamples in stages. Note that the
obvious choice — reusing the cifar-10/imagenet block directly — is NOT usable here and
is kept only for reproducibility; see the NOTE above `ARCHS`.

Result (2026-08-05): identical to the surrogate on both splits, 73.982% val /
75.360% test. Capacity is not the binding constraint. Written up in FINDINGS.md and
ADR 0020.

Implemented by patching `util.transfer.build_target_config` in the module namespace
`run_one` resolves it from, so the production config path is otherwise untouched.
"""

import sys

sys.path.insert(0, "/home/psaunder/Documents/Masters/differentiable-privacy-percentages/src")
sys.path.insert(
    0,
    "/home/psaunder/Documents/Masters/differentiable-privacy-percentages/results/diagnostics/2026-08-05-target-floors",
)

import dataclasses
import json
from pathlib import Path

import check2_control

# The architectures under contrast, taken verbatim from net_factory's own
# DATASET_NETWORK_DEFAULTS so this is the real surrogate, not a re-spelling of it.
from networks.cnn.config import CNNConfig
from networks.mlp.config import MLPConfig

# Results land beside this script rather than in the job scratch they were first
# written to, which does not outlive the session that produced them.
OUT = Path(__file__).resolve().parent / "arch_control"
OUT.mkdir(parents=True, exist_ok=True)
check2_control.OUT = OUT

#
# NOTE: the cifar-10/imagenet default (`channels=(32,64)`, 3x3, stride 1, head 256)
# is NOT a usable "larger arch" here. EyePACS is 256x256x3, not 32x32x3, and that
# block downsamples only via its two pools, so it hands the head a 64x64x64 feature
# map: 67,129,797 parameters (277x the surrogate), which OOMs at 98 GiB and would be
# a terrible DP model regardless, since the noise cost grows with dimension. The
# honest larger architecture adds *downsampling stages*, not just width.
ARCHS = {
    "small": CNNConfig(
        channels=(16, 32),
        kernel_sizes=(8, 4),
        paddings=(2, 0),
        strides=(2, 2),
        pool_kernel_size=2,
        mlp=MLPConfig(hidden_sizes=(32,)),
    ),
    # 466,661 params (1.9x surrogate): a third stride-2 block, so the 256x256 input
    # reaches the head through enough downsampling to have a useful receptive field.
    "deep3": CNNConfig(
        channels=(32, 64, 128),
        kernel_sizes=(8, 4, 4),
        paddings=(2, 0, 0),
        strides=(2, 2, 2),
        pool_kernel_size=2,
        mlp=MLPConfig(hidden_sizes=(256,)),
    ),
    # 1,882,981 params: same depth as the surrogate, more width and a bigger head.
    # Tests capacity rather than receptive field.
    "wider": CNNConfig(
        channels=(32, 64),
        kernel_sizes=(8, 4),
        paddings=(2, 0),
        strides=(2, 2),
        pool_kernel_size=2,
        mlp=MLPConfig(hidden_sizes=(128,)),
    ),
    # The cifar-10 block, kept only so the 67M-param blowup above is reproducible.
    "cifar": CNNConfig(
        channels=(32, 64),
        kernel_sizes=(3, 3),
        paddings=(1, 1),
        strides=(1, 1),
        pool_kernel_size=2,
        mlp=MLPConfig(hidden_sizes=(256,)),
    ),
}


def patch_arch(arch_name: str) -> None:
    """Force every `build_target_config` call in this process onto `arch_name`."""
    import util.transfer as T

    original = T.build_target_config

    def patched(target, batch_size):
        config = original(target, batch_size)
        env = dataclasses.replace(config.sweep.env, network=ARCHS[arch_name])
        sweep = dataclasses.replace(config.sweep, env=env)
        return dataclasses.replace(config, sweep=sweep)

    T.build_target_config = patched


def param_count(arch_name: str, target: str = "eyepacs") -> int:
    """Trainable parameter count of `arch_name` on `target`'s real input shape."""
    import equinox as eqx
    import jax

    from conf.scope import RunContext, using
    from conf.singleton_conf import SingletonConfig
    from networks.net_factory import net_factory
    from util.dataloaders import get_dataset_shapes
    from util.transfer import TargetSpec, build_target_config

    spec = TargetSpec(name=target, eps=10.0, delta=1e-7, T=5000, arch="")
    config = build_target_config(spec, 250)
    with SingletonConfig.override(config), using(RunContext(config)):
        X_shape, y_shape, _, _ = get_dataset_shapes()
    net = net_factory(ARCHS[arch_name], input_shape=X_shape, output_shape=y_shape)
    arrays = eqx.filter(net, eqx.is_inexact_array)
    return int(sum(x.size for x in jax.tree.leaves(arrays)))


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="deep3", choices=sorted(ARCHS))
    ap.add_argument("--target", default="eyepacs")
    ap.add_argument("--T", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=250)
    ap.add_argument("--lrs", type=float, nargs="+", default=[0.3, 0.1, 0.03])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--microbatch", type=int, default=-1)
    ap.add_argument("--scan-segments", type=int, default=-1)
    args = ap.parse_args()

    floor = {"eyepacs": 73.982, "imagenet": 1.125, "chexpert": 60.072}.get(args.target)

    n_small = param_count("small", args.target)
    n_large = param_count(args.arch, args.target)
    print(
        f"param counts on {args.target}: small={n_small:,}  large={n_large:,} "
        f"({n_large / n_small:.1f}x)",
        flush=True,
    )

    patch_arch(args.arch)

    rows = []
    for lr in args.lrs:
        print(
            f"\n=== {args.target} arch={args.arch} NON-PRIVATE lr={lr} "
            f"C={args.clip} T={args.T} ===",
            flush=True,
        )
        r = check2_control.run_one(
            args.target,
            lr,
            args.T,
            args.batch_size,
            args.seed,
            arm="nonprivate",
            clip=args.clip,
            scan_segments=args.scan_segments,
            microbatch=args.microbatch,
        )
        r["arch"] = args.arch
        r["n_params"] = n_large
        rows.append(r)
        print(
            f"    val={r['val_accuracy']:.3f}%  test={r['test_accuracy']:.3f}%  "
            f"loss {r['train_loss_step0']:.4g} -> {r['train_loss_last100']:.4g}  "
            f"({r['seconds']:.0f}s)",
            flush=True,
        )
        (OUT / f"arch_{args.arch}_{args.target}_T{args.T}.json").write_text(
            json.dumps(rows, indent=2)
        )

    print(
        f"\n{'=' * 92}\nARCH CONTROL — {args.target}, arch={args.arch}, "
        f"T={args.T}, C={args.clip}, non-private"
        + (f"   (majority floor = {floor:.3f}%)" if floor else "")
        + f"\n{'=' * 92}"
    )
    print(
        f"{'lr':>8} {'val_acc':>9} {'test_acc':>9} {'trloss_0':>10} "
        f"{'trloss_min':>11} {'trloss_last':>12} {'sec':>7}"
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
            f"{r['lr']:>8g} {r['val_accuracy']:>8.3f}% {r['test_accuracy']:>8.3f}% "
            f"{r['train_loss_step0']:>10.4g} {r['train_loss_min']:>11.4g} "
            f"{r['train_loss_last100']:>12.4g} {r['seconds']:>7.0f}{flag}"
        )
    print("\nclass-prior entropy reference for eyepacs: 0.873")


if __name__ == "__main__":
    main()
