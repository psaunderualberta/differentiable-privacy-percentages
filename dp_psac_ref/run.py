"""CLI entrypoint for the standalone DP-PSAC runner.

Usage:
    uv run run.py --sigmas sigmas.npy --clips clips.npy \\
                  --dataset mnist --batch-size 512 --lr 1.0 --r 0.1 \\
                  --delta 1e-5 --arch cnn --seed 0 --out results.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import accountant
import dp_psac
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import tyro

import data


@dataclass
class Args:
    sigmas: Path
    clips: Path
    dataset: str = "mnist"
    arch: str = "mlp"  # "mlp" or "cnn"
    batch_size: int = 512
    lr: float = 1.0
    r: float = 0.1
    delta: float = 1e-5
    seed: int = 0
    out: Path | None = None
    log_every: int = 50


def main(args: Args) -> None:
    sigmas = np.load(args.sigmas)
    clips = np.load(args.clips)
    assert sigmas.shape == clips.shape and sigmas.ndim == 1, (
        f"sigmas and clips must be 1D arrays of equal length; got {sigmas.shape} and {clips.shape}"
    )
    T = int(sigmas.shape[0])
    print(f"loaded schedules: T={T}  sigma[0]={sigmas[0]:.4f} clip[0]={clips[0]:.4f}")

    x_train, y_train, x_test, y_test = data.load(args.dataset, args.arch)
    n = int(x_train.shape[0])
    q = args.batch_size / n
    print(f"dataset={args.dataset} arch={args.arch} N={n} B={args.batch_size} q={q:.6f}")

    key = jr.PRNGKey(args.seed)
    init_key, train_key = jr.split(key)
    if args.arch == "mlp":
        in_dim = int(x_train.shape[1])
        model = dp_psac.MLP(in_dim=in_dim, hidden=256, out_dim=10, key=init_key)
    elif args.arch == "cnn":
        model = dp_psac.CNN(in_channels=1, out_dim=10, key=init_key)
    else:
        raise ValueError(args.arch)

    _, metrics = dp_psac.train(
        model=model,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        sigmas=jnp.asarray(sigmas, dtype=jnp.float32),
        clips=jnp.asarray(clips, dtype=jnp.float32),
        batch_size=args.batch_size,
        lr=args.lr,
        r=args.r,
        key=train_key,
        log_every=args.log_every,
    )

    eps = accountant.epsilon_spent(sigmas, sample_rate=q, delta=args.delta)

    result = {
        "test_accuracy": float(metrics["test_accuracy"]),
        "final_train_loss": metrics["final_train_loss"],
        "epsilon_spent": eps,
        "delta": args.delta,
        "T": T,
        "B": args.batch_size,
        "q": q,
        "r": args.r,
        "lr": args.lr,
        "dataset": args.dataset,
        "arch": args.arch,
        "seed": args.seed,
    }
    print(json.dumps(result, indent=2))

    if args.out is not None:
        args.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
