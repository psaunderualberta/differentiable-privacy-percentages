"""CLI entrypoint for the standalone DP-PSAC runner.

Usage (local schedules):
    uv run run.py schedule:local-schedule \
        --schedule.sigmas sigmas.npy --schedule.clips clips.npy \
        --dataset mnist --batch-size 512 --lr 1.0 --r 0.1 \
        --delta 1e-5 --arch cnn --seed 0 --out results.json

Usage (W&B checkpoint — learned schedule):
    uv run run.py schedule:wandb-schedule \
        --schedule.run-id <run_id> --schedule.entity <entity> \
        --schedule.project <project> \
        --dataset mnist --batch-size 512 --lr 1.0 --r 0.1 \
        --delta 1e-5 --arch cnn --seed 0 --out results.json

Usage (W&B baseline — Dynamic-DPSGD schedule from baseline artifact):
    uv run run.py schedule:wandb-baseline-schedule \
        --schedule.run-id <run_id> --schedule.entity <entity> \
        --schedule.project <project> \
        --dataset mnist --batch-size 512 --lr 1.0 --r 0.1 \
        --delta 1e-5 --arch cnn --seed 0 --out results.json
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import accountant
import dp_psac
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import plotille
import tyro
from opacus.accountants.utils import get_noise_multiplier

import data
import wandb


@dataclass
class LocalSchedule:
    clip: float = 0.1
    T: int = 2000


@dataclass
class WandbSchedule:
    run_id: str
    entity: str
    project: str
    step: int | None = None  # None → "latest" checkpoint


@dataclass
class WandbBaselineSchedule:
    run_id: str
    entity: str
    project: str


@dataclass
class SGDConfig:
    momentum: float = 0.9


@dataclass
class AdamConfig:
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8


@dataclass
class AdamWConfig:
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 1e-4


def _build_optimizer(lr: float, cfg: SGDConfig | AdamConfig | AdamWConfig):
    if isinstance(cfg, SGDConfig):
        return optax.sgd(lr, momentum=cfg.momentum)
    if isinstance(cfg, AdamConfig):
        return optax.adam(lr, b1=cfg.b1, b2=cfg.b2, eps=cfg.eps)
    return optax.adamw(lr, b1=cfg.b1, b2=cfg.b2, eps=cfg.eps, weight_decay=cfg.weight_decay)


@dataclass
class Args:
    schedule: LocalSchedule | WandbSchedule | WandbBaselineSchedule
    dataset: str = "mnist"
    arch: str = "mlp"  # "mlp" or "cnn"
    batch_size: int = 250
    lr: float = 1.0
    r: float = 0.1
    delta: float = 1e-6
    eps: float = 1
    seed: int = 0
    out: Path | None = None
    log_every: int = 50
    optimizer: SGDConfig | AdamConfig | AdamWConfig = dataclasses.field(default_factory=SGDConfig)


def _load_schedules(args: Args, n: int, q: float) -> tuple[np.ndarray, np.ndarray]:
    schedule = args.schedule
    if isinstance(schedule, LocalSchedule):
        noise_multiplier = get_noise_multiplier(
            target_epsilon=args.eps,
            target_delta=args.delta,
            sample_rate=q,
            steps=schedule.T,
            accountant="rdp",
        )
        clips = np.full(schedule.T, schedule.clip, dtype=np.float32)
        sigmas = clips * np.full(schedule.T, noise_multiplier, dtype=np.float32)
        return sigmas, clips

    api = wandb.Api()

    if isinstance(schedule, WandbSchedule):
        alias = "latest" if schedule.step is None else f"step-{schedule.step}"
        artifact_path = f"{schedule.entity}/{schedule.project}/checkpoint-{schedule.run_id}:{alias}"
    else:
        artifact_path = f"{schedule.entity}/{schedule.project}/baseline-{schedule.run_id}:latest"

    print(f"Downloading artifact: {artifact_path}")
    artifact = api.artifact(artifact_path)
    local_dir = Path(artifact.download())
    return np.load(local_dir / "sigmas.npy"), np.load(local_dir / "clips.npy")


def main(args: Args) -> None:
    x_train, y_train, x_test, y_test = data.load(args.dataset, args.arch)
    n = int(x_train.shape[0])
    q = args.batch_size / n
    print(f"dataset={args.dataset} arch={args.arch} N={n} B={args.batch_size} q={q:.6f}")

    sigmas, clips = _load_schedules(args, n, q)
    assert sigmas.shape == clips.shape and sigmas.ndim == 1, (
        f"sigmas and clips must be 1D arrays of equal length; got {sigmas.shape} and {clips.shape}"
    )
    T = int(sigmas.shape[0])
    fig = plotille.Figure()
    fig.width = 60
    fig.height = 15
    fig.x_label = "step"
    fig.y_label = "value"
    fig.plot(range(T), sigmas.tolist(), label="sigma")
    fig.plot(range(T), clips.tolist(), label="clip")
    fig.set_x_limits(min_=0.0, max_=T)
    print(fig.show(legend=True))

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
        optimizer=_build_optimizer(args.lr, args.optimizer),
    )

    fig = plotille.Figure()
    fig.width = 60
    fig.height = 15
    fig.x_label = "step"
    fig.y_label = "loss"
    fig.plot(range(T), metrics["train_losses"])
    fig.set_x_limits(min_=0.0, max_=T)
    print(fig.show())

    eps = accountant.epsilon_spent(sigmas / clips, sample_rate=q, delta=args.delta)

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
