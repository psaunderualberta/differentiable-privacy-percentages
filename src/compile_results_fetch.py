#!/usr/bin/env python3
"""compile_results_fetch.py — Pull per-run scalars and final-schedule arrays
from a W&B project produced by ``create_experiments.py``, and write four
artefacts under a cache dir:

    scalars.parquet    one row per (run_id, schedule)
    schedules.parquet  one row per (run_id, inner_step, var ∈ {sigma, clip})
    histories.parquet  one row per (run_id, outer_step) — Learned only
    missing.csv        runs that were skipped, with reason

Run once per project; ``compile_results_plot.py`` and ``symbolic_regression.py``
both read these caches.

Usage (from src/):
    uv run compile_results_fetch.py --project schedule-T-arch --entity <entity>
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import tqdm
import tyro

import wandb

CACHE_ROOT = Path(__file__).parent / "cache" / "results"

# Mirrors symbolic_regression.DATASET_SHAPES / _AUTO_CNN / _AUTO_MLP. Kept local
# so this script is independent of the training code.
DATASET_SHAPES: dict[str, tuple[tuple[int, ...], int]] = {
    "mnist": ((1, 28, 28), 10),
    "fashion-mnist": ((1, 28, 28), 10),
    "cifar-10": ((3, 32, 32), 10),
    "california": ((8,), 2),
    "eyepacs": ((3, 224, 224), 2),
}

_AUTO_CNN: dict[str, dict] = {
    "mnist": {
        "channels": [16, 32],
        "kernel_sizes": [8, 4],
        "paddings": [2, 0],
        "strides": [2, 2],
        "pool_kernel_size": 2,
        "mlp": {"hidden_sizes": [32]},
    },
    "fashion-mnist": {
        "channels": [16, 32],
        "kernel_sizes": [8, 4],
        "paddings": [2, 0],
        "strides": [2, 2],
        "pool_kernel_size": 2,
        "mlp": {"hidden_sizes": [32]},
    },
    "cifar-10": {
        "channels": [32, 64],
        "kernel_sizes": [3, 3],
        "paddings": [1, 1],
        "strides": [1, 1],
        "pool_kernel_size": 2,
        "mlp": {"hidden_sizes": [256]},
    },
    "eyepacs": {
        "channels": [16, 32],
        "kernel_sizes": [8, 4],
        "paddings": [2, 0],
        "strides": [2, 2],
        "pool_kernel_size": 2,
        "mlp": {"hidden_sizes": [32]},
    },
}
_AUTO_MLP: dict[str, dict] = {"california": {"hidden_sizes": [64, 32]}}

# Only T values used by the T-sweep axis in create_experiments.py. Used as an
# axis-tag fallback when run.tags is empty.
_T_SWEEP_T_VALUES: set[int] = {1000, 1500, 2000, 3000, 5000}

_OPTIMIZER_TYPE_TO_NAME: dict[str, str] = {
    "SGDConfig": "sgd",
    "AdamConfig": "adam",
    "AdamWConfig": "adamw",
}

_BASELINE_SCHEDULES: tuple[str, ...] = (
    "Constant σ/clip",  # noqa: RUF001
    "Clip to Median Gradient Norm",
    "Dynamic-DPSGD",
)


# ---------------------------------------------------------------------------
# Param-count helpers (mirror symbolic_regression.py)
# ---------------------------------------------------------------------------


def _mlp_param_count(din: int, hidden_sizes: list[int], nclasses: int) -> int:
    sizes = [din, *hidden_sizes, nclasses]
    total = 0
    for i in range(len(sizes) - 1):
        a, b = sizes[i], sizes[i + 1]
        total += a * b + b
        if i < len(sizes) - 2:
            total += 2 * b
    return total


def _cnn_param_count(input_shape: tuple[int, ...], net: dict, nclasses: int) -> int:
    channels = net.get("channels", [16, 32])
    kernels = net.get("kernel_sizes", [8, 4])
    paddings = net.get("paddings", [2, 0])
    strides = net.get("strides", [2, 2])
    pool_k = net.get("pool_kernel_size", 2)
    mlp_hidden = net.get("mlp", {}).get("hidden_sizes", [32])

    total = 0
    in_ch, h, w = input_shape
    for out_ch, k, p, s in zip(channels, kernels, paddings, strides):
        total += in_ch * out_ch * k * k + out_ch
        h = (h + 2 * p - k) // s + 1
        w = (w + 2 * p - k) // s + 1
        h //= pool_k
        w //= pool_k
        in_ch = out_ch

    total += _mlp_param_count(in_ch * h * w, mlp_hidden, nclasses)
    return total


# ---------------------------------------------------------------------------
# Config interpretation
# ---------------------------------------------------------------------------


def resolve_optimizer(env_dict: dict) -> str:
    """Map ``env.optimizer`` to {"sgd", "adam", "adamw"}.

    Handles both legacy literal-string runs and current OptimizerConfig dicts.
    """
    opt = env_dict.get("optimizer")
    if isinstance(opt, str):
        name = opt.lower()
        if name not in {"sgd", "adam", "adamw"}:
            raise ValueError(f"unknown optimizer string: {opt!r}")
        return name
    if isinstance(opt, dict):
        t = opt.get("_type")
        if t in _OPTIMIZER_TYPE_TO_NAME:
            return _OPTIMIZER_TYPE_TO_NAME[t]
        raise ValueError(f"unknown OptimizerConfig _type: {t!r}")
    raise ValueError(f"missing or unrecognised env.optimizer: {opt!r}")


def _arch_info(env_dict: dict, dataset: str) -> tuple[str, int | None]:
    """Return ``(label, num_params)``. Mirrors create_experiments._arch_label."""
    net = env_dict.get("network", {})
    net_type = net.get("_type", "AutoNetworkConfig")

    resolved_net = net
    if net_type == "AutoNetworkConfig":
        if dataset in _AUTO_CNN:
            net_type, resolved_net = "CNNConfig", _AUTO_CNN[dataset]
        elif dataset in _AUTO_MLP:
            net_type, resolved_net = "MLPConfig", _AUTO_MLP[dataset]

    if net_type == "MLPConfig":
        hs = list(resolved_net.get("hidden_sizes", []))
        label = "mlp-" + "x".join(str(h) for h in hs)
    elif net_type == "CNNConfig":
        ch = "x".join(str(c) for c in resolved_net.get("channels", []))
        head = "x".join(str(h) for h in resolved_net.get("mlp", {}).get("hidden_sizes", []))
        label = f"cnn-{ch}-head{head}"
    else:
        label = f"unknown-{net_type}"

    n_params: int | None = None
    if dataset in DATASET_SHAPES:
        input_shape, nclasses = DATASET_SHAPES[dataset]
        din = 1
        for d in input_shape:
            din *= d
        if net_type == "MLPConfig":
            n_params = _mlp_param_count(din, list(resolved_net.get("hidden_sizes", [])), nclasses)
        elif net_type == "CNNConfig":
            n_params = _cnn_param_count(input_shape, resolved_net, nclasses)
    return label, n_params


def _seed(cfg: dict) -> int | None:
    raw = cfg.get("prng_seed")
    if isinstance(raw, dict):
        v = raw.get("value")
        return int(v) if v is not None else None
    if raw is None:
        return None
    return int(raw)


def _axis(tags: list[str], T: int | None) -> str:
    if "T-sweep" in tags:
        return "T-sweep"
    if "arch-sweep" in tags:
        return "arch-sweep"
    if T is not None and T in _T_SWEEP_T_VALUES:
        return "T-sweep"
    return "arch-sweep"


# ---------------------------------------------------------------------------
# Per-run fetch
# ---------------------------------------------------------------------------


def _history(run: Any) -> list[dict]:
    """Return the full per-outer-step history as a list of dicts.

    Each dict has keys: outer_step, val_acc, val_loss. NaN/Inf values are kept
    so downstream plotting can show divergence as a break in the curve.
    """
    rows = list(run.scan_history(keys=["val-accuracy", "val-loss"]))
    if not rows:
        raise RuntimeError("no val-accuracy / val-loss rows in run history")
    return [
        {
            "outer_step": i,
            "val_acc": float(r["val-accuracy"]),
            "val_loss": float(r["val-loss"]),
        }
        for i, r in enumerate(rows)
    ]


def _baseline_means(api: wandb.Api, entity: str, project: str, run_id: str) -> pd.DataFrame:
    artifact = api.artifact(f"{entity}/{project}/baseline-{run_id}:latest")
    local = Path(artifact.download())
    pkls = list(local.glob("*.pkl"))
    if not pkls:
        raise RuntimeError("baseline artifact has no .pkl file")
    df = pd.read_pickle(str(pkls[0]))
    if not {"type", "accuracy", "loss"}.issubset(df.columns):
        raise RuntimeError(f"baseline df missing required cols: {df.columns}")
    return df


def _final_schedule_arrays(run: Any) -> tuple[list[float], list[float]]:
    """Pull the final-outer-step row from the sigmas/clips W&B tables."""
    tables: dict[str, pd.DataFrame] = {}
    targets = ("sigmas", "clips")
    for art in run.logged_artifacts():
        for tn in targets:
            if tn in tables:
                continue
            if f"{tn}:v" in art.name:
                t = art.get(tn)
                tables[tn] = pd.DataFrame(data=t.data, columns=t.columns)
        if len(tables) == len(targets):
            break
    for tn in targets:
        if tn not in tables:
            raise RuntimeError(f"missing '{tn}' artifact")

    def _final_row(df: pd.DataFrame) -> list[float]:
        cols = [c for c in df.columns if c != "step"]
        return [float(v) for v in df[cols].iloc[-1].tolist()]

    return _final_row(tables["sigmas"]), _final_row(tables["clips"])


def _fetch_one_run(
    api: wandb.Api, run: Any, entity: str, project: str
) -> tuple[list[dict], list[dict], list[dict]]:
    cfg = run.config
    env = cfg.get("env", {}) or {}
    dataset = cfg.get("dataset")
    if dataset is None:
        raise RuntimeError("missing dataset in run.config")

    eps = float(env.get("eps")) if env.get("eps") is not None else None
    T = int(env.get("num_training_steps")) if env.get("num_training_steps") is not None else None
    seed = _seed(cfg)
    arch_label, n_params = _arch_info(env, dataset)
    optimizer = resolve_optimizer(env)
    axis = _axis(list(run.tags or []), T)

    common = {
        "run_id": run.id,
        "run_name": run.name,
        "dataset": dataset,
        "eps": eps,
        "T": T,
        "arch_label": arch_label,
        "arch_param_count": n_params,
        "seed": seed,
        "axis": axis,
        "optimizer": optimizer,
    }

    history = _history(run)
    learned_acc = history[-1]["val_acc"]
    learned_loss = history[-1]["val_loss"]
    bdf = _baseline_means(api, entity, project, run.id)
    means = bdf.groupby("type")[["accuracy", "loss"]].mean()
    counts = bdf.groupby("type").size()

    scalars: list[dict] = []
    scalars.append(
        {
            **common,
            "schedule": "Learned Schedule",
            "mean_acc": learned_acc,
            "mean_loss": learned_loss,
            "n_reps": 1,
        }
    )
    for sched in _BASELINE_SCHEDULES:
        if sched not in means.index:
            continue
        scalars.append(
            {
                **common,
                "schedule": sched,
                "mean_acc": float(means.loc[sched, "accuracy"]),
                "mean_loss": float(means.loc[sched, "loss"]),
                "n_reps": int(counts.loc[sched]),
            }
        )

    sigmas, clips = _final_schedule_arrays(run)
    if T is not None and (len(sigmas) != T or len(clips) != T):
        raise RuntimeError(
            f"final schedule length mismatch (sigmas={len(sigmas)}, clips={len(clips)}, T={T})"
        )

    schedule_rows: list[dict] = []
    for inner_step, (s_val, c_val) in enumerate(zip(sigmas, clips)):
        step_norm = inner_step / T if T else None
        schedule_rows.append(
            {
                **common,
                "inner_step": inner_step,
                "step_norm": step_norm,
                "sigma": s_val,
                "clip": c_val,
            }
        )

    history_rows: list[dict] = [{**common, **h} for h in history]

    return scalars, schedule_rows, history_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class FetchConfig:
    project: str
    entity: str
    out_dir: str = ""
    """Cache directory. Defaults to src/cache/results/<entity>__<project>/."""
    limit: int = 0
    """If >0, fetch only this many runs (debugging)."""


def main(conf: FetchConfig) -> None:
    out_dir = Path(conf.out_dir) if conf.out_dir else CACHE_ROOT / f"{conf.entity}__{conf.project}"
    out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    runs = list(
        api.runs(
            f"{conf.entity}/{conf.project}", filters={"state": {"$in": ["crashed", "finished"]}}
        )
    )
    if conf.limit > 0:
        runs = runs[: conf.limit]
    print(f"{len(runs)} finished runs in {conf.entity}/{conf.project}")

    scalars: list[dict] = []
    schedules: list[dict] = []
    histories: list[dict] = []
    missing: list[dict] = []

    for run in tqdm.tqdm(runs, desc="runs"):
        try:
            s, sch, hist = _fetch_one_run(api, run, conf.entity, conf.project)
            scalars.extend(s)
            schedules.extend(sch)
            histories.extend(hist)
        except Exception as exc:
            missing.append({"run_id": run.id, "run_name": run.name, "reason": str(exc)})
            tqdm.tqdm.write(f"  skipping {run.id} ({run.name}): {exc}")

    scalars_df = pd.DataFrame(scalars)
    schedules_df = pd.DataFrame(schedules)
    histories_df = pd.DataFrame(histories)
    missing_df = pd.DataFrame(missing)

    scalars_df.to_parquet(out_dir / "scalars.parquet", index=False)
    schedules_df.to_parquet(out_dir / "schedules.parquet", index=False)
    histories_df.to_parquet(out_dir / "histories.parquet", index=False)
    missing_df.to_csv(out_dir / "missing.csv", index=False)

    print(f"\n→ {out_dir}")
    print(f"  scalars.parquet:   {len(scalars_df)} rows")
    print(f"  schedules.parquet: {len(schedules_df)} rows")
    print(f"  histories.parquet: {len(histories_df)} rows")
    print(f"  missing.csv:       {len(missing_df)} runs")


if __name__ == "__main__":
    main(tyro.cli(FetchConfig))
