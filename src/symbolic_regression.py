import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import tqdm
import tyro
from pysr import PySRRegressor

import wandb

CACHE_DIR = Path(__file__).parent / "cache" / "symbolic_regression"

# (per-sample input shape, nclasses) — matches net_factory.DATASET_NETWORK_DEFAULTS
DATASET_SHAPES: dict[str, tuple[tuple[int, ...], int]] = {
    "mnist": ((1, 28, 28), 10),
    "fashion-mnist": ((1, 28, 28), 10),
    "cifar-10": ((3, 32, 32), 10),
    "california": ((8,), 2),
    "eyepacs": ((3, 224, 224), 2),
}

# AutoNetworkConfig defaults — mirrors net_factory.DATASET_NETWORK_DEFAULTS
_AUTO_CNN = {
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
_AUTO_MLP = {
    "california": {"hidden_sizes": [64, 32]},
}


def _mlp_param_count(din: int, hidden_sizes: list[int], nclasses: int) -> int:
    """Count trainable parameters in an MLP matching MLP.from_config construction."""
    sizes = [din, *hidden_sizes, nclasses]
    total = 0
    for i in range(len(sizes) - 1):
        a, b = sizes[i], sizes[i + 1]
        total += a * b + b  # Linear: weight + bias
        if i < len(sizes) - 2:
            total += 2 * b  # LayerNorm on hidden layers: weight + bias
    return total


def _cnn_param_count(input_shape: tuple[int, ...], net_dict: dict, nclasses: int) -> int:
    """Count trainable parameters in a CNN matching CNN.from_config construction."""
    channels_list = net_dict.get("channels", [16, 32])
    kernels = net_dict.get("kernel_sizes", [8, 4])
    paddings = net_dict.get("paddings", [2, 0])
    strides = net_dict.get("strides", [2, 2])
    pool_k = net_dict.get("pool_kernel_size", 2)
    mlp_hidden = net_dict.get("mlp", {}).get("hidden_sizes", [32])

    total = 0
    in_ch, H, W = input_shape
    for out_ch, k, p, s in zip(channels_list, kernels, paddings, strides):
        total += in_ch * out_ch * k * k + out_ch  # Conv2d: weight + bias
        H = (H + 2 * p - k) // s + 1
        W = (W + 2 * p - k) // s + 1
        H = H // pool_k
        W = W // pool_k
        in_ch = out_ch

    total += _mlp_param_count(in_ch * H * W, mlp_hidden, nclasses)
    return total


def _compute_num_network_params(raw_config: dict) -> int | None:
    """Derive the number of trainable network parameters from a raw W&B run config."""
    dataset = raw_config.get("dataset")
    if dataset not in DATASET_SHAPES:
        return None
    input_shape, nclasses = DATASET_SHAPES[dataset]
    net_dict = raw_config.get("env", {}).get("network", {})
    net_type = net_dict.get("_type", "AutoNetworkConfig")

    if net_type == "AutoNetworkConfig":
        if dataset in _AUTO_CNN:
            net_type, net_dict = "CNNConfig", _AUTO_CNN[dataset]
        elif dataset in _AUTO_MLP:
            net_type, net_dict = "MLPConfig", _AUTO_MLP[dataset]
        else:
            return None

    din = 1
    for d in input_shape:
        din *= d

    if net_type == "MLPConfig":
        return _mlp_param_count(din, net_dict.get("hidden_sizes", [32]), nclasses)
    if net_type == "CNNConfig":
        return _cnn_param_count(input_shape, net_dict, nclasses)
    return None


@dataclass
class PySRConfig:
    wandb_proj: str
    wandb_entity: str
    mode: Literal["warmup_constant", "final_schedule"] = "final_schedule"
    """Which values to regress on:
      warmup_constant — the final constant sigma/clip at the end of the warmup phase.
      final_schedule  — the full optimised sigma/clip schedule at the end of training."""
    omit_features: tuple[str, ...] = ()
    keep_features: tuple[str, ...] = ()
    run_ids: tuple[str, ...] = ()
    """Explicit W&B run IDs to include.  When empty, all finished runs in the
    project are used."""


def compile_features(run_config: Any) -> dict[str, float | int] | float | int:
    if not isinstance(run_config, dict):
        return run_config

    flat_dict: dict[str, Any] = {}
    for key in run_config:
        nested_features = compile_features(run_config[key])

        if isinstance(nested_features, dict):
            for nested_key in nested_features:
                flat_dict[f"{key}.{nested_key}"] = nested_features[nested_key]
        else:
            flat_dict[key] = nested_features

    # Remove 'parameters' artifact from keys
    old_dict = copy.deepcopy(flat_dict)
    for old_key in old_dict:
        new_key = old_key.replace("parameters.", "").replace(".value", ".").replace(".", "_")
        if new_key != old_key:
            flat_dict[new_key] = flat_dict[old_key]
            del flat_dict[old_key]

    return flat_dict


def _compute_warmup_steps(run_config: dict) -> int:
    """Derive the number of warmup outer-loop steps from a W&B run config dict.

    The run config is stored at the SweepConfig level, so the keys are
    ``num_outer_steps`` and ``schedule_optimizer.schedule.warmup_pct``.
    """
    num_outer_steps = run_config.get("num_outer_steps", 100)
    schedule_conf = run_config.get("schedule_optimizer", {}).get("schedule", {})
    warmup_pct = schedule_conf.get("warmup_pct", 0.3)
    return max(1, int(warmup_pct * num_outer_steps))


def _cache_paths(run_id: str, mode: str) -> dict[str, Path]:
    return {
        "sigmas": CACHE_DIR / run_id / mode / "sigmas.parquet",
        "clips": CACHE_DIR / run_id / mode / "clips.parquet",
    }


def _filter_features(df: pd.DataFrame, conf: PySRConfig) -> pd.DataFrame:
    """Keep only numeric features, applying omit/keep filters from conf."""

    df.dropna(axis=1, inplace=True)

    # Remove constant columns
    df = df.loc[:, (df != df.iloc[0]).any()]

    if conf.keep_features:
        df = df[conf.keep_features]
    elif conf.omit_features:
        print(df.columns, [f in df.columns for f in conf.omit_features])
        df = df.drop(columns=[*conf.omit_features])
    return df


def _fetch_tables(
    conf: PySRConfig, api: wandb.Api, run_id: str
) -> tuple[pd.DataFrame, pd.DataFrame, wandb.Run]:
    """Download the raw sigmas and clips tables from a W&B run (no caching)."""
    run = api.run(f"{conf.wandb_entity}/{conf.wandb_proj}/{run_id}")

    tables: dict[str, pd.DataFrame] = {}
    target_artifacts = ("sigmas", "clips")
    for artifact in tqdm.tqdm(run.logged_artifacts(), desc=f"{run_id} artifacts"):
        for table_name in target_artifacts:
            if table_name in tables:
                continue
            if f"{table_name}:v" in artifact.name:
                table = artifact.get(table_name)
                tables[table_name] = pd.DataFrame(data=table.data, columns=table.columns)

        if len(tables) == len(target_artifacts):
            break

    assert "sigmas" in tables, f"No 'sigmas' table found in logged artifacts for run {run_id}"
    assert "clips" in tables, f"No 'clips' table found in logged artifacts for run {run_id}"

    return tables["sigmas"], tables["clips"], run


def run2dataset(conf: PySRConfig, api: wandb.Api, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = _cache_paths(run_id, conf.mode)

    if all(p.exists() for p in cache.values()):
        return pd.read_parquet(cache["sigmas"]), pd.read_parquet(cache["clips"])

    sigmas_raw, clips_raw, run = _fetch_tables(conf, api, run_id)

    num_outer_steps = run.config.get("num_outer_steps")
    if num_outer_steps is not None:
        for table_name, df in (("sigmas", sigmas_raw), ("clips", clips_raw)):
            if len(df) != num_outer_steps:
                raise ValueError(
                    f"Run {run_id}: '{table_name}' table has {len(df)} rows"
                    f" but num_outer_steps={num_outer_steps} — run likely crashed early, skipping."
                )

    flat_features = compile_features(run.config)
    assert isinstance(flat_features, dict)

    n_params = _compute_num_network_params(run.config)
    if n_params is not None:
        flat_features["num_network_params"] = n_params

    result: dict[str, pd.DataFrame] = {}

    if conf.mode == "warmup_constant":
        warmup_steps = _compute_warmup_steps(run.config)
        for table_name, value_col, df in (
            ("sigmas", "sigma", sigmas_raw),
            ("clips", "clip", clips_raw),
        ):
            # Keep only rows logged during the warmup phase, then take the last one.
            warmup_rows = df[df["step"] < warmup_steps]
            assert len(warmup_rows) > 0, (
                f"No warmup rows found for run {run_id} (warmup_steps={warmup_steps})"
            )
            last_warmup_row = warmup_rows.drop(columns=["step"]).iloc[-1]
            # Constant schedule → all inner-step columns are equal; collapse to scalar.
            constant_val = float(last_warmup_row.mean())
            row = {**flat_features, value_col: constant_val}
            result[table_name] = pd.DataFrame([row]).reset_index(drop=True)
    else:  # final_schedule
        for table_name, value_col, df in (
            ("sigmas", "sigma", sigmas_raw),
            ("clips", "clip", clips_raw),
        ):
            last_row = df.drop(columns=["step"]).iloc[-1]
            melted = last_row.reset_index()
            melted.columns = pd.Index(["step", value_col])
            melted["step"] = pd.to_numeric(melted["step"])

            for feat_key, feat_val in flat_features.items():
                melted[feat_key] = feat_val

            result[table_name] = melted[
                [*list(flat_features.keys()), "step", value_col]
            ].reset_index(drop=True)

    for table_name, df in result.items():
        cache[table_name].parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache[table_name], index=False)

    return result["sigmas"], result["clips"]


def run_regression(df: pd.DataFrame, target_col: str) -> PySRRegressor:
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    model = PySRRegressor(
        maxsize=30,
        niterations=2000,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[
            "sqrt",
            "exp",
            "log",
            "inv(x) = 1/x",
        ],
        extra_sympy_mappings={
            "inv": lambda x: 1 / x,
        },
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    )
    model.fit(X, y, variable_names=list(X.columns))
    return model


def main(conf: PySRConfig):
    api = wandb.Api()

    if conf.run_ids:
        run_ids = list(conf.run_ids)
    else:
        runs = api.runs(f"{conf.wandb_entity}/{conf.wandb_proj}", filters={"state": "finished"})
        run_ids = [r.id for r in runs]

    print(f"Processing {len(run_ids)} run(s) in mode='{conf.mode}'")

    all_sigmas: list[pd.DataFrame] = []
    all_clips: list[pd.DataFrame] = []

    for run_id in tqdm.tqdm(run_ids, desc="runs"):
        try:
            sigmas_df, clips_df = run2dataset(conf, api, run_id)
            all_sigmas.append(sigmas_df)
            all_clips.append(clips_df)
        except Exception as exc:
            print(f"  Skipping {run_id}: {exc}")

    assert all_sigmas, "No runs produced usable data."

    # Concatenate into one dataframe
    sigmas_combined = pd.concat(all_sigmas, ignore_index=True).reset_index(drop=True)
    clips_combined = pd.concat(all_clips, ignore_index=True).reset_index(drop=True)

    # Drop columns with at least one null value (i.e not present in all runs)
    sigmas_combined = _filter_features(sigmas_combined, conf)
    clips_combined = _filter_features(clips_combined, conf)

    print("=== Sigma regression ===")
    print(f"=== Features: {list(sigmas_combined.columns)} ===")
    sigma_model = run_regression(sigmas_combined, "sigma")
    print(sigma_model)

    print("=== Clip regression ===")
    print(f"=== Features: {list(clips_combined.columns)} ===")
    clip_model = run_regression(clips_combined, "clip")
    print(clip_model)


if __name__ == "__main__":
    conf = tyro.cli(PySRConfig)
    main(conf)
