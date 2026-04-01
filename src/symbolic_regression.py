import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import tqdm
import tyro
from pysr import PySRRegressor

import wandb

CACHE_DIR = Path(__file__).parent / "cache" / "symbolic_regression"


@dataclass
class PySRConfig:
    wandb_proj: str
    wandb_entity: str
    omit_features: tuple[str, ...] = ()
    keep_features: tuple[str, ...] = ()


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


def _cache_paths(run_id: str) -> dict[str, Path]:
    return {
        "sigmas": CACHE_DIR / run_id / "sigmas.parquet",
        "clips": CACHE_DIR / run_id / "clips.parquet",
    }


def run2dataset(conf: PySRConfig, api: wandb.Api, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = _cache_paths(run_id)

    # Return cached tables if both exist, skipping the artifact scan.
    if all(p.exists() for p in cache.values()):
        return pd.read_parquet(cache["sigmas"]), pd.read_parquet(cache["clips"])

    run = api.run(f"{conf.wandb_entity}/{conf.wandb_proj}/{run_id}")

    # Tables logged via wandb.log() with log_mode="IMMUTABLE" are stored as W&B artifacts.
    # Iterate logged artifacts and pull out sigmas / clips by name.
    tables: dict[str, pd.DataFrame] = {}
    target_artifacts = ("sigmas", "clips")
    for artifact in tqdm.tqdm(run.logged_artifacts()):
        for table_name in target_artifacts:
            if table_name in tables:
                continue
            try:
                table = artifact.get(table_name)
            except Exception:
                continue
            if table is not None:
                tables[table_name] = pd.DataFrame(data=table.data, columns=table.columns)

        # Early stop, no need to continue reading artifacts
        if len(tables) == len(target_artifacts):
            break

    assert "sigmas" in tables, f"No 'sigmas' table found in logged artifacts for run {run_id}"
    assert "clips" in tables, f"No 'clips' table found in logged artifacts for run {run_id}"

    flat_features = compile_features(run.config)
    assert isinstance(flat_features, dict)

    if len(conf.keep_features) > 0:
        for key in copy.deepcopy(list(flat_features.keys())):
            if type(flat_features[key]) not in [float, int] or key not in conf.keep_features:
                del flat_features[key]
    else:
        # Trim non-int and non-float values and omit user-specified features
        for key in copy.deepcopy(list(flat_features.keys())):
            if type(flat_features[key]) not in [float, int] or key in conf.omit_features:
                del flat_features[key]

    result: dict[str, pd.DataFrame] = {}
    for table_name, value_col in (("sigmas", "sigma"), ("clips", "clip")):
        df = tables[table_name]
        # The table has a "step" column (outer loop step) + one column per inner DP-SGD step.
        # Take the last row (final optimised schedule) and melt inner steps into rows.
        last_row = df.drop(columns=["step"]).iloc[-1]
        melted = last_row.reset_index()
        melted.columns = pd.Index(["step", value_col])
        melted["step"] = pd.to_numeric(melted["step"])

        # Broadcast flat_features across all rows
        for feat_key, feat_val in flat_features.items():
            melted[feat_key] = feat_val

        # Reorder: flat_features + step + value
        result[table_name] = melted[[*list(flat_features.keys()), "step", value_col]].reset_index(
            drop=True
        )

    # Persist to cache for future calls
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
    api = wandb.Api({"entity": "psaunder", "project": "Testing Mu-gdp"})
    sigmas_df, clips_df = run2dataset(conf, api, "vel38r12")

    print("=== Sigma regression ===")
    sigma_model = run_regression(sigmas_df, "sigma")
    print(sigma_model)

    print("=== Clip regression ===")
    clip_model = run_regression(clips_df, "clip")
    print(clip_model)


if __name__ == "__main__":
    conf = tyro.cli(PySRConfig)
    main(conf)
