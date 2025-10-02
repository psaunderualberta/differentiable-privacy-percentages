from typing import Tuple

import jax.numpy as jnp
import pandas as pd
import plotly.express as px
from pprint import pprint
import plotly.graph_objects as go
from typing import Optional
import numpy as np
from conf.singleton_conf import SingletonConfig

from util.util import str_to_jnp_array


### ----
# Losses
### ---


def max_loss_aggregator(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, str]:
    agg = df.groupby("step")["loss"].max().reset_index()
    return agg["step"], agg["loss"], "Max Loss"


def mean_loss_aggregator(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, str]:
    agg = df.groupby("step")["loss"].mean().reset_index()
    return agg["step"], agg["loss"], "Mean Loss"


def min_loss_aggregator(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, str]:
    agg = df.groupby("step")["loss"].min().reset_index()
    return agg["step"], agg["loss"], "Min Loss"


def std_loss_aggregator(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, str]:
    agg = df.groupby("step")["loss"].std().reset_index()
    return agg["step"], agg["loss"], "Std Dev of Loss"


### ----
# Accuracy
### ---


def max_accuracy_aggregator(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, str]:
    agg = df.groupby("step")["accuracy"].max().reset_index()
    return agg["step"], agg["accuracy"], "Max Accuracy"


def mean_accuracy_aggregator(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, str]:
    agg = df.groupby("step")["accuracy"].mean().reset_index()
    return agg["step"], agg["accuracy"], "Mean Accuracy"


def min_accuracy_aggregator(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, str]:
    agg = df.groupby("step")["accuracy"].min().reset_index()
    return agg["step"], agg["accuracy"], "Min Accuracy"


def std_accuracy_aggregator(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, str]:
    agg = df.groupby("step")["accuracy"].std().reset_index()
    return agg["step"], agg["accuracy"], "Std Dev of Accuracy"


### ---
# Actions
### ---


def actions_plotter(df: pd.DataFrame, timesteps: Optional[list[int]] = None) -> go.Figure:
    batches = [df["batch_idx"].min()]
    num_training_steps = df["step"].max()
    plotting_freq = num_training_steps // SingletonConfig.get_experiment_config_instance().sweep.plotting_steps
    steps_to_log = [i for i in range(0, num_training_steps, plotting_freq)] + [num_training_steps]
    return _actions_plotter(df, "actions", timesteps=steps_to_log, batches=batches)


def policy_plotter(df: pd.DataFrame, timesteps: Optional[list[int]] = None) -> go.Figure:
    batches = [df["batch_idx"].min()]
    num_training_steps = df["step"].max()
    plotting_freq = num_training_steps // SingletonConfig.get_experiment_config_instance().sweep.plotting_steps
    steps_to_log = [i for i in range(0, num_training_steps, plotting_freq)] + [num_training_steps]
    return _actions_plotter(df, "policy", timesteps=steps_to_log, batches=batches)


def lr_plotter(df: pd.DataFrame, timesteps: Optional[list[int]] = None) -> go.Figure:
    return _actions_plotter(df, "lrs", timesteps)


def losses_plotter(df: pd.DataFrame, timesteps: Optional[list[int]] = None) -> go.Figure:
    timesteps = [df["step"].max()]
    batches = df["batch_idx"].unique()
    return _actions_plotter(df, "losses", timesteps=timesteps, batches=batches)


def accuracy_plotter(df: pd.DataFrame, timesteps: Optional[list[int]] = None) -> go.Figure:
    return _actions_plotter(df, "accuracies", timesteps)


def _actions_plotter(df: pd.DataFrame, col_name: str, timesteps: Optional[list[int]] = None, batches: Optional[list[int] | np.ndarray] = None) -> go.Figure:
    if timesteps is None:
        timesteps = [df["step"].max()]
    if batches is None:
        batches = df['batch_idx'].unique()

    assert timesteps is not None
    assert batches is not None
    assert (len(timesteps) == 1) or (len(batches) == 1), f"len(timesteps) = {len(timesteps)},len(batches) = {len(batches)}"
    idxs = df["step"].isin(timesteps) & df["batch_idx"].isin(batches)

    final_df = df[idxs][["step", "batch_idx", col_name]]
    final_df[col_name] = final_df[col_name].apply(str_to_jnp_array)

    final_df["iter"] = final_df.apply(lambda row: list(range(len(row[col_name]))), axis=1)

    # Flatten actions into a single list
    final_df = (
        final_df
        .explode(['iter', col_name])
        .reset_index(drop=True)
    )

    color_indicator = None
    if len(timesteps) > 1:
        color_indicator = 'step'
    else:
        color_indicator = 'batch_idx'

    fig = px.line(
        final_df,
        x="iter",
        y=col_name,
        color=color_indicator,
    )

    fig.update_layout(
        xaxis_title="Training Step",
        yaxis_title=col_name.capitalize(),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        )
    )

    return fig
