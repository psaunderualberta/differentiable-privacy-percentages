from pprint import pprint
from typing import Optional, Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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


def actions_plotter(
    df: pd.DataFrame, timesteps: Optional[list[int]] = None
) -> go.Figure:
    batches = [df["batch_idx"].min()]
    num_training_steps = df["step"].max()
    plotting_freq = SingletonConfig.get_sweep_config_instance().plotting_interval
    steps_to_log = [i for i in range(0, num_training_steps, plotting_freq)] + [
        num_training_steps
    ]
    return _actions_plotter(df, "actions", timesteps=steps_to_log, batches=batches)


def policy_plotter(
    df: pd.DataFrame, timesteps: Optional[list[int]] = None
) -> go.Figure:
    batches = [df["batch_idx"].min()]
    num_training_steps = df["step"].max()
    plotting_freq = (
        num_training_steps
        // SingletonConfig.get_experiment_config_instance().sweep.plotting_steps
    )
    steps_to_log = [i for i in range(0, num_training_steps, plotting_freq)] + [
        num_training_steps
    ]
    return _actions_plotter(df, "policy", timesteps=steps_to_log, batches=batches)


def lr_plotter(df: pd.DataFrame, timesteps: Optional[list[int]] = None) -> go.Figure:
    return _actions_plotter(df, "lrs", timesteps)


def losses_plotter(
    df: pd.DataFrame, timesteps: Optional[list[int]] = None
) -> go.Figure:
    timesteps = [df["step"].max()]
    batches = df["batch_idx"].unique()
    return _actions_plotter(df, "losses", timesteps=timesteps, batches=batches)


def accuracy_plotter(
    df: pd.DataFrame, timesteps: Optional[list[int]] = None
) -> go.Figure:
    return _actions_plotter(df, "accuracies", timesteps)


def multi_line_plotter(
    df: pd.DataFrame, col_name: str, color_indicator: str = "step"
) -> go.Figure:
    df_melted = df.melt(id_vars=color_indicator, var_name="iter", value_name=col_name)
    fig = px.line(
        df_melted,
        x="iter",
        y=col_name,
        color=color_indicator,
    )

    _ = fig.update_layout(
        xaxis_title="Private Training Step",
        yaxis_title=col_name.capitalize(),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig
