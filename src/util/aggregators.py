from typing import Tuple

import jax.numpy as jnp
import pandas as pd
import plotly.express as px
from pprint import pprint
import plotly.graph_objects as go
from typing import Optional

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
    return _actions_plotter(df, "actions", timesteps)


def lr_plotter(df: pd.DataFrame, timesteps: Optional[list[int]] = None) -> go.Figure:
    return _actions_plotter(df, "lrs", timesteps)

def losses_plotter(df: pd.DataFrame, timesteps: Optional[list[int]] = None) -> go.Figure:
    return _actions_plotter(df, "losses", timesteps)


def accuracy_plotter(df: pd.DataFrame, timesteps: Optional[list[int]] = None) -> go.Figure:
    return _actions_plotter(df, "accuracies", timesteps)


def _actions_plotter(df: pd.DataFrame, col_name: str, timesteps: Optional[list[int]] = None) -> go.Figure:
    if timesteps is None:
        idxs = df["step"] == df["step"].max()
    else:
        idxs = df["step"].isin(timesteps)
    final_df = df[idxs][["step", "batch_idx", col_name]]
    final_df[col_name] = final_df[col_name].apply(str_to_jnp_array)

    final_df["step"] = [
        list(range(final_df[col_name].iloc[-1].size))
        for _ in range(len(final_df))
    ]

    # Flatten actions into a single list
    final_df = (
        final_df
        .explode(['step', col_name])
        .reset_index(drop=True)
    )

    fig = px.line(
        final_df,
        x="step",
        y=col_name,
        color="batch_idx",
    )

    fig.update_layout(
        xaxis_title="Training Step",
        yaxis_title=col_name.capitalize(),
    )

    return fig
