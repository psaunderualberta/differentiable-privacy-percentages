from typing import cast

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

### ----
# Losses
### ---


def max_loss_aggregator(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    agg = cast(pd.DataFrame, df.groupby("step")["loss"].max().reset_index())
    return cast(pd.Series, agg["step"]), cast(pd.Series, agg["loss"]), "Max Loss"


def mean_loss_aggregator(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    agg = cast(pd.DataFrame, df.groupby("step")["loss"].mean().reset_index())
    return cast(pd.Series, agg["step"]), cast(pd.Series, agg["loss"]), "Mean Loss"


def min_loss_aggregator(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    agg = cast(pd.DataFrame, df.groupby("step")["loss"].min().reset_index())
    return cast(pd.Series, agg["step"]), cast(pd.Series, agg["loss"]), "Min Loss"


def std_loss_aggregator(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    std_series = cast(pd.Series, df.groupby("step")["loss"].std())
    agg = cast(pd.DataFrame, std_series.reset_index())
    return cast(pd.Series, agg["step"]), cast(pd.Series, agg["loss"]), "Std Dev of Loss"


### ----
# Accuracy
### ---


def max_accuracy_aggregator(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    agg = cast(pd.DataFrame, df.groupby("step")["accuracy"].max().reset_index())
    return cast(pd.Series, agg["step"]), cast(pd.Series, agg["accuracy"]), "Max Accuracy"


def mean_accuracy_aggregator(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    agg = cast(pd.DataFrame, df.groupby("step")["accuracy"].mean().reset_index())
    return cast(pd.Series, agg["step"]), cast(pd.Series, agg["accuracy"]), "Mean Accuracy"


def min_accuracy_aggregator(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    agg = cast(pd.DataFrame, df.groupby("step")["accuracy"].min().reset_index())
    return cast(pd.Series, agg["step"]), cast(pd.Series, agg["accuracy"]), "Min Accuracy"


def std_accuracy_aggregator(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, str]:
    std_series = cast(pd.Series, df.groupby("step")["accuracy"].std())
    agg = cast(pd.DataFrame, std_series.reset_index())
    return cast(pd.Series, agg["step"]), cast(pd.Series, agg["accuracy"]), "Std Dev of Accuracy"


def multi_line_plotter(
    df: pd.DataFrame,
    col_name: str,
    color_indicator: str = "step",
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
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
        },
    )

    return fig
