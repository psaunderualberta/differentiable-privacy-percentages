import os
import pickle
from typing import Any, Callable, Dict, List, Tuple

import chex
import equinox as eqx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from util.aggregators import (
    max_accuracy_aggregator,
    mean_accuracy_aggregator,
    min_accuracy_aggregator,
    std_accuracy_aggregator,
    max_loss_aggregator,
    mean_loss_aggregator,
    min_loss_aggregator,
    std_loss_aggregator,
    actions_plotter,
    losses_plotter,
    accuracy_plotter
)

import wandb

class ExperimentLogger(eqx.Module):
    directories: List[str]
    columns: List[str]
    large_columns: List[str]
    aggregators: List[Callable[[pd.DataFrame], Tuple[pd.Series, pd.Series, str]]]
    plotters: List[Callable[[pd.DataFrame], go.Figure]]
    # baseline_plotters: List[Callable[[RewardAnalyzer, pd.DataFrame], go.Figure]]

    def __init__(
        self,
        directories: List[str],
        columns: List[str],
        large_columns: List[str] = [],
        aggregators: List[Callable[[pd.DataFrame], go.Figure]] = [],
        clear_files: bool = True
    ):
        self.directories = directories
        self.columns = columns
        self.large_columns = large_columns
        self.aggregators = [mean_loss_aggregator]

        self.plotters = [
            actions_plotter,
            losses_plotter,
            accuracy_plotter,
        ]

        # self.baseline_plotters = []

        for pos, directory in enumerate(directories):
            # create folder if it doesn't exist
            # slight opportunity for race condition here, but it's fine bc I say so
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            if clear_files:
                fname = self.get_data_file(pos)
                with open(fname, "w") as f:
                    f.write(",".join(columns) + "\n")

    def get_data_file(self, pos: int) -> str:
        return os.path.join(self.directories[pos], "data.csv")

    def write_object(self, pos: int, fname: str, data: Any) -> None:
        with open(os.path.join(self.directories[pos], fname), "wb") as f:
            pickle.dump(data, f)

    def read_object(self, pos: int, fname: str) -> Any:
        with open(os.path.join(self.directories[pos], fname), "rb") as f:
            obj = pickle.load(f)

        return obj

    def log(self, pos: int, data: Dict[str, Any]):
        fname = self.get_data_file(pos)
        with open(fname, "a") as f:
            organized_data = [self.obj_to_str(data[col]) for col in self.columns]
            f.write(",".join(organized_data) + "\n")

    def log_multiple(self, pos: int, data: List[Dict[str, Any]]) -> None:
        fname = self.get_data_file(pos)
        with open(fname, "a") as f:
            for d in data:
                organized_data = [self.obj_to_str(d[col]) for col in self.columns]
                f.write(",".join(organized_data) + "\n")

    def obj_to_str(self, obj: Any):
        if isinstance(obj, chex.Array) and "[" in str(obj.tolist()):
            # remove nans
            arr = obj.tolist()
            arr_wo_nan = [num for num in arr if ~np.isnan(num)]
            return f'"{str(arr_wo_nan)}"'  # add quotes for parsability

        return str(obj)

    def create_plots(
        self,
        pos: int,
        aggregators: List[Callable[[pd.DataFrame], go.Figure]] = [],
        plotters: List[Callable[[pd.DataFrame], go.Figure]] = [],
        log_to_wandb: bool = False,
        return_aggregates=False,
        with_baselines: bool = False,
        show: bool = False,
    ) -> Dict[str, go.Figure]:

        if len(aggregators) == 0:
            aggregators = self.aggregators

        if len(plotters) == 0:
            plotters = self.plotters

        fname = self.get_data_file(pos)
        df = pd.read_csv(fname)
        aggregates = {}
        max_len = 0

        # 'step' in wandb.log must be monotonically increasing, so must aggregate all data, then log
        for aggregator in aggregators:
            steps, feats, col_name = aggregator(df)
            aggregator_name = " ".join(
                [str.capitalize(s) for s in aggregator.__name__.split("_")][:-1]
            )
            aggregates[aggregator_name] = (steps, feats, col_name)
            max_len = max(max_len, len(steps))

        if log_to_wandb:
            for pos in range(max_len):
                data = {}

                for aggregator in aggregates:
                    steps, feats, col_name = aggregates[aggregator]
                    if pos < len(steps):
                        data[col_name] = feats[pos]

                if log_to_wandb:
                    wandb.log(data, step=steps[pos])

        for plotter in plotters:
            plotter_name = " ".join(
                [str.capitalize(s) for s in plotter.__name__.split("_")][:-1]
            )
            fig = plotter(df)

            if log_to_wandb:
                wandb.log({plotter_name: fig})

        # Expensive computations, only do if instructed
        if with_baselines:
            reward_analyzer = RewardAnalyzer(env_config['lr'])
            for plotter in self.baseline_plotters:
                # Only change, creating a reward_analyzer w/ this specific config
                plotter_name = " ".join(
                    [str.capitalize(s) for s in plotter.__name__.split("_")][:-1]
                )
                fig = plotter(reward_analyzer, df)

                if log_to_wandb:
                    wandb.log({plotter_name: fig}) 

        return aggregates

    def get_csv(
        self, pos: int, log_to_wandb: bool = False, with_large_columns: bool = False
    ) -> pd.DataFrame:
        fname = self.get_data_file(pos)
        df = pd.read_csv(fname)

        if not with_large_columns:
            df = df.drop(self.large_columns, axis=1)

        if log_to_wandb:
            wandb.log({"data": wandb.Table(dataframe=df)})

        return df
