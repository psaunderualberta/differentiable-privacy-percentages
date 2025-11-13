import os
import pickle
from typing import Any, Callable, List, Tuple, Mapping

import chex
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from typing import Optional
from util.baselines import Baseline

from util.aggregators import multi_line_plotter

import wandb


class ExperimentLogger(eqx.Module):
    directories: List[str]
    columns: List[str]
    large_columns: List[str]
    aggregators: List[Callable[[pd.DataFrame], Tuple[pd.Series, pd.Series, str]]]
    plotters: List[Callable[[pd.DataFrame], go.Figure]]
    baseline_plotters: List[Callable[[Baseline, pd.DataFrame], go.Figure]]

    def __init__(
        self,
        directories: List[str],
        columns: List[str],
        large_columns: List[str] = [],
        aggregators: List[Callable[[pd.DataFrame], go.Figure]] = [],
        clear_files: bool = True,
    ):
        self.directories = directories
        self.columns = columns
        self.large_columns = large_columns
        self.aggregators = [mean_loss_aggregator]

        self.plotters = [
            actions_plotter,
            losses_plotter,
            accuracy_plotter,
            policy_plotter,
        ]

        self.baseline_plotters = [
            Baseline.baseline_comparison_accuracy_plotter,
            Baseline.baseline_comparison_final_loss_plotter,
        ]

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

    def log(self, pos: int, data: Mapping[str, Any]):
        fname = self.get_data_file(pos)
        with open(fname, "a") as f:
            organized_data = [self.obj_to_str(data[col]) for col in self.columns]
            f.write(",".join(organized_data) + "\n")

    def log_multiple(self, pos: int, data: List[Mapping[str, Any]]) -> None:
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
        baseline: Optional[Baseline] = None,
        show: bool = False,
    ) -> Mapping[str, go.Figure]:
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

        if baseline is not None:
            for plotter in self.baseline_plotters:
                # Only change, creating a reward_analyzer w/ this specific config
                plotter_name = " ".join(
                    [str.capitalize(s) for s in plotter.__name__.split("_")][:-1]
                )
                fig = plotter(baseline, df)

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


class WandbTableLogger(eqx.Module):
    tables: Mapping[str, wandb.Table]
    cols: Mapping[str, list[str]]
    freqs: Mapping[str, int]
    counts: Mapping[str, int]

    def __init__(self, schemas: Mapping[str, list[str]], freqs: Mapping[str, int]):
        super().__init__()
        self.tables = {}
        self.cols = {}
        for name, cols in schemas.items():
            self.tables[name] = wandb.Table(columns=cols, log_mode="INCREMENTAL")
            self.cols[name] = cols

        self.freqs = {name: freqs.get(name, 1) for name in schemas.keys()}
        self.counts = {name: 0 for name in schemas.keys()}

    def log(
        self,
        name: str,
        data: Mapping[str, int | float],
        force: bool = False,
        plot: bool = False,
    ) -> bool:
        assert name in self.tables, (
            f"Name '{name}' not found in set of tables: {list(self.tables.keys())}"
        )
        log_time = self.counts[name] % self.freqs[name] == 0
        if log_time or force:
            table = self.tables[name]
            data_ordered = [jnp.asarray(data[col]).tolist() for col in table.columns]

            table.add_data(*data_ordered)

            if plot:
                self.line_plot(name, [data_ordered])

        self.counts[name] += 1
        return log_time or force

    def log_array(
        self,
        name: str,
        arr: chex.Array,
        aux: Mapping[str, object] | None = None,
        force: bool = False,
        plot: bool = False,
    ) -> bool:
        cols = {str(j): item for (j, item) in enumerate(arr)}

        aux = aux if isinstance(aux, dict) else dict()
        return self.log(name, cols | aux, force=force, plot=plot)

    def commit(self, metrics: Mapping[str, int] | None = None):
        if metrics is None:
            metrics = dict()
        assert len(self.tables.keys() & metrics) == 0, "Name Overlap"

        # For type checker
        assert isinstance(metrics, dict)
        wandb.log(metrics)

    def finish(self):
        final_tables = {
            name: wandb.Table(
                columns=table.columns, data=table.data, log_mode="IMMUTABLE"
            )
            for (name, table) in self.tables.items()
        }

        wandb.log(final_tables)

    def line_plot(self, table_name: str, data: list | None = None):
        wandb_table = self.tables[table_name]
        if data is None:
            df = pd.DataFrame(columns=wandb_table.columns, data=wandb_table.data)
        else:
            df = pd.DataFrame(columns=wandb_table.columns, data=data)

        # multi-line, but only plotting one line
        fig = multi_line_plotter(df, table_name)
        wandb.log({f"{table_name}-plot": fig})

    def bulk_line_plots(self, table_name: str):
        data = self.tables[table_name].data[-1]

        # implicitly checks for square-ness
        bulk_lines = jnp.asarray(data[-1])
        num_cols = bulk_lines.shape[-1]

        pd_compatible_data = []
        for i, line in enumerate(bulk_lines):
            pd_compatible_data.append([i, *[num.item() for num in line]])

        cols = ["run no.", *map(str, range(num_cols))]
        df = pd.DataFrame(columns=cols, data=pd_compatible_data)

        # multi-line, but only plotting one line
        fig = multi_line_plotter(df, table_name, color_indicator="run no.")
        wandb.log({f"{table_name}-plot": fig})
