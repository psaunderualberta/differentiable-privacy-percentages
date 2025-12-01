from typing import Mapping

from jaxtyping import Array
import jax.numpy as jnp
import equinox as eqx
import pandas as pd
import wandb
from conf.singleton_conf import SingletonConfig
from util.aggregators import multi_line_plotter


class Loggable(eqx.Module):
    table_name: str
    data: dict[str, int | float | Array]
    plot: bool = False
    commit: bool = False
    force: bool = False
    add_timestep: bool = True


class LoggableArray(eqx.Module):
    table_name: str
    array: Array
    aux: dict[str, object] | None = None
    plot: bool = False
    commit: bool = False
    force: bool = False
    add_timestep: bool = True


class LoggingSchema(eqx.Module):
    table_name: str
    cols: list[str]
    freq: int = SingletonConfig.get_sweep_config_instance().plotting_interval
    add_step_column: bool = True


class WandbTableLogger(eqx.Module):
    tables: dict[str, wandb.Table]
    cols: dict[str, list[str]]
    freqs: dict[str, int]
    counts: dict[str, int]

    def __init__(self):
        self.tables = dict()
        self.cols = dict()
        self.freqs = dict()
        self.counts = dict()

    def add_schema(self, schema: LoggingSchema):
        assert schema.table_name not in self.tables, (
            f"Table name '{schema.table_name}' already exists in logger."
        )
        name = schema.table_name
        cols = ["step"] + schema.cols if schema.add_step_column else schema.cols
        self.tables[name] = wandb.Table(columns=cols, log_mode="INCREMENTAL")
        self.cols[name] = cols
        self.freqs[name] = schema.freq
        self.counts[name] = 0

    def log(
        self,
        item: Loggable | LoggableArray,
    ) -> bool:
        # Convert to loggable
        if isinstance(item, LoggableArray):
            return self.log_array(item)

        name = item.table_name
        data = item.data
        force = item.force
        plot = item.plot
        assert name in self.tables, (
            f"Name '{name}' not found in set of tables: {list(self.tables.keys())}"
        )
        log_time = self.counts[name] % self.freqs[name] == 0
        if log_time or force:
            table = self.tables[name]

            if item.add_timestep:
                data["step"] = self.counts[name]

            data_ordered = [jnp.asarray(data[col]).tolist() for col in table.columns]
            table.add_data(*data_ordered)

            if plot:
                self.line_plot(name, [data_ordered])

        self.counts[name] += 1
        return log_time or force

    def log_array(
        self,
        item: LoggableArray,
    ) -> bool:
        name = item.table_name
        arr = item.array
        aux = item.aux
        plot = item.plot
        force = item.force
        cols = {str(j): item for (j, item) in enumerate(arr)}

        aux = aux if isinstance(aux, dict) else dict()
        loggable_item = Loggable(
            table_name=name,
            data=cols | aux,
            plot=plot,
            force=force,
            add_timestep=item.add_timestep,
        )
        return self.log(loggable_item)

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
