import csv
import tempfile
from collections.abc import Mapping

import equinox as eqx
import jax.numpy as jnp
import pandas as pd
from jaxtyping import Array

import wandb
from conf.singleton_conf import SingletonConfig
from util.aggregators import multi_line_plotter
from util.util import str_to_jnp_array


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
    freq: int
    add_step_column: bool

    def __init__(
        self,
        table_name: str,
        cols: list[str],
        freq: int | None = None,
        add_step_column: bool = True,
    ):
        """Create a logging schema for a named W&B table.

        Args:
            table_name: W&B table name used as a key throughout the logger.
            cols: Column names (not including the automatic step column).
            freq: Log every `freq` calls. Defaults to the config's plotting_interval.
            add_step_column: If True, a 'step' column is prepended automatically.
        """
        if freq is None:
            freq = SingletonConfig.get_sweep_config_instance().plotting_interval

        self.table_name = table_name
        self.cols = cols
        self.freq = freq
        self.add_step_column = add_step_column


class WandbTableLogger(eqx.Module):
    files: dict[str, tempfile._TemporaryFileWrapper]
    writers: dict[str, csv.DictWriter]
    cols: dict[str, list[str]]
    freqs: dict[str, int]
    counts: dict[str, int]

    def __init__(self):
        """Initialise the logger with empty per-table file, writer, and counter dicts."""
        self.files = {}
        self.writers = {}
        self.cols = {}
        self.freqs = {}
        self.counts = {}

    def add_schema(self, schema: LoggingSchema):
        """Register a new table schema, creating the backing temp CSV file and DictWriter."""
        assert schema.table_name not in self.writers, (
            f"Table name '{schema.table_name}' already exists in logger."
        )
        name = schema.table_name
        cols = ["step", *schema.cols] if schema.add_step_column else schema.cols
        file = tempfile.NamedTemporaryFile(newline="", suffix=".csv", mode="w")
        self.files[name] = file
        self.writers[name] = csv.DictWriter(file, fieldnames=cols)
        self.writers[name].writeheader()
        self.cols[name] = cols
        self.freqs[name] = schema.freq
        self.counts[name] = 0

    def log(
        self,
        item: Loggable | LoggableArray,
    ) -> bool:
        """Log an item to its table, respecting the configured frequency.

        Returns:
            True if the item was actually written this call, False if skipped.
        """
        # Convert to loggable
        if isinstance(item, LoggableArray):
            return self.log_array(item)

        name = item.table_name
        data = item.data
        force = item.force
        plot = item.plot
        assert name in self.writers, (
            f"Name '{name}' not found in set of writers: {list(self.writers.keys())}"
        )
        log_time = self.counts[name] % self.freqs[name] == 0
        if log_time or force:
            writer = self.writers[name]

            if item.add_timestep:
                data["step"] = self.counts[name]

            data_ordered = {col: jnp.asarray(data[col]).tolist() for col in self.cols[name]}
            writer.writerow(data_ordered)
            self.files[name].flush()

            if plot:
                self.line_plot(name, data_ordered)

        self.counts[name] += 1
        return log_time or force

    def log_array(
        self,
        item: LoggableArray,
    ) -> bool:
        """Convert a LoggableArray to a Loggable dict and delegate to `log`."""
        name = item.table_name
        arr = item.array
        aux = item.aux
        plot = item.plot
        force = item.force
        cols = {str(j): item for (j, item) in enumerate(arr)}

        aux = aux if isinstance(aux, dict) else {}
        loggable_item = Loggable(
            table_name=name,
            data=cols | aux,
            plot=plot,
            force=force,
            add_timestep=item.add_timestep,
        )
        return self.log(loggable_item)

    def commit(self, metrics: Mapping[str, int] | None = None):
        """Flush additional scalar metrics to W&B via wandb.log."""
        if metrics is None:
            metrics = {}
        assert len(self.writers.keys() & metrics) == 0, "Name Overlap"

        # For type checker
        assert isinstance(metrics, dict)
        wandb.log(metrics)

    def finish(self):
        """Upload all accumulated CSV tables to W&B as immutable W&B Tables and close temp files."""
        final_tables_pd = {
            tablename: pd.read_csv(filename.name) for tablename, filename in self.files.items()
        }
        final_tables = {
            name: wandb.Table(dataframe=table, log_mode="IMMUTABLE")
            for (name, table) in final_tables_pd.items()
        }

        wandb.log(final_tables)

        for file in self.files.values():
            file.close()

    def line_plot(self, table_name: str, data: dict[str, list] | None = None):
        """Render and log a multi-line plot for a table to W&B.

        Args:
            table_name: Name of the registered table to plot.
            data: Optional single-row dict to plot instead of reading the full CSV.
        """
        if data is not None:
            cols = self.cols[table_name]
            data_ordered = [data[col] for col in cols]
            df = pd.DataFrame(columns=cols, data=[data_ordered])  # type: ignore[arg-type]
        else:
            df = pd.read_csv(self.files[table_name].name)

        # multi-line, but only plotting one line
        fig = multi_line_plotter(df, table_name)
        wandb.log({f"{table_name}-plot": fig})

    def bulk_line_plots(self, table_name: str):
        """Parse the last CSV row as a 2-D array and log one line per inner row to W&B."""
        dataframe = pd.read_csv(self.files[table_name].name).iloc[-1]
        data = dataframe.drop("step").values

        # data is a list with a string array, such as ['[1, 2, 3]']
        jnp_data = str_to_jnp_array(data[0], with_brackets=False)

        # implicitly checks for square-ness
        bulk_lines = jnp.asarray(jnp_data)
        num_cols = bulk_lines.shape[-1]

        pd_compatible_data = []
        for i, line in enumerate(bulk_lines):
            pd_compatible_data.append([i, *[num.item() for num in line]])

        cols = ["run no.", *map(str, range(num_cols))]
        df = pd.DataFrame(columns=cols, data=pd_compatible_data)  # type: ignore[arg-type]
        print(df)

        # multi-line, but only plotting one line
        fig = multi_line_plotter(df, table_name, color_indicator="run no.")
        wandb.log({f"{table_name}-plot": fig})
