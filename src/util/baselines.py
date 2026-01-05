import os

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
import plotly.express as px
import tqdm
from jaxtyping import Array, PRNGKeyArray

from environments.dp import (
    DP_RL_Params,
    train_with_noise,
    train_with_stateful_noise,
)
from privacy.gdp_privacy import GDPPrivacyParameters
from privacy.schedules import AbstractNoiseAndClipSchedule, PolicyAndClipSchedule
from privacy.stateful_schedules import (
    AbstractScheduleState,
    AbstractStatefulNoiseAndClipSchedule,
    StatefulMedianGradientNoiseAndClipSchedule,
)

file_location = os.path.abspath(os.path.dirname(__file__))


class Baseline:
    def __init__(
        self,
        env_params: DP_RL_Params,
        privacy_params: GDPPrivacyParameters,
        num_reps: int = 8,
    ):
        self.env_params = env_params
        self.privacy_params = privacy_params
        self.num_repetitions: int = num_reps
        self.columns: list[str] = [
            "type",
            "step",
            "loss",
            "accuracy",
            "losses",
            "accuracies",
        ]

    def delete_non_baseline_data(self):
        self.df = self.original_df.copy()
        return

    def combine_dataset(
        self, df: pd.DataFrame | None, schedule_name: str = "Learned Policy"
    ) -> pd.DataFrame:
        if df is None:
            return self.df

        df = df[df["step"] == df["step"].max()].copy()
        df["type"] = schedule_name

        self.df = pd.concat([self.df, df], axis=0).reset_index(
            drop=True
        )  # concatenating along rows
        return self.df

    def baseline_comparison_final_loss_plotter(self, df=None):
        df = self.combine_dataset(df)
        return px.box(
            df,
            x="type",
            y="loss",
            title="Final Loss Plot",
            points="all",
            notched=True,
        )

    def baseline_comparison_accuracy_plotter(self, df=None):
        df = self.combine_dataset(df)
        return px.box(
            df,
            x="type",
            y="accuracy",
            title="Accuracy Plot",
            points="all",
            notched=True,
        )

    def create_baseline_figures(self, save_figs=False):
        figs = [
            self.baseline_comparison_final_loss_plotter(),
            self.baseline_comparison_accuracy_plotter(),
        ]

        fig_names = [
            "final_loss",
            "accuracy",
        ]

        if save_figs:
            for fig, fig_name in zip(figs, fig_names):
                self.save_fig(fig, fig_name)

        return figs, fig_names

    def save_fig(self, fig, name):
        html_directory = os.path.join(".", "plots", name + ".html")
        fig.write_html(html_directory)
        pdf_directory = os.path.join(".", "plots", name + ".pdf")
        fig.write_image(pdf_directory)

    def generate_schedule_data(
        self,
        schedule: AbstractNoiseAndClipSchedule | AbstractStatefulNoiseAndClipSchedule,
        name: str,
        key: PRNGKeyArray,
        with_progress_bar: bool = True,
        iterations: int = -1,
    ) -> pd.DataFrame:
        df = pd.DataFrame(columns=self.columns)

        if iterations < 0:
            iterations = self.num_repetitions
        iterator = range(iterations)
        if with_progress_bar:
            iterator = tqdm.tqdm(iterator, desc=name, total=iterations)

        for _ in iterator:
            key, mb_key, init_key = jr.split(key, 3)
            key, noise_key = jr.split(key)
            if isinstance(schedule, AbstractNoiseAndClipSchedule):
                _, val_loss, losses, accuracies, val_acc = train_with_noise(
                    schedule, self.env_params, mb_key, init_key, noise_key
                )
            else:
                _, val_loss, losses, accuracies, val_acc = train_with_stateful_noise(
                    schedule,
                    self.env_params,
                    mb_key,
                    init_key,
                    noise_key,
                )
            df.loc[len(df)] = {  # type: ignore
                "type": name,
                "step": 0,  # only recording one step for these
                "loss": val_loss,
                "accuracy": val_acc,
                "losses": losses,
                "accuracies": accuracies,
            }

        # Create a copy of baseline data, then another to be modified
        return df

    def generate_baseline_data(
        self, key: PRNGKeyArray, with_progress_bar: bool = True
    ) -> pd.DataFrame:
        # Uniform schedule
        c_0 = jnp.asarray(0.1)
        eta_C = jnp.asarray(0.2)
        schedule = StatefulMedianGradientNoiseAndClipSchedule(
            c_0, eta_C, self.privacy_params
        )

        name = "Clip to Median Gradient Norm"

        df = self.generate_schedule_data(
            schedule, name, key, with_progress_bar=with_progress_bar
        )

        self.original_df = df
        self.df = self.original_df.copy()

        return self.df.copy()
