import os

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
import plotly.express as px
import tqdm

from environments.dp import DP_RL, DP_RL_Params
from environments.dp import train_with_noise
from privacy.gdp_privacy import vec_to_mu_schedule

file_location = os.path.abspath(os.path.dirname(__file__))


class Baseline:
    def __init__(self, env_params: DP_RL_Params, mu: float, num_reps: int = 8):
        self.mu = mu
        assert len(env_params.X.shape) >= 1
        self.p = env_params.dummy_batch.size / env_params.X.shape[0]
        self.env_params = env_params
        self.num_repetitions = num_reps
        self.columns = [
            "type",
            "step",
            "loss",
            "accuracy",
            "losses",
            "accuracies",
            "actions"
        ]

    def delete_non_baseline_data(self):
        self.df = self.original_df.copy()
        return

    def combine_dataset(self, df, schedule_name="Learned Policy"):
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
            df, x="type", y="loss", title="Final Loss Box Plot", log_y=True
        )

    def baseline_comparison_accuracy_plotter(self, df=None):
        df = self.combine_dataset(df)
        return px.box(df, x="type", y="accuracy", title="Accuracy Box Plot")

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

    def generate_baseline_data(self, with_progress_bar = True):
        T = self.env_params.max_steps_in_episode
        weights = jnp.ones((1, T,)) / T
        assert jnp.isclose(jnp.sum(weights), 1.0, 1e-5), f"{jnp.sum(weights)}, {T}"
        sigmas = vec_to_mu_schedule(weights, self.mu, self.p, T).squeeze()
        self.sigma = float(sigmas[0])  # type: ignore

        df = pd.DataFrame(columns=self.columns)

        iterator = range(self.num_repetitions)
        if with_progress_bar:
            iterator = tqdm.tqdm(
                iterator,
                desc="Evaluating Baselines",
                total=self.num_repetitions
            )

        key = jr.PRNGKey(0)
        for _ in iterator:
            key, _key = jr.split(key)
            _, losses, accuracies = train_with_noise(sigmas, self.env_params, _key)
            df.loc[len(df)] = {  # type: ignore
                "type": f"Constant Noise ({round(self.sigma, 2)})",
                "step": 0,  # only recording one step for these
                "loss": losses[-1],
                "accuracy": accuracies[-1],
                "losses": losses,
                "accuracies": accuracies,
                "actions": sigmas,
            }

        # Create a copy of baseline data, then another to be modified
        self.original_df = df
        self.df = self.original_df.copy()

        return
