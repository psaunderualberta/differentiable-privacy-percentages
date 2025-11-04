import os

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray
import pandas as pd
import plotly.express as px
import tqdm
import equinox as eqx

from environments.dp import DP_RL_Params
from environments.dp import train_with_noise
from privacy.gdp_privacy import weights_to_sigma_schedule


file_location = os.path.abspath(os.path.dirname(__file__))


class Baseline:
    def __init__(self, env_params: DP_RL_Params, mu: float, num_reps: int = 8):
        self.mu: float = mu
        assert len(env_params.X.shape) >= 1
        self.p: float = env_params.dummy_batch.size / env_params.X.shape[0]
        self.env_params: DP_RL_Params = env_params
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
        return px.violin(
            df,
            x="type",
            y="loss",
            title="Final Loss Violin Plot",
            box=True,
        )

    def baseline_comparison_accuracy_plotter(self, df=None):
        df = self.combine_dataset(df)
        return px.violin(
            df, x="type", y="accuracy", title="Accuracy Violin Plot", box=True
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
        sigmas: Array,
        name: str,
        key: PRNGKeyArray,
        with_progress_bar: bool = True,
    ) -> pd.DataFrame:
        df = pd.DataFrame(columns=self.columns)

        iterator = range(self.num_repetitions)
        if with_progress_bar:
            iterator = tqdm.tqdm(iterator, desc=name, total=self.num_repetitions)

        for _ in iterator:
            key, _key = jr.split(key)
            k1, k2, k3 = jr.split(_key, 3)
            _, _, losses, accuracies = train_with_noise(
                sigmas, self.env_params, k1, k2, k3
            )
            df.loc[len(df)] = {  # type: ignore
                "type": name,
                "step": 0,  # only recording one step for these
                "loss": losses[-1],
                "accuracy": accuracies[-1],
                "losses": losses,
                "accuracies": accuracies,
                "actions": sigmas,
            }

        # Create a copy of baseline data, then another to be modified
        return df

    def generate_baseline_data(
        self, key: PRNGKeyArray, with_progress_bar: bool = True
    ) -> pd.DataFrame:
        T = self.env_params.max_steps_in_episode

        # Uniform schedule
        weights = jnp.ones(
            (
                1,
                T,
            ),
            dtype=jnp.float32,
        )
        sigmas = weights_to_sigma_schedule(weights, self.mu, self.p, T).squeeze()
        sigma = float(sigmas[0])
        name = f"Constant Noise ({round(sigma, 2)})"

        df = self.generate_schedule_data(
            sigmas, name, key, with_progress_bar=with_progress_bar
        )

        self.original_df = df
        self.df = self.original_df.copy()

        return self.df.copy()
