import inspect
import os
import pathlib
from collections.abc import Callable
from typing import Any, cast

import jax.random as jr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tqdm
from jaxtyping import Array, PRNGKeyArray

import wandb
from environments.dp import (
    DPTrainingParams,
    train_with_noise,
    train_with_stateful_noise,
)
from policy.base_schedules.constant import ConstantSchedule
from policy.schedules.abstract import AbstractNoiseAndClipSchedule
from policy.schedules.dynamic_dpsgd import DynamicDPSGDSchedule
from policy.schedules.sigma_and_clip import SigmaAndClipSchedule
from policy.stateful_schedules.abstract import (
    AbstractStatefulNoiseAndClipSchedule,
)
from policy.stateful_schedules.median_gradient import (
    StatefulMedianGradientNoiseAndClipSchedule,
)
from privacy.gdp_privacy import GDPPrivacyParameters
from util.aggregators import multi_line_plotter
from util.checkpointing import _ckpt_dir
from util.logger import WandbTableLogger

file_location = os.path.abspath(os.path.dirname(__file__))


def _baseline_path(run_id: str) -> pathlib.Path:
    return _ckpt_dir(run_id) / "baseline_data.pkl"


def _baseline_artifact_name(run_id: str) -> str:
    return f"baseline-{run_id}"


class Baseline:
    def __init__(
        self,
        env_params: DPTrainingParams,
        privacy_params: GDPPrivacyParameters,
        schedule_data_generation_key: PRNGKeyArray,
        num_reps: int = 8,
    ):
        self.env_params = env_params
        self.privacy_params = privacy_params
        self.schedule_data_generation_key = schedule_data_generation_key
        self.num_repetitions: int = num_reps
        self.columns: list[str] = [
            "type",
            "step",
            "loss",
            "accuracy",
            "losses",
            "accuracies",
        ]

    def save(self, run_id: str, run: Any) -> None:
        """Pickle the baseline DataFrame locally and upload as a W&B artifact.

        If ``best_dynamic_schedule`` is available (i.e. ``generate_baseline_data``
        has been called), also saves ``sigmas.npy`` / ``clips.npy`` alongside the
        pickle so that ``dp_psac_ref`` can consume the schedule without importing
        any ``src/`` modules.
        """
        path = _baseline_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.original_df.to_pickle(str(path))

        artifact = wandb.Artifact(
            name=_baseline_artifact_name(run_id),
            type="baseline",
            metadata={"run_id": run_id},
        )
        artifact.add_file(str(path))

        if hasattr(self, "best_dynamic_schedule"):
            sigmas_path = path.parent / "dynamic" / "sigmas.npy"
            clips_path = path.parent / "dynamic" / "clips.npy"
            sigmas_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(sigmas_path, np.asarray(self.best_dynamic_schedule.get_private_sigmas()))
            np.save(clips_path, np.asarray(self.best_dynamic_schedule.get_private_clips()))
            artifact.add_file(str(sigmas_path))
            artifact.add_file(str(clips_path))

        run.log_artifact(artifact, aliases=["latest"])
        print(f"Baseline data saved → {path}")

    def restore_from_cache(
        self,
        run_id: str,
        entity: str | None,
        project: str | None,
    ) -> bool:
        """Populate ``original_df`` / ``df`` from local disk or a W&B artifact.

        Returns ``True`` on success, ``False`` if neither source is available.
        """
        path = _baseline_path(run_id)
        if path.exists():
            print(f"Loading baseline data from {path}")
            df = pd.read_pickle(str(path))
            self.original_df = df
            self.df = df.copy()
            return True

        if entity is None or project is None:
            return False

        artifact_path = f"{entity}/{project}/{_baseline_artifact_name(run_id)}:latest"
        print(f"Attempting to download baseline artifact: {artifact_path}")
        try:
            artifact = wandb.Api().artifact(artifact_path)
            local_dir = pathlib.Path(artifact.download())
            pkl_files = list(local_dir.glob("*.pkl"))
            if not pkl_files:
                return False
            df = pd.read_pickle(str(pkl_files[0]))
            self.original_df = df
            self.df = df.copy()
            return True
        except Exception as e:
            print(f"Warning: could not load baseline artifact {artifact_path}: {e}")
            return False

    def delete_non_baseline_data(self):
        self.df = self.original_df.copy()
        return

    def combine_dataset(
        self,
        df: pd.DataFrame | None,
        schedule_name: str = "Learned Schedule",
    ) -> pd.DataFrame:
        if df is None:
            return self.df

        df = cast(pd.DataFrame, df[df["step"] == df["step"].max()].copy())
        df["type"] = schedule_name

        self.df = pd.concat([self.df, df], axis=0).reset_index(
            drop=True,
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
        key: PRNGKeyArray | None = None,
        with_progress_bar: bool = True,
        iterations: int = -1,
    ) -> pd.DataFrame:
        df = pd.DataFrame(columns=self.columns)  # type: ignore[arg-type]

        if key is None:
            key = self.schedule_data_generation_key

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
                    schedule,
                    self.env_params,
                    mb_key,
                    init_key,
                    noise_key,
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

    def baseline_sweep(
        self,
        key: PRNGKeyArray,
        params: list[Callable[[PRNGKeyArray], Array]],
        name: str,
        schedule_class: type[AbstractNoiseAndClipSchedule]
        | type[AbstractStatefulNoiseAndClipSchedule],
        num_runs_in_sweep: int = 30,
        with_progress_bar: bool = True,
    ) -> tuple[pd.DataFrame, AbstractNoiseAndClipSchedule | AbstractStatefulNoiseAndClipSchedule]:
        num_params = len(params)
        best_params = []
        best_run_accuracy = 0
        for _ in tqdm.tqdm(range(num_runs_in_sweep), desc=f"Sweep: {name}"):
            run_params = []
            key, _key = jr.split(key)
            param_keys = jr.split(_key, num_params)
            for param_fun, param_key in zip(params, param_keys):
                run_params.append(param_fun(param_key))

            schedule = schedule_class(*run_params)
            df = self.generate_schedule_data(schedule, name, with_progress_bar=False, iterations=10)

            run_accuracy = df["accuracy"].mean()
            if df["accuracy"].mean() > best_run_accuracy:
                best_run_accuracy = run_accuracy
                best_params = run_params

        # Print sweep results w/ argument names
        class_params = inspect.signature(schedule_class).parameters
        print(f"Best Accuracy for {schedule_class.__name__}: {best_run_accuracy:0.4f}")
        print(f"Best Parameters for {schedule_class.__name__}:")
        for param_name, param in zip(class_params, best_params):
            print(f"\t{param_name} = {param}")

        schedule = schedule_class(*best_params)

        return self.generate_schedule_data(
            schedule, name, key, with_progress_bar=with_progress_bar
        ), schedule

    def log_comparison(
        self,
        schedule: "AbstractNoiseAndClipSchedule | AbstractStatefulNoiseAndClipSchedule",
        eval_key: PRNGKeyArray,
        logger: WandbTableLogger,
        label: str = "Learned Schedule",
    ) -> None:
        """Log the final baseline comparison, generating baseline data if needed.

        If ``generate_baseline_data`` was already called during training,
        discards any mid-training learned-schedule rows before re-evaluating.
        Otherwise generates fresh baseline data first.
        """
        if not hasattr(self, "original_df"):
            self.generate_baseline_data(eval_key)
        else:
            self.delete_non_baseline_data()

        eval_df = self.generate_schedule_data(schedule, label)
        logger.log_figure(
            "Baseline vs. Losses", self.baseline_comparison_final_loss_plotter(eval_df)
        )
        logger.log_figure(
            "Baseline vs. Accuracy", self.baseline_comparison_accuracy_plotter(eval_df)
        )
        fig_sigmas, fig_clips = self.plot_sigma_clip_schedules()
        logger.log_figure("Baseline Sigma Schedule", fig_sigmas)
        logger.log_figure("Baseline Clip Schedule", fig_clips)

    def plot_sigma_clip_schedules(self) -> tuple[go.Figure, go.Figure]:
        if not hasattr(self, "best_dynamic_schedule"):
            raise RuntimeError("Call generate_baseline_data first.")
        schedule = self.best_dynamic_schedule
        T = int(schedule.privacy_params.T)
        name = "Dynamic-DPSGD"

        sigmas = [float(v) for v in schedule.get_private_sigmas()]
        sigma_df = pd.DataFrame([{"type": name, **{i: sigmas[i] for i in range(T)}}])

        clips = [float(v) for v in schedule.get_private_clips()]
        clip_df = pd.DataFrame([{"type": name, **{i: clips[i] for i in range(T)}}])

        return (
            multi_line_plotter(sigma_df, col_name="sigma", color_indicator="type"),
            multi_line_plotter(clip_df, col_name="clip", color_indicator="type"),
        )

    def _constant_schedule_sweep(
        self,
        key: PRNGKeyArray,
        name: str = "Constant σ/clip",  # noqa: RUF001
        num_runs_in_sweep: int = 30,
        with_progress_bar: bool = True,
    ) -> pd.DataFrame:
        """Sweep over constant σ/clip values and return data for the best setting.

        `project()` rescales σ to satisfy the privacy budget while keeping the
        sampled clip value fixed, so the sweep effectively explores different
        constant clip thresholds.
        """
        T = int(self.privacy_params.T)
        best_sigma: Array | None = None
        best_clip: Array | None = None
        best_run_accuracy = 0.0

        for _ in tqdm.tqdm(range(num_runs_in_sweep), desc=f"Sweep: {name}"):
            key, sigma_key, clip_key = jr.split(key, 3)
            sigma_val = jr.uniform(sigma_key, shape=(), minval=0.1, maxval=5.0)
            clip_val = jr.uniform(clip_key, shape=(), minval=0.1, maxval=5.0)

            schedule = SigmaAndClipSchedule(
                ConstantSchedule(sigma_val, T),
                ConstantSchedule(clip_val, T),
                self.privacy_params,
            ).project()

            # Update to values post-projection
            sigma_val = schedule.get_private_sigmas().mean()
            clip_val = schedule.get_private_clips().mean()

            df = self.generate_schedule_data(schedule, name, with_progress_bar=False, iterations=10)

            run_accuracy = float(df["accuracy"].mean())
            if run_accuracy > best_run_accuracy:
                best_run_accuracy = run_accuracy
                best_sigma = sigma_val
                best_clip = clip_val

        print(f"Best Accuracy for {name}: {best_run_accuracy:0.4f}")
        print(f"Best Parameters for {name}:")
        print(f"\tsigma = {best_sigma}")
        print(f"\tclip  = {best_clip}")

        best_schedule = SigmaAndClipSchedule(
            ConstantSchedule(best_sigma, T),
            ConstantSchedule(best_clip, T),
            self.privacy_params,
        )

        return self.generate_schedule_data(best_schedule, name, with_progress_bar=with_progress_bar)

    def generate_baseline_data(
        self,
        key: PRNGKeyArray,
        with_progress_bar: bool = True,
    ) -> pd.DataFrame:
        name = "Clip to Median Gradient Norm"
        params = [
            lambda key: jr.uniform(key, shape=(), minval=0.01, maxval=5.0),  # c_0
            lambda key: jr.uniform(key, shape=(), minval=0.01, maxval=1.0),  # eta_C
            lambda _: self.privacy_params,  # privacy_params
        ]

        key, sweep_key = jr.split(key)
        median_df, _ = self.baseline_sweep(
            sweep_key,
            params,
            name,
            StatefulMedianGradientNoiseAndClipSchedule,
            with_progress_bar=with_progress_bar,
        )

        name = "Dynamic-DPSGD"
        params = [
            lambda key: jr.uniform(key, shape=(), minval=0.5, maxval=5.0),  # rho_mu
            lambda key: jr.uniform(key, shape=(), minval=0.5, maxval=5.0),  # rho_c
            lambda key: jr.uniform(key, shape=(), minval=0.5, maxval=5.0),  # c_0
            lambda _: self.privacy_params,  # privacy_params
        ]

        key, sweep_key = jr.split(key)
        dynamic_df, dynamic_schedule = self.baseline_sweep(
            sweep_key,
            params,
            name,
            DynamicDPSGDSchedule,
            with_progress_bar=with_progress_bar,
        )
        self.best_dynamic_schedule: DynamicDPSGDSchedule = cast(
            DynamicDPSGDSchedule, dynamic_schedule
        )

        # c_0 = jnp.asarray(2.5)
        # rho_mu = jnp.asarray(2)
        # rho_c = jnp.asarray(2)
        # schedule = DynamicDPSGDSchedule(rho_mu, rho_c, c_0, self.privacy_params)

        key, sweep_key = jr.split(key)
        constant_df = self._constant_schedule_sweep(
            sweep_key,
            with_progress_bar=with_progress_bar,
        )

        self.original_df = pd.concat([median_df, dynamic_df, constant_df], axis=0)
        self.df = self.original_df.copy()

        return self.df.copy()
