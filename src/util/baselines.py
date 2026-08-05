import inspect
import os
import pathlib
from collections.abc import Callable
from typing import Any, cast

import jax.random as jr
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tqdm
import wandb
from jaxtyping import PRNGKeyArray

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
from util.util import jnp2np2jnp

file_location = os.path.abspath(os.path.dirname(__file__))


_BASELINE_CACHE_VERSION = "v2"

"""Bumped whenever the baselines' mechanism changes, because ``restore_from_cache``
keys only on ``run_id`` and would otherwise fail open and restore stale numbers.
v2 = the adaptive-clip baseline privatises its count release (ADR 0013)."""


# The native references, in the order generate_baseline_data has always swept them.
# The order is load-bearing: reference_sweep_keys reproduces the original sequential
# key split, so changing it would silently change every reference's results.
REFERENCES = (
    "Adaptive Clip (Andrew et al.)",
    "Dynamic-DPSGD",
    "Constant σ/clip",
)


NUM_SWEEP_CANDIDATES = 20
"""Candidates in a reference's random search. Kept at 20 through ADR 0019's split:
shrinking it is the cheap alternative that would make the references a straw man."""

SWEEP_SCORING_ITERATIONS = 3
"""Inner trainings per candidate when scoring. Was 10; ADR 0019 cut it so one
(reference x target x candidate) SLURM task fits ~1.3h. Selection is noisier but
unbiased, and the winner is re-evaluated cleanly for the reported number."""

_SWEEP_SCORING_STREAM = 7919
"""Fold-in constant separating the candidate-scoring draws from the final evaluation's,
so a winner's reported accuracy is not the same draw that selected it."""


def describe_schedule(schedule) -> dict[str, object]:
    """The drawn parameters of a candidate, for the sweep's printout.

    These schedules store their constructor arguments as same-named fields, so the
    signature recovers what was drawn. ``privacy_params`` is dropped (shared by every
    candidate) and a nested base schedule is reduced to its mean value, which is what
    a constant σ/clip candidate is actually characterised by.
    """
    from policy.base_schedules.abstract import AbstractSchedule

    def summarise(value):
        if isinstance(value, AbstractSchedule):
            return float(np.mean(np.asarray(value.get_valid_schedule())))
        return value

    names = (n for n in inspect.signature(type(schedule)).parameters if n != "privacy_params")
    return {n: summarise(getattr(schedule, n)) for n in names if hasattr(schedule, n)}


def reference_sweep_keys(key: PRNGKeyArray) -> dict[str, PRNGKeyArray]:
    """The per-reference sweep key, as a pure function of the master key.

    Reproduces ``generate_baseline_data``'s original running-split so a reference
    swept in isolation (one SLURM job per reference, per the transfer launcher)
    draws exactly the parameters it would have drawn in the combined sweep — the
    subset stays comparable to previously generated baselines.
    """
    keys = {}
    for reference in REFERENCES:
        key, sweep_key = jr.split(key)
        keys[reference] = sweep_key
    return keys


def _baseline_path(run_id: str) -> pathlib.Path:
    return _ckpt_dir(run_id) / f"baseline_data_{_BASELINE_CACHE_VERSION}.pkl"


def _baseline_artifact_name(run_id: str) -> str:
    return f"baseline-{_BASELINE_CACHE_VERSION}-{run_id}"


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
            dynamic_dir = path.parent / "dynamic"
            checkpointer = ocp.StandardCheckpointer()
            checkpointer.save(dynamic_dir, self.best_dynamic_schedule)
            # StandardCheckpointer uses async I/O internally; wait here so the step
            # directory exists on disk before we try to add it to the W&B artifact.
            checkpointer.wait_until_finished()
            artifact.add_dir(str(dynamic_dir))

            # Save human-readable schedules alongside the Orbax checkpoint so that
            # dp_psac_ref/run.py can consume them without importing src/.
            sigmas_path = dynamic_dir / "sigmas.npy"
            clips_path = dynamic_dir / "clips.npy"
            sigmas_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(sigmas_path, np.asarray(self.best_dynamic_schedule.get_private_noise_scales()))
            np.save(clips_path, np.asarray(self.best_dynamic_schedule.get_private_clips()))
            artifact.add_file(str(sigmas_path))
            artifact.add_file(str(clips_path))

            print(f"DynamicDPSGD Baseline data saved → {dynamic_dir}")

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
        """
        path = _baseline_path(run_id)
        if not path.exists():
            path = wandb_artifact_path

        """

        pkl_path = _baseline_path(run_id)
        local_dir = pkl_path.parent
        if not pkl_path.exists():
            if entity is None or project is None:
                return False

            artifact_path = f"{entity}/{project}/{_baseline_artifact_name(run_id)}:latest"
            print(f"Attempting to download baseline artifact: {artifact_path}..")
            try:
                artifact = wandb.Api().artifact(artifact_path)
                local_dir = pathlib.Path(artifact.download())
                pkl_files = list(local_dir.glob("*.pkl"))
                if not pkl_files:
                    return False
                pkl_path = pathlib.Path(str(pkl_files[0]))
                print("Downloaded baseline artifact")
            except Exception as e:
                print(f"Warning: could not load baseline artifact {artifact_path}: {e}")
                return False

        print(f"Loading baseline data from {pkl_path}...")
        df = pd.read_pickle(str(pkl_path))
        self.original_df = df
        self.df = df.copy()
        print("Baseline data loaded")

        try:
            print("Loading DynamicDPSGD schedule...")
            checkpointer = ocp.StandardCheckpointer()
            self.best_dynamic_schedule = checkpointer.restore(
                # values save for privacy_params don't matter, will be overwritten by checkpointer
                local_dir,
                target=DynamicDPSGDSchedule(1.0, 1.0, 1.0, self.privacy_params),
            )
            self.best_dynamic_schedule = jnp2np2jnp(self.best_dynamic_schedule)
            print("DynamicDPSGD loaded")
        except Exception as e:
            print(f"Warning: could not load dynamicDPSGD baseline: {e}")

        return True

    def delete_non_baseline_data(self):
        self.df = self.original_df.copy()
        return

    def combine_dataset(
        self,
        df: pd.DataFrame | None,
        schedule_name: str = "Learned Schedule",
    ) -> pd.DataFrame:
        if df is None:
            return cast(pd.DataFrame, self.df)

        df = cast(pd.DataFrame, df[df["step"] == df["step"].max()].copy())
        df["type"] = schedule_name

        self.df = pd.concat([self.df, df], axis=0).reset_index(
            drop=True,
        )  # concatenating along rows
        return cast(pd.DataFrame, self.df)

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
        test_data: bool = False,
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
                _, statistics = train_with_noise(
                    schedule,
                    self.env_params,
                    mb_key,
                    init_key,
                    noise_key,
                )
            else:
                _, statistics = train_with_stateful_noise(
                    schedule,
                    self.env_params,
                    mb_key,
                    init_key,
                    noise_key,
                )
            df.loc[len(df)] = {  # type: ignore
                "type": name,
                "step": 0,  # only recording one step for these
                "loss": statistics.test_loss if test_data else statistics.val_loss,
                "accuracy": statistics.test_accuracy if test_data else statistics.val_accuracy,
                "losses": statistics.losses,
                "accuracies": statistics.accuracies,
            }

        # Create a copy of baseline data, then another to be modified
        return df

    def candidate_schedules(
        self,
        reference: str,
        key: PRNGKeyArray,
        num_candidates: int = NUM_SWEEP_CANDIDATES,
    ) -> list[AbstractNoiseAndClipSchedule | AbstractStatefulNoiseAndClipSchedule]:
        """The schedules ``reference``'s random search evaluates, in sweep order.

        A pure function of ``(reference, key, index)``: candidate *i* is the same
        schedule whether it is reached by enumerating twenty of them in one process
        or by asking for the first eight in another. That is what lets ADR 0019 run
        one SLURM task per candidate without changing the search — the full
        20-candidate space is preserved, only its packaging changes.

        The draw order per reference is the one the monolithic sweep has always
        used, so the search space itself is unchanged.
        """
        schedules = []
        if reference == "Constant σ/clip":
            # project() rescales sigma to satisfy the budget while holding the sampled
            # clip fixed, so the search effectively explores constant clip thresholds.
            T = int(self.privacy_params.T)
            for _ in range(num_candidates):
                key, sigma_key, clip_key = jr.split(key, 3)
                schedules.append(
                    SigmaAndClipSchedule(
                        ConstantSchedule(
                            jr.uniform(sigma_key, shape=(), minval=0.1, maxval=5.0), T
                        ),
                        ConstantSchedule(jr.uniform(clip_key, shape=(), minval=0.1, maxval=5.0), T),
                        self.privacy_params,
                    ).project()
                )
            return schedules

        params, schedule_class = self._search_space(reference)
        for _ in range(num_candidates):
            key, _key = jr.split(key)
            param_keys = jr.split(_key, len(params))
            schedules.append(
                schedule_class(*(fun(param_key) for fun, param_key in zip(params, param_keys)))
            )
        return schedules

    def _search_space(
        self, reference: str
    ) -> tuple[
        list[Callable[[PRNGKeyArray], Any]],
        type[AbstractNoiseAndClipSchedule] | type[AbstractStatefulNoiseAndClipSchedule],
    ]:
        """One reference's per-parameter samplers and the schedule they construct.

        Order is load-bearing: the sampler list is zipped against ``jr.split``'s
        output, so reordering it would silently change every candidate drawn.
        """
        if reference == "Adaptive Clip (Andrew et al.)":
            return [
                lambda key: jr.uniform(key, shape=(), minval=0.01, maxval=5.0),  # c_0
                lambda key: jr.uniform(key, shape=(), minval=0.01, maxval=1.0),  # eta_C
                lambda _: self.privacy_params,  # privacy_params
            ], StatefulMedianGradientNoiseAndClipSchedule
        if reference == "Dynamic-DPSGD":
            return [
                lambda key: jr.uniform(key, shape=(), minval=0.5, maxval=5.0),  # rho_mu
                lambda key: jr.uniform(key, shape=(), minval=0.5, maxval=5.0),  # rho_c
                lambda key: jr.uniform(key, shape=(), minval=0.5, maxval=5.0),  # c_0
                lambda _: self.privacy_params,  # privacy_params
            ], DynamicDPSGDSchedule
        raise ValueError(f"unknown reference {reference!r}; expected one of {REFERENCES}")

    def score_candidate(
        self,
        schedule: AbstractNoiseAndClipSchedule | AbstractStatefulNoiseAndClipSchedule,
        name: str,
        iterations: int = SWEEP_SCORING_ITERATIONS,
    ) -> float:
        """A candidate's mean validation accuracy over ``iterations`` inner trainings.

        Every candidate is scored under the *same* key, so the comparison between
        them is a common-random-numbers one and a candidate cannot win on a lucky
        draw of initialisations. That key is deliberately **not** the one
        :meth:`evaluate_candidate` reports on: reusing it would make the winner's
        reported number the very draw that selected it (ADR 0019).
        """
        df = self.generate_schedule_data(
            schedule,
            name,
            key=jr.fold_in(self.schedule_data_generation_key, _SWEEP_SCORING_STREAM),
            with_progress_bar=False,
            iterations=iterations,
        )
        return float(df["accuracy"].mean())

    def evaluate_candidate(
        self,
        schedule: AbstractNoiseAndClipSchedule | AbstractStatefulNoiseAndClipSchedule,
        name: str,
        with_progress_bar: bool = True,
    ) -> pd.DataFrame:
        """The reported final evaluation of a chosen candidate, on the held-out split.

        Uses the Baseline's own generation key — the same one the curve and equation
        transfer producers evaluate their schedules under — so a reference cell and
        the transferred cells it is compared against see common random numbers.
        """
        return self.generate_schedule_data(
            schedule, name, with_progress_bar=with_progress_bar, test_data=True
        )

    def baseline_sweep(
        self,
        key: PRNGKeyArray,
        name: str,
        num_runs_in_sweep: int = NUM_SWEEP_CANDIDATES,
        with_progress_bar: bool = True,
        iterations: int = SWEEP_SCORING_ITERATIONS,
    ) -> tuple[pd.DataFrame, AbstractNoiseAndClipSchedule | AbstractStatefulNoiseAndClipSchedule]:
        """Random-search one reference's hyperparameters and evaluate the winner.

        The whole search in one process. ADR 0019 splits exactly this loop across
        SLURM tasks for transfer, which is why the candidate enumeration and the two
        evaluation halves are separate methods — the split path calls the same three.
        """
        schedules = self.candidate_schedules(name, key, num_runs_in_sweep)
        scores = [
            self.score_candidate(schedule, name, iterations)
            for schedule in tqdm.tqdm(
                schedules, desc=f"Sweep: {name}", disable=not with_progress_bar
            )
        ]
        best = schedules[int(np.argmax(scores))]

        print(f"Best Accuracy for {name}: {max(scores):0.4f}")
        print(f"Best Parameters for {name}:")
        for param_name, value in describe_schedule(best).items():
            print(f"\t{param_name} = {value}")

        return self.evaluate_candidate(best, name, with_progress_bar=with_progress_bar), best

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

        eval_df = self.generate_schedule_data(schedule, label, test_data=True)
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

        sigmas = [float(v) for v in schedule.get_private_noise_scales()]
        sigma_df = pd.DataFrame([{"type": name, **{i: sigmas[i] for i in range(T)}}])

        clips = [float(v) for v in schedule.get_private_clips()]
        clip_df = pd.DataFrame([{"type": name, **{i: clips[i] for i in range(T)}}])

        return (
            multi_line_plotter(sigma_df, col_name="sigma", color_indicator="type"),
            multi_line_plotter(clip_df, col_name="clip", color_indicator="type"),
        )

    def sweep_reference(
        self,
        reference: str,
        key: PRNGKeyArray,
        with_progress_bar: bool = True,
    ) -> pd.DataFrame:
        """Sweep one native reference and return its final-eval rows.

        Split out of ``generate_baseline_data`` so the transfer launcher can run
        each reference as its own SLURM job (they are the longest stage, and
        sharing a job would serialise three sweeps behind one wall clock). The
        ``key`` comes from :func:`reference_sweep_keys`, so a reference swept alone
        sees exactly the key it would have seen in the combined sweep.
        """
        if reference not in REFERENCES:
            raise ValueError(f"unknown reference {reference!r}; expected one of {REFERENCES}")

        df, best = self.baseline_sweep(key, reference, with_progress_bar=with_progress_bar)
        if reference == "Dynamic-DPSGD":
            self.best_dynamic_schedule: DynamicDPSGDSchedule = cast(DynamicDPSGDSchedule, best)
        return df

    def generate_baseline_data(
        self,
        key: PRNGKeyArray,
        with_progress_bar: bool = True,
        references: tuple[str, ...] = REFERENCES,
    ) -> pd.DataFrame:
        """Sweep the native references and return their concatenated final-eval rows.

        ``references`` narrows the sweep without perturbing it: each one is keyed by
        :func:`reference_sweep_keys`, so a subset produces byte-identical rows to the
        same references inside the full run.
        """
        keys = reference_sweep_keys(key)
        frames = [
            self.sweep_reference(reference, keys[reference], with_progress_bar=with_progress_bar)
            for reference in references
        ]

        self.original_df = pd.concat(frames, axis=0)
        self.df = self.original_df.copy()

        return self.df.copy()
