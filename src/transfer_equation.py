"""Equation-transfer producer (ADR 0008).

Evaluate the SR-distilled universal shape ``f(step_norm)`` on the target's step
grid, then feed the *identical* ``seat_on_budget`` + eval core as curve transfer.
Only the schedule shape differs: curve transfer resamples a raw length-T curve,
this evaluates a closed form.

The template's per-condition constants are indexed by discrete
``(dataset, eps, T, arch)`` and are **not** a function of eps/T, so the closed
form is undefined off-grid. Equation transfer therefore runs only at a target
``(eps, T)`` that exactly matches a trained condition, borrowing that condition's
constants; every condition present at that ``(eps, T)`` is transferred (read off,
not selected). Both sigma and clip come from their distilled equations.
"""

import dataclasses
from pathlib import Path

import numpy as np
import tyro
from jax import random as jr

from sr_category import CategoryMap

# A *condition* is (dataset, eps, T, arch) and carries no arm, but a synthesis is fitted
# over one arm's runs (ADR 0016), so the arm belongs to the fit — and, since both arms
# distil the same conditions, to the cell name (ADR 0021). Both live launcher-side so the
# skip filter predicts exactly the name written here.
from transfer_launch import condition_source_id, synthesis_arm
from util.transfer import SourcePolicy, TargetSpec


def equation_source(category: int, condition: dict, arm: str = "") -> SourcePolicy:
    """The ``SourcePolicy`` for a distilled condition transferred as an equation.

    A condition is not a single learned run, so its provenance IS the condition
    ``(dataset, eps, T, arch)``, tagged by an fs-safe id from
    ``transfer_launch.condition_source_id`` — shared with the SLURM launcher, whose
    skip filter must predict the cell filename this id becomes part of. ``delta``
    and ``p`` are NaN: a category map carries neither. ``arm`` comes from the
    synthesis rather than the condition (:func:`synthesis_arm`).
    """
    dataset, arch = condition["dataset"], condition["arch_label"]
    return SourcePolicy(
        run_id=condition_source_id(category, condition),
        dataset=str(dataset),
        eps=float(condition["eps"]),
        delta=float("nan"),
        T=int(condition["T"]),
        p=float("nan"),
        arch=str(arch),
        arm=arm,
    )


def evaluate_equation_shape(
    predictor, category: int, target_T: int, max_plausible: float | None = None
) -> np.ndarray:
    """Evaluate the selected distilled shape on the target step grid.

    ``f`` is closed-form over ``step_norm``, so the shape is *evaluated* — one value
    per target step — rather than resampled from a fixed-length array. ``category``
    selects the condition whose per-condition constants are inlined into ``f``.

    The grid is ``inner_step / target_T``, the convention the equation was fitted
    over (compile_results_fetch.py), so the last step sits at ``(T-1)/T`` and not at
    1. Evaluating on ``linspace(0, 1, T)`` instead would stretch the shape by
    ``T/(T-1)`` and read every fitted feature off by up to one step.

    ``max_plausible`` bounds how far the shape may stray from the data it was fitted
    on (see :func:`sr_predict.plausible_bound`); ``None`` skips that criterion.
    """
    step_norm = np.arange(target_T, dtype=float) / target_T
    X = np.column_stack([step_norm, np.full(target_T, category)])
    with np.errstate(all="ignore"):
        shape = np.asarray(predictor.predict(X), dtype=float)
    _reject_unusable_shape(shape, category, step_norm, max_plausible)
    return shape


def _reject_unusable_shape(
    shape: np.ndarray, category: int, step_norm: np.ndarray, max_plausible: float | None = None
) -> None:
    """Refuse a shape that is not a schedule, naming the equation that produced it.

    Syntheses predating ADR 0025 could distil an equation with a pole *between* the
    points it was fitted on — f152229a's selected clip equation had one at
    step_norm ≈ 0.9917, which the full target grid lands 2.8e-4 away from and which
    took the shape to 1.3e15. Unchecked, that flows into ``seat_on_budget``, whose
    bisection then saturates at its bracket ceiling and reports only that it "did not
    bind the budget" — true, but it names neither the equation nor the category, and
    points at the seater rather than at what it was handed.

    All three criteria are needed, and the magnitude one is not optional: whether a
    pole is *hit* depends on the target T. At T=2000 that same equation lands 2.2e-4
    away from its pole and returns 2.9e13 — finite, positive, and ruinous.
    """
    bad = ~np.isfinite(shape)
    if bad.any():
        first = float(step_norm[np.argmax(bad)])
        raise ValueError(
            f"the distilled shape is non-finite at {int(bad.sum())} of {len(shape)} target "
            f"steps for category {category} (first at step_norm={first:.6g}) — the equation "
            "has a pole inside the step grid. Re-synthesise under ADR 0025 (log-space "
            "target + capped denominator complexity)."
        )
    if (shape <= 0.0).any():
        n = int((shape <= 0.0).sum())
        raise ValueError(
            f"the distilled shape is non-positive at {n} of {len(shape)} target steps for "
            f"category {category} (min {float(shape.min()):.3g}) — a σ of 0 has no privacy "
            "multiplier and a C of 0 kills the gradient. Re-synthesise under ADR 0025 "
            "(log-space target makes positivity structural)."
        )
    if max_plausible is not None and float(np.max(np.abs(shape))) > max_plausible:
        raise ValueError(
            f"the distilled shape reaches {float(np.max(np.abs(shape))):.3g} for category "
            f"{category}, more than {max_plausible:.3g} — it has left the data it was fitted "
            "on entirely, which is what a near-miss of an interior pole looks like on a "
            "grid that does not land on the pole. Re-synthesise under ADR 0025 (capped "
            "denominator complexity)."
        )


def plausible_bound_for(eval_dir: Path | str, target: str) -> float | None:
    """How large a transferred ``target`` shape may get, from the synthesis's own data.

    Equation transfer is on-grid only, so a shape is always evaluated with the constants
    of a condition the synthesis actually fitted — it should resemble those curves. The
    bound is 1000× the largest fitted value (:func:`sr_predict.plausible_bound`), which no
    sane shape approaches and which a near-missed pole clears by ten orders of magnitude.

    ``None`` when the synthesis kept no feature table: better to leave the criterion off
    than to invent a scale for it.
    """
    import pandas as pd

    from sr_predict import plausible_bound

    path = Path(eval_dir) / "features_full.parquet"
    if not path.is_file():
        return None
    observed = pd.read_parquet(path, columns=[target])[target].to_numpy(dtype=float)
    return plausible_bound(observed)


def matching_conditions(
    category_map: CategoryMap, target_eps: float, target_T: int
) -> list[tuple[int, dict]]:
    """The trained conditions whose ``(eps, T)`` exactly matches the target.

    Returns ``(category, condition)`` for every map entry at the exact target
    ``(eps, T)`` — read off, not selected — each tagged with its 1-indexed
    ``category`` (its position in the map). Empty when the target is off-grid: the
    closed form has no constants there.
    """
    return [
        (i + 1, condition)
        for i, condition in enumerate(category_map)
        if condition["eps"] == target_eps and condition["T"] == target_T
    ]


# ---------------------------------------------------------------------------
# Cell orchestration + CLI (integration glue; exercised end-to-end, not unit-tested)
# ---------------------------------------------------------------------------


def run_equation_cell(
    eval_dir: Path | str,
    target: TargetSpec,
    cache_root: Path | str = "cache",
    batch_size: int = 250,
    num_reps: int = 3,
    seed: int = 0,
    category: int = 0,
) -> list[Path]:
    """Transfer the distilled condition(s) at the target ``(eps, T)`` and write cells.

    Loads the sigma and clip closed forms from an SR ``eval_dir``, and for each
    condition present at the exact target ``(eps, T)`` (read off, not selected):
    evaluates both shapes on the target step grid, seats the sigma shape on the
    target budget, carries the distilled clip, evaluates natively on the target for
    ``num_reps`` seeds via the shared eval core, and writes a ``producer="equation"``
    cell. Raises if the target is off-grid or the run lacks a clip equation.

    ``category`` narrows the run to a single condition — the launcher gives each its
    own task so a task's cost is one evaluation rather than one per condition. It
    changes nothing about which cell is written, only how many per invocation; 0
    keeps every matching condition.
    """
    from conf.scope import RunContext, using
    from conf.singleton_conf import SingletonConfig
    from environments.dp_params import DPTrainingParams
    from privacy.gdp_privacy import get_privacy_params
    from sr_category import load_category_map
    from symbolic_regression_eval import _load_target
    from transfer_curve import schedule_data_to_results
    from util.baselines import Baseline
    from util.dataloaders import get_dataset_shapes
    from util.transfer import (
        RawArraySchedule,
        build_target_config,
        seat_on_budget,
        transfer_rows,
        write_transfer_cell,
    )

    eval_dir = Path(eval_dir)
    category_map = load_category_map(eval_dir / "category_map.json")
    conditions = matching_conditions(category_map, target.eps, target.T)
    if not conditions:
        raise SystemExit(
            f"no trained condition at (eps={target.eps:g}, T={target.T}); "
            "equation transfer is on-grid only (the template constants are indexed "
            "by discrete condition, not a function of eps/T)"
        )
    if category:
        conditions = [(cat, cond) for cat, cond in conditions if cat == category]
        if not conditions:
            raise SystemExit(
                f"category {category} is not among the conditions trained at "
                f"(eps={target.eps:g}, T={target.T})"
            )

    sigma_model = _load_target(eval_dir, "sigma")
    clip_model = _load_target(eval_dir, "clip")
    if sigma_model is None or clip_model is None:
        raise SystemExit(
            f"equation transfer needs both 'sigma' and 'clip' distilled under {eval_dir}; "
            "re-run symbolic_regression.py with --targets sigma clip"
        )

    bounds = {t: plausible_bound_for(eval_dir, t) for t in ("sigma", "clip")}
    arm = synthesis_arm(eval_dir)
    # The synthesis is scoped to one arm (ADR 0016), so the closed form it distilled is
    # that arm's shape and its target runs at that arm's momentum (ADR 0021).
    config = build_target_config(target, batch_size, arm)
    target_T = int(target.T)
    paths: list[Path] = []
    # The inner DP-SGD path also reads the singleton / RunContext, so the whole eval
    # (not just param construction) must stay inside both scopes.
    with SingletonConfig.override(config), using(RunContext(config)):
        X_shape, *_ = get_dataset_shapes()
        gdp_params = get_privacy_params(X_shape[0])
        env_params = DPTrainingParams.create_direct_from_config()

        for category, condition in conditions:
            sigma_shape = evaluate_equation_shape(
                sigma_model.model, category, target_T, max_plausible=bounds["sigma"]
            )
            clip_shape = evaluate_equation_shape(
                clip_model.model, category, target_T, max_plausible=bounds["clip"]
            )
            # seat_on_budget takes the multiplier s = sigma/clip, not the raw noise
            # scale — the GDP budget is sum_i exp((C_i/sigma_i)^2), so the clips are
            # part of the constraint. Same divide-then-multiply as build_curve_schedule.
            sigmas = seat_on_budget(sigma_shape / clip_shape, gdp_params) * clip_shape
            schedule = RawArraySchedule(sigmas, clip_shape)

            source = equation_source(category, condition, arm=arm)
            baseline = Baseline(env_params, gdp_params, jr.PRNGKey(seed), num_reps=num_reps)
            df = baseline.generate_schedule_data(
                schedule, name=f"Equation Transfer ({source.run_id})"
            )
            rows = transfer_rows("equation", source, target, schedule_data_to_results(df))
            paths.append(write_transfer_cell(rows, cache_root))
    return paths


@dataclasses.dataclass
class EquationCellConfig:
    """One equation-transfer invocation: every distilled condition at a target (eps, T)."""

    eval_dir: str
    """SR evaluation dir with sigma/ and clip/ equations.csv + category_map.json."""
    target: str
    """Target dataset name (eyepacs, imagenet, chexpert, ...)."""
    target_eps: float
    target_T: int
    target_delta: float = 1e-7
    target_arch: str = ""
    """Arch label recorded on the cell rows; the arch itself is auto-derived from the dataset."""
    batch_size: int = 250
    cache_root: str = "cache"
    num_reps: int = 3
    seed: int = 0
    category: int = 0
    """Transfer only this condition category; 0 = every condition at the target (eps, T)."""


def main(conf: EquationCellConfig) -> None:
    target = TargetSpec(
        name=conf.target,
        eps=conf.target_eps,
        delta=conf.target_delta,
        T=conf.target_T,
        arch=conf.target_arch,
    )
    for out in run_equation_cell(
        conf.eval_dir,
        target,
        cache_root=conf.cache_root,
        batch_size=conf.batch_size,
        num_reps=conf.num_reps,
        seed=conf.seed,
        category=conf.category,
    ):
        print(f"wrote {out}")


if __name__ == "__main__":
    main(tyro.cli(EquationCellConfig))
