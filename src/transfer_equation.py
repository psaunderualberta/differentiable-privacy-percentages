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
from util.transfer import SourcePolicy, TargetSpec


def equation_source(category: int, condition: dict) -> SourcePolicy:
    """The ``SourcePolicy`` for a distilled condition transferred as an equation.

    A condition is not a single learned run, so its provenance IS the condition
    ``(dataset, eps, T, arch)``, tagged by an fs-safe id (it becomes part of the
    cell filename). ``delta`` and ``p`` are NaN: a category map carries neither.
    """
    dataset, arch = condition["dataset"], condition["arch_label"]
    run_id = f"{dataset}_eps{condition['eps']:g}_T{int(condition['T'])}_{arch}_cat{category}"
    return SourcePolicy(
        run_id=run_id,
        dataset=str(dataset),
        eps=float(condition["eps"]),
        delta=float("nan"),
        T=int(condition["T"]),
        p=float("nan"),
        arch=str(arch),
    )


def evaluate_equation_shape(predictor, category: int, target_T: int) -> np.ndarray:
    """Evaluate the selected distilled shape on the target step grid.

    ``f`` is closed-form over ``step_norm``, so the shape is *evaluated* on
    ``linspace(0, 1, target_T)`` — one value per target step — rather than
    resampled from a fixed-length array. ``category`` selects the condition whose
    per-condition constants are inlined into ``f``.
    """
    step_norm = np.linspace(0.0, 1.0, target_T)
    X = np.column_stack([step_norm, np.full(target_T, category)])
    return np.asarray(predictor.predict(X), dtype=float)


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
    num_reps: int = 8,
    seed: int = 0,
) -> list[Path]:
    """Transfer every distilled condition at the target ``(eps, T)`` and write its cell.

    Loads the sigma and clip closed forms from an SR ``eval_dir``, and for each
    condition present at the exact target ``(eps, T)`` (read off, not selected):
    evaluates both shapes on the target step grid, seats the sigma shape on the
    target budget, carries the distilled clip, evaluates natively on the target for
    ``num_reps`` seeds via the shared eval core, and writes a ``producer="equation"``
    cell. Raises if the target is off-grid or the run lacks a clip equation.
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

    sigma_model = _load_target(eval_dir, "sigma")
    clip_model = _load_target(eval_dir, "clip")
    if sigma_model is None or clip_model is None:
        raise SystemExit(
            f"equation transfer needs both 'sigma' and 'clip' distilled under {eval_dir}; "
            "re-run symbolic_regression.py with --targets sigma clip"
        )

    config = build_target_config(target, batch_size)
    target_T = int(target.T)
    paths: list[Path] = []
    # The inner DP-SGD path also reads the singleton / RunContext, so the whole eval
    # (not just param construction) must stay inside both scopes.
    with SingletonConfig.override(config), using(RunContext(config)):
        X_shape, *_ = get_dataset_shapes()
        gdp_params = get_privacy_params(X_shape[0])
        env_params = DPTrainingParams.create_direct_from_config()

        for category, condition in conditions:
            sigma_shape = evaluate_equation_shape(sigma_model.model, category, target_T)
            clip_shape = evaluate_equation_shape(clip_model.model, category, target_T)
            sigmas = seat_on_budget(sigma_shape, gdp_params)
            schedule = RawArraySchedule(sigmas, clip_shape)

            source = equation_source(category, condition)
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
    num_reps: int = 8
    seed: int = 0


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
    ):
        print(f"wrote {out}")


if __name__ == "__main__":
    main(tyro.cli(EquationCellConfig))
