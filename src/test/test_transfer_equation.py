import numpy as np
import pandas as pd

from symbolic_regression_eval import _TemplatePredictor
from transfer_equation import (
    equation_source,
    evaluate_equation_shape,
    matching_conditions,
)


def _predictor(expr: str, p1: list[float]) -> _TemplatePredictor:
    """A template predictor for shape ``f`` with one per-condition constant ``p1``.

    ``#1`` is step_norm and ``#2`` is that condition's ``p1`` value; ``p1`` carries
    the constant for every 1-indexed category. One (selected) equation row."""
    equations = pd.DataFrame(
        {"equation": [f"f = {expr}; p1 = [{', '.join(map(str, p1))}]"], "selected": [True]}
    )
    return _TemplatePredictor(equations, ["step_norm", "category"])


# A category map is an ordered list of condition records; entry i -> category i+1
# (sr_category.CategoryMap). Two conditions share (eps, T) here, one is off-grid.
_CATEGORY_MAP = [
    {"dataset": "eyepacs", "eps": 1.0, "T": 200, "arch_label": "cnn"},
    {"dataset": "cifar-10", "eps": 1.0, "T": 200, "arch_label": "mlp"},
    {"dataset": "eyepacs", "eps": 2.0, "T": 200, "arch_label": "cnn"},
]


class TestExactEpsTMatchGuard:
    """Template constants are indexed by discrete (dataset, eps, T, arch), not a
    function of eps/T, so the closed form is undefined off-grid (ADR 0008).
    Equation transfer runs ONLY at a target (eps, T) that exactly matches a trained
    condition; every condition present at that (eps, T) is transferred (read off)."""

    def test_returns_every_condition_at_the_exact_eps_T_with_its_category(self):
        matches = matching_conditions(_CATEGORY_MAP, target_eps=1.0, target_T=200)

        # Both conditions at (1.0, 200) are kept — read off, not selected — each
        # tagged with its 1-indexed category (position in the map).
        assert [(cat, c["dataset"]) for cat, c in matches] == [
            (1, "eyepacs"),
            (2, "cifar-10"),
        ]

    def test_off_grid_eps_T_returns_nothing(self):
        assert matching_conditions(_CATEGORY_MAP, target_eps=1.5, target_T=200) == []
        assert matching_conditions(_CATEGORY_MAP, target_eps=1.0, target_T=100) == []


class TestClosedFormEvaluatedOnTargetGrid:
    """f is closed-form over step_norm, so the producer *evaluates* it on the target
    step grid rather than resampling a length-T array (ADR 0008). Each condition's
    category selects its own constants, so different conditions give different
    shapes at the same (eps, T)."""

    def test_shape_has_target_T_points_over_the_normalized_grid(self):
        predictor = _predictor("#1 * #2", p1=[10.0, 20.0])

        shape = evaluate_equation_shape(predictor, category=1, target_T=200)

        # One value per target step, evaluated on step_norm in [0, 1].
        assert shape.shape == (200,)
        # f = step_norm * p1[cat=1] = step_norm * 10, endpoints pinned.
        assert shape[0] == 0.0
        assert shape[-1] == np.float32(10.0) or np.isclose(shape[-1], 10.0)

    def test_different_conditions_evaluate_to_different_shapes(self):
        predictor = _predictor("#1 * #2", p1=[10.0, 20.0])

        cat1 = evaluate_equation_shape(predictor, category=1, target_T=50)
        cat2 = evaluate_equation_shape(predictor, category=2, target_T=50)

        # Same shape f, different per-condition constant -> cat2 is 2x cat1.
        assert not np.allclose(cat1, cat2)
        assert np.allclose(cat2, 2.0 * cat1)


class TestConditionBecomesItsOwnSourceCell:
    """Read off, not selected (ADR 0008): every condition at the matching (eps, T)
    becomes its own matrix row. Its source_id must be distinct per condition and
    filesystem-safe, since it lands in the cell's parquet filename."""

    def test_source_carries_condition_provenance_with_fs_safe_distinct_id(self):
        cond_a = {"dataset": "cifar-10", "eps": 1.0, "T": 200, "arch_label": "cnn"}
        cond_b = {"dataset": "eyepacs", "eps": 1.0, "T": 200, "arch_label": "mlp"}

        src_a = equation_source(1, cond_a)
        src_b = equation_source(2, cond_b)

        # Provenance is the condition itself; delta/p unknown from a category map.
        assert (src_a.dataset, src_a.eps, src_a.T, src_a.arch) == ("cifar-10", 1.0, 200, "cnn")
        assert np.isnan(src_a.delta) and np.isnan(src_a.p)

        # Distinct per condition, and safe to embed in a filename (no separators).
        assert src_a.run_id != src_b.run_id
        for run_id in (src_a.run_id, src_b.run_id):
            assert "/" not in run_id and " " not in run_id
