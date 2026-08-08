import numpy as np
import pandas as pd
import pytest

from sr_predict import TemplatePredictor as _TemplatePredictor
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
        # Shapes here stay strictly positive: a σ or C of 0 is not a schedule, and the
        # producer rejects one before it can reach the budget seater (ADR 0025).
        predictor = _predictor("#1 * #2 + 1", p1=[10.0, 20.0])

        shape = evaluate_equation_shape(predictor, category=1, target_T=200)

        # One value per target step, evaluated on step_norm in [0, 1).
        assert shape.shape == (200,)
        # f = step_norm * p1[cat=1] + 1 = 10*step_norm + 1, over inner_step / T.
        assert shape[0] == 1.0
        assert np.isclose(shape[-1], 10.0 * 199 / 200 + 1.0)

    def test_the_grid_is_the_one_the_equation_was_fitted_over(self):
        """step_norm is inner_step / T (compile_results_fetch), so the last step of a
        length-T run sits at (T-1)/T, not at 1. Evaluating on linspace(0, 1, T) instead
        stretches the whole shape by T/(T-1) and reads every fitted feature off by up to
        one step — worst at the small T the transfer grid actually uses."""
        predictor = _predictor("#1 * #2 + #2", p1=[4.0])  # 4*step_norm + 4

        shape = evaluate_equation_shape(predictor, category=1, target_T=4)

        assert np.allclose(shape, [4.0, 5.0, 6.0, 7.0])  # step_norm = 0, .25, .5, .75

    def test_different_conditions_evaluate_to_different_shapes(self):
        predictor = _predictor("(#1 + 1) * #2", p1=[10.0, 20.0])

        cat1 = evaluate_equation_shape(predictor, category=1, target_T=50)
        cat2 = evaluate_equation_shape(predictor, category=2, target_T=50)

        # Same shape f, different per-condition constant -> cat2 is 2x cat1.
        assert not np.allclose(cat1, cat2)
        assert np.allclose(cat2, 2.0 * cat1)


class TestAnUnusableShapeIsRejectedWhereItIsProduced:
    """Equations distilled before ADR 0025 can still hold an interior pole, and this
    producer is where one first becomes visible. Left unchecked it flows into
    seat_on_budget, whose bisection saturates at its bracket ceiling and reports
    "spent 40258 of 1.11e+06 (3.6%)" — true, but it names neither the equation nor
    the category, and points at the seater rather than at the shape it was handed.
    """

    def test_a_pole_on_the_target_grid_is_named_at_the_point_of_evaluation(self):
        # Vanishes at step_norm = 0.5, which T=4 lands on exactly.
        predictor = _predictor("#2 / (#1 - 0.5)", p1=[1.0])

        with pytest.raises(ValueError, match="category 1"):
            evaluate_equation_shape(predictor, category=1, target_T=4)

    def test_a_non_positive_shape_is_rejected(self):
        """σ ≤ 0 is not a noise scale and C ≤ 0 kills the gradient; either one makes
        the multiplier σ/C meaningless before the budget is ever consulted."""
        predictor = _predictor("#1 - #2", p1=[0.5])

        with pytest.raises(ValueError, match="non-positive"):
            evaluate_equation_shape(predictor, category=1, target_T=8)

    def test_a_finite_but_implausible_spike_is_rejected(self):
        """Whether a pole is *hit* depends on the target T, so finiteness is not enough.
        At T=2000 the f152229a clip equation lands 2.2e-4 from its pole and returns
        2.9e13 — finite, positive, and ruinous once seat_on_budget squares it."""
        predictor = _predictor("exp(#1 * #2)", p1=[40.0])

        with pytest.raises(ValueError, match="left the data"):
            evaluate_equation_shape(predictor, category=1, target_T=4, max_plausible=100.0)

    def test_without_a_bound_the_magnitude_criterion_is_simply_off(self):
        """A synthesis that kept no feature table gives no scale to judge against;
        the other two criteria still apply."""
        predictor = _predictor("exp(#1 * #2)", p1=[40.0])

        shape = evaluate_equation_shape(predictor, category=1, target_T=4)

        assert np.all(np.isfinite(shape))

    def test_a_healthy_shape_passes_through_untouched(self):
        predictor = _predictor("#1 + #2", p1=[1.0])

        shape = evaluate_equation_shape(predictor, category=1, target_T=4)

        assert np.allclose(shape, [1.0, 1.25, 1.5, 1.75])


class TestThePlausibilityBoundComesFromTheSynthesis:
    """The producer has no independent idea of what a σ or a C should be worth. It reads
    the scale off the very targets the synthesis was fitted on, which is exactly the
    range an on-grid transferred shape should reproduce."""

    def _eval_dir(self, tmp_path, sigma):
        pd.DataFrame({"sigma": sigma, "clip": sigma}).to_parquet(
            tmp_path / "features_full.parquet", index=False
        )
        return tmp_path

    def test_the_bound_is_a_wide_multiple_of_the_largest_fitted_target(self, tmp_path):
        from transfer_equation import plausible_bound_for

        eval_dir = self._eval_dir(tmp_path, [0.1, 1.5, 0.8])

        assert plausible_bound_for(eval_dir, "sigma") == 1e3 * 1.5

    def test_a_synthesis_without_a_feature_table_yields_no_bound(self, tmp_path):
        from transfer_equation import plausible_bound_for

        assert plausible_bound_for(tmp_path, "sigma") is None

    def test_non_finite_fitted_rows_do_not_set_the_scale(self, tmp_path):
        """features_full.parquet keeps every inner step, including any the run wrote as
        inf; one of those would push the bound to infinity and disable the criterion."""
        from transfer_equation import plausible_bound_for

        eval_dir = self._eval_dir(tmp_path, [0.1, 2.0, np.inf, np.nan])

        assert plausible_bound_for(eval_dir, "sigma") == 1e3 * 2.0


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


class TestSynthesisArm:
    """A synthesis is scoped to a single arm (ADR 0016), but the *condition* is
    (dataset, eps, T, arch) and carries no arm — so an equation cell's arm comes from
    the synthesis that produced it, read off the eval dir's manifest. It has to be
    there: `_OVERLAY_KEYS` includes the arm, so an equation cell tagged with the wrong
    one would never overlay the curve cells it was distilled from."""

    def _manifest(self, tmp_path, optimizers):
        import json

        (tmp_path / "manifest.json").write_text(json.dumps({"config": {"optimizers": optimizers}}))
        return tmp_path

    def test_a_single_arm_synthesis_reports_that_arm(self, tmp_path):
        from transfer_equation import synthesis_arm

        assert synthesis_arm(self._manifest(tmp_path, ["sgd-m0.9"])) == "sgd-m0.9"

    def test_an_unscoped_synthesis_reports_no_arm(self, tmp_path):
        # `optimizers: []` means the fit was not filtered by arm, so its conditions
        # pool both — there is no single arm to claim, and "" keeps it out of the
        # per-arm overlay rather than mislabelling it as one arm's.
        from transfer_equation import synthesis_arm

        assert synthesis_arm(self._manifest(tmp_path, [])) == ""

    def test_the_arm_lands_on_the_equation_cells_source_policy(self, tmp_path):
        condition = {"dataset": "cifar-10", "eps": 1.0, "T": 200, "arch_label": "cnn"}

        assert equation_source(1, condition, arm="sgd-m0.9").arm == "sgd-m0.9"
