"""Tests for the *space* a synthesis is fitted in (ADR 0025).

σ and C are strictly positive and span 11–65× within a single run, so fitting them
in natural units lets absolute squared error ignore the low end of every curve and
lets the search propose equations that go negative or zero. Fitting ``log`` of the
target instead makes the error relative and makes positivity structural — the
prediction is ``exp(f)``, which cannot be ≤ 0 whatever ``f`` does.

The transform is a property of the *fit*, so it is a synthesis-identity field: a
log-space synthesis must never warm-start from a natural-space one.
"""

import json

import numpy as np
import pandas as pd
import pytest

from sr_predict import (
    InvertingPredictor,
    TemplatePredictor,
    fittable_mask,
    from_fit_space,
    to_fit_space,
)


def _equations(expr: str, p1: list[float]) -> pd.DataFrame:
    """A one-row front for shape ``f`` with one per-condition constant ``p1``."""
    return pd.DataFrame(
        {"equation": [f"f = {expr}; p1 = [{', '.join(map(str, p1))}]"], "selected": [True]}
    )


def _X(step_norm, category=1) -> np.ndarray:
    step_norm = np.atleast_1d(np.asarray(step_norm, dtype=float))
    return np.column_stack([step_norm, np.full(len(step_norm), category)])


class TestFitSpaceRoundTrip:
    def test_log_space_round_trips_a_positive_schedule(self):
        y = np.array([0.0089, 0.15, 1.95])  # the observed σ/C range of a FirSweep arm

        assert np.allclose(from_fit_space(to_fit_space(y, "log"), "log"), y)

    def test_identity_leaves_the_target_untouched(self):
        y = np.array([-1.0, 0.0, 2.5])

        assert np.array_equal(to_fit_space(y, "identity"), y)
        assert np.array_equal(from_fit_space(y, "identity"), y)

    def test_an_unknown_transform_is_rejected_rather_than_silently_ignored(self):
        with pytest.raises(ValueError, match="sqrt"):
            to_fit_space(np.array([1.0]), "sqrt")


class TestRowsTheTransformCannotRepresent:
    """log has no value at y ≤ 0. Such rows must be dropped explicitly — left in,
    they become NaN targets and PySR's all-or-nothing validity check would reject
    every candidate equation for the whole dataset."""

    def test_non_positive_targets_are_not_fittable_in_log_space(self):
        y = np.array([1.0, 0.0, -0.5, np.nan, np.inf, 0.25])

        assert list(fittable_mask(y, "log")) == [True, False, False, False, False, True]

    def test_identity_space_only_rejects_non_finite_targets(self):
        y = np.array([1.0, 0.0, -0.5, np.nan, np.inf])

        assert list(fittable_mask(y, "identity")) == [True, True, True, False, False]


class TestPredictionsComeBackInNaturalUnits:
    """Everything downstream — the evaluator's metrics, and transfer's seating of the
    shape on the target budget — consumes σ and C in natural units. The inversion
    therefore belongs at the predictor boundary, so no caller can forget it."""

    def test_a_log_space_predictor_exponentiates_its_equation(self):
        # f = step_norm * p1 in LOG space; natural-space prediction is exp of that.
        inner = TemplatePredictor(_equations("#1 * #2", p1=[2.0]), ["step_norm", "category"])
        predictor = InvertingPredictor(inner, "log")

        assert np.allclose(predictor.predict(_X([0.0, 0.5, 1.0])), np.exp([0.0, 1.0, 2.0]))

    def test_an_equation_that_goes_negative_still_predicts_a_positive_schedule(self):
        """The structural win: no fitted shape can produce a negative σ or a zero C.
        On the pre-0025 front, σ row 6 reached −0.0077 and four clip rows hit exactly 0."""
        inner = TemplatePredictor(_equations("#1 - #2", p1=[100.0]), ["step_norm", "category"])
        predictor = InvertingPredictor(inner, "log")

        pred = predictor.predict(_X(np.linspace(0.0, 1.0, 64)))

        assert np.all(pred > 0.0)

    def test_identity_space_predictions_pass_straight_through(self):
        inner = TemplatePredictor(_equations("#1 * #2", p1=[2.0]), ["step_norm", "category"])

        raw = inner.predict(_X([0.0, 0.5, 1.0]))
        wrapped = InvertingPredictor(inner, "identity").predict(_X([0.0, 0.5, 1.0]))

        assert np.array_equal(raw, wrapped)

    def test_a_front_row_selected_by_index_is_inverted_too(self):
        """plot_pareto walks the whole front by index; those predictions are compared
        against natural-units actuals, so they need the same inversion as the selected row."""
        equations = pd.DataFrame(
            {
                "equation": ["f = #2; p1 = [0.0]", "f = #1 * #2; p1 = [2.0]"],
                "selected": [False, True],
            }
        )
        predictor = InvertingPredictor(
            TemplatePredictor(equations, ["step_norm", "category"]), "log"
        )

        assert np.allclose(predictor.predict(_X([0.5]), index=0), 1.0)  # exp(0)
        assert np.allclose(predictor.predict(_X([0.5]), index=1), np.exp(1.0))


class TestTheTransformIsRecoveredFromTheSynthesis:
    """A persisted synthesis has to say which space it was fitted in, or a consumer
    would exponentiate a natural-units equation (or fail to exponentiate a log one).
    Syntheses predating ADR 0025 carry no such record and are natural-units."""

    def _synthesis(self, tmp_path, config: dict | None) -> object:
        tdir = tmp_path / "sigma"
        tdir.mkdir()
        pd.DataFrame({"equation": ["f = #1 * #2; p1 = [2.0]"], "selected": [True]}).to_csv(
            tdir / "equations.csv", index=False
        )
        (tdir / "feature_names.json").write_text(json.dumps(["step_norm", "category"]))
        (tmp_path / "category_map.json").write_text(
            json.dumps([{"dataset": "cifar-10", "eps": 1.0, "T": 200, "arch_label": "mlp"}])
        )
        if config is not None:
            (tmp_path / "manifest.json").write_text(json.dumps({"config": config}))
        from symbolic_regression_eval import _load_target

        return _load_target(tmp_path, "sigma")

    def test_a_log_space_synthesis_predicts_in_natural_units(self, tmp_path):
        tm = self._synthesis(tmp_path, {"target_transform": "log"})

        assert np.allclose(tm.model.predict(_X([0.5])), np.exp(1.0))

    def test_a_pre_adr_0025_synthesis_is_read_as_natural_units(self, tmp_path):
        # No manifest at all: the oldest artefact layout must still evaluate as fitted.
        tm = self._synthesis(tmp_path, config=None)

        assert np.allclose(tm.model.predict(_X([0.5])), 1.0)

    def test_a_manifest_without_the_field_is_read_as_natural_units(self, tmp_path):
        tm = self._synthesis(tmp_path, {"points_per_run": 50})

        assert np.allclose(tm.model.predict(_X([0.5])), 1.0)


class TestTheTransformIsPartOfTheSynthesisIdentity:
    def test_fitting_in_a_different_space_changes_the_slug(self):
        """Two spaces are two different problems; sharing a run directory would let
        one warm-start from the other's front and fit log-targets to natural-units state."""
        from sr_identity import slug_for

        natural = {"cache_dir": "sweep", "target_transform": "identity"}
        log = {"cache_dir": "sweep", "target_transform": "log"}

        assert slug_for(natural) != slug_for(log)

    def test_the_transform_survives_a_chain_resubmit(self):
        from dataclasses import asdict

        import tyro

        from sr_identity import identity_flags, slug_for
        from symbolic_regression import PySRConfig

        conf = PySRConfig(cache_dir="sweep", target_transform="identity")
        reparsed = tyro.cli(
            PySRConfig, args=["--cache_dir", "sweep", *identity_flags(asdict(conf))]
        )

        assert reparsed.target_transform == "identity"
        assert slug_for(asdict(reparsed)) == slug_for(asdict(conf))


class TestWhatTheSearchIsHandedIsAlreadyTransformed:
    """PySR minimises squared error on the y it is given, so the transform has to be
    applied to the column before the fit — not wrapped around the loss."""

    def test_the_target_column_is_handed_over_in_log_space(self):
        from symbolic_regression import to_fit_space_df

        df = pd.DataFrame({"step_norm": [0.0, 0.5], "sigma": [1.0, np.e]})

        out = to_fit_space_df(df, "sigma", "log")

        assert np.allclose(out["sigma"], [0.0, 1.0])
        assert np.allclose(out["step_norm"], [0.0, 0.5]), "features must not be transformed"

    def test_rows_the_transform_cannot_represent_are_dropped(self):
        from symbolic_regression import to_fit_space_df

        df = pd.DataFrame({"step_norm": [0.0, 0.5, 1.0], "sigma": [1.0, 0.0, np.e]})

        out = to_fit_space_df(df, "sigma", "log")

        assert list(out["step_norm"]) == [0.0, 1.0]
        assert np.allclose(out["sigma"], [0.0, 1.0])
