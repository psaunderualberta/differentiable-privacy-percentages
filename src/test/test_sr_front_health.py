"""Tests for the dense-grid tripwire run over a finished Pareto front (ADR 0025).

The pole in f152229a survived because the fit only ever evaluated its equations on
the ~50 step_norm values it was trained on. Nothing between the search and the
transfer producer looked at the shape on a grid as fine as the one it would be
*used* on, so a 1.3e15 spike shipped as the selected clip equation.

This is a tripwire, not the mechanism — the denominator cap and the log-space fit
are what make the failure impossible. It exists so that if some future operator or
constraint change reopens the door, it is caught at synthesis time (with the fit
still cheap to redo) rather than weeks later inside seat_on_budget.
"""

import numpy as np
import pandas as pd

from sr_predict import TemplatePredictor, front_health


def _front(exprs: list[str], p1: list[float]) -> pd.DataFrame:
    """A front of ``exprs``, all sharing per-condition constants ``p1``; last selected."""
    consts = ", ".join(map(str, p1))
    return pd.DataFrame(
        {
            "complexity": list(range(1, len(exprs) + 1)),
            "equation": [f"f = {e}; p1 = [{consts}]" for e in exprs],
            "selected": [i == len(exprs) - 1 for i in range(len(exprs))],
        }
    )


def _health(exprs: list[str], p1: list[float], **kwargs) -> pd.DataFrame:
    equations = _front(exprs, p1)
    predictor = TemplatePredictor(equations, ["step_norm", "category"])
    return front_health(predictor, equations, n_conditions=len(p1), **kwargs)


class TestTheGridIsDenserThanAnyTargetItWillBeUsedOn:
    def test_it_reports_one_row_per_front_row(self):
        health = _health(["#2", "#1 * #2"], p1=[1.0])

        assert list(health["complexity"]) == [1, 2]
        assert list(health["selected"]) == [False, True]

    def test_a_clean_shape_is_healthy(self):
        health = _health(["#1 * #2 + #2"], p1=[1.0, 2.0])

        assert bool(health["healthy"].iloc[0])
        assert int(health["n_nonfinite"].iloc[0]) == 0

    def test_a_pole_inside_the_domain_is_caught(self):
        """The f152229a failure, reduced: a denominator that vanishes at step_norm
        ≈ 0.9917 — between training samples, but squarely inside the grid transfer
        evaluates on."""
        health = _health(["#2 / (#1 - 0.991717)"], p1=[1.0], max_plausible=2.0)

        assert not bool(health["healthy"].iloc[0])
        # How large the spike gets depends on how near a grid point falls to the pole;
        # what matters is that it is orders of magnitude outside the fitted range.
        assert float(health["max_abs"].iloc[0]) > 1e3

    def test_a_pole_is_caught_even_when_only_one_condition_hits_it(self):
        """The per-condition constants shift the shape, so a front row can be finite
        for most categories and diverge for one. Every category is swept."""
        # p1=5 gives a bounded, positive 1/(5 - step_norm); p1=0.5 puts a pole at 0.5.
        health = _health(["1.0 / (#2 - #1)"], p1=[5.0, 0.5])

        assert not bool(health["healthy"].iloc[0])
        assert int(health["n_nonfinite"].iloc[0]) > 0

    def test_a_shape_that_reaches_zero_is_flagged(self):
        """σ = 0 is not a schedule (division by it defines the privacy multiplier),
        and C = 0 kills the gradient. Both were live on the pre-0025 front."""
        health = _health(["#1 - #2"], p1=[0.5])

        assert not bool(health["healthy"].iloc[0])
        assert int(health["n_nonpositive"].iloc[0]) > 0

    def test_a_finite_but_implausible_spike_is_flagged(self):
        """is_valid_array is a sum-based NaN/Inf test, so a finite 1e15 spike passes
        every check PySR makes. The bound is what the fitted targets actually reached."""
        health = _health(["exp(#1 * #2)"], p1=[40.0], max_plausible=10.0)

        assert not bool(health["healthy"].iloc[0])

    def test_a_shape_within_the_plausible_bound_stays_healthy(self):
        health = _health(["exp(#1 * #2)"], p1=[1.0], max_plausible=10.0)

        assert bool(health["healthy"].iloc[0])

    def test_the_grid_follows_the_fits_step_norm_convention(self):
        """step_norm is inner_step / T (compile_results_fetch), so it reaches 1 - 1/n,
        never 1. A grid that included 1.0 would probe a point no run ever visits."""
        from sr_predict import dense_step_grid

        grid = dense_step_grid(4)

        assert np.allclose(grid, [0.0, 0.25, 0.5, 0.75])
