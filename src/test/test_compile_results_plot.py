"""Behaviour of the plot layer's read-off and caption logic.

Runs within a cell do not all reach the same outer step, so "final accuracy" is
not one quantity across a cell's seeds (ADR 0014). These tests pin what the
figures are actually reporting.
"""

from __future__ import annotations

import pandas as pd
import pytest

import compile_results_plot as crp

LEARNED = crp.LEARNED
CONSTANT = crp.CONSTANT


def _scalars(rows: list[dict]) -> pd.DataFrame:
    base = {
        "optimizer": "sgd-m0.9",
        "dataset": "mnist",
        "eps": 10.0,
        "T": 5000,
        "arch_label": "mlp-64",
        "axis": "arch",
        "schedule": LEARNED,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def _histories(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestCommonStepReadOff:
    def test_seeds_are_read_at_the_shortest_seeds_last_step(self):
        scalars = _scalars(
            [
                {"seed": 0, "mean_acc": 0.90, "mean_loss": 0.10, "final_outer_step": 5},
                {"seed": 1, "mean_acc": 0.50, "mean_loss": 0.90, "final_outer_step": 2},
            ]
        )
        histories = _histories(
            [
                {
                    "run_id": "a",
                    "seed": 0,
                    "outer_step": s,
                    "test_acc": 0.5 + 0.1 * s,
                    "test_loss": 1.0 - 0.1 * s,
                }
                for s in range(6)
            ]
            + [
                {
                    "run_id": "b",
                    "seed": 1,
                    "outer_step": s,
                    "test_acc": 0.3 + 0.1 * s,
                    "test_loss": 1.2 - 0.1 * s,
                }
                for s in range(3)
            ]
        )
        for col in ("optimizer", "dataset", "eps", "T", "arch_label"):
            histories[col] = scalars[col].iloc[0]

        out = crp.read_off_at_common_step(scalars, histories)

        assert set(out["read_off_step"]) == {2}
        by_seed = out.set_index("seed")
        assert by_seed.loc[0, "mean_acc"] == pytest.approx(0.7)
        assert by_seed.loc[1, "mean_acc"] == pytest.approx(0.5)
        assert by_seed.loc[0, "mean_loss"] == pytest.approx(0.8)

    def test_cells_are_truncated_independently(self):
        scalars = _scalars(
            [
                {"seed": 0, "arch_label": "mlp-64", "mean_acc": 0.9, "final_outer_step": 5},
                {"seed": 0, "arch_label": "mlp-128", "mean_acc": 0.8, "final_outer_step": 1},
            ]
        )
        histories = _histories(
            [
                {"seed": 0, "arch_label": "mlp-64", "outer_step": s, "test_acc": 0.5 + 0.1 * s}
                for s in range(6)
            ]
            + [
                {"seed": 0, "arch_label": "mlp-128", "outer_step": s, "test_acc": 0.2 + 0.1 * s}
                for s in range(2)
            ]
        )
        for col in ("optimizer", "dataset", "eps", "T"):
            histories[col] = scalars[col].iloc[0]
        histories["test_loss"] = 0.0

        out = crp.read_off_at_common_step(scalars, histories).set_index("arch_label")

        # A lagging rung must not truncate a healthy one.
        assert out.loc["mlp-64", "read_off_step"] == 5
        assert out.loc["mlp-64", "mean_acc"] == pytest.approx(1.0)
        assert out.loc["mlp-128", "read_off_step"] == 1
        assert out.loc["mlp-128", "mean_acc"] == pytest.approx(0.3)

    def test_baseline_rows_keep_their_values_but_carry_the_read_off_step(self):
        scalars = _scalars(
            [
                {"seed": 0, "mean_acc": 0.9, "final_outer_step": 5},
                {"seed": 0, "schedule": CONSTANT, "mean_acc": 0.42, "final_outer_step": 5},
            ]
        )
        histories = _histories(
            [{"seed": 0, "outer_step": s, "test_acc": 0.5 + 0.1 * s} for s in range(6)]
        )
        for col in ("optimizer", "dataset", "eps", "T", "arch_label"):
            histories[col] = scalars[col].iloc[0]
        histories["test_loss"] = 0.0

        out = crp.read_off_at_common_step(scalars, histories).set_index("schedule")

        # The baseline artifact holds one end-of-run evaluation; it cannot be
        # re-read per step, so it is left alone.
        assert out.loc[CONSTANT, "mean_acc"] == pytest.approx(0.42)
        assert out.loc[CONSTANT, "read_off_step"] == 5

    def test_empty_history_leaves_scalars_untouched(self):
        scalars = _scalars([{"seed": 0, "mean_acc": 0.9, "final_outer_step": 5}])

        out = crp.read_off_at_common_step(scalars, pd.DataFrame())

        assert out["mean_acc"].tolist() == [0.9]

    def test_a_run_missing_from_histories_keeps_its_fetched_value(self):
        # Seed 1 has no history rows at all; it must not become NaN.
        scalars = _scalars(
            [
                {"seed": 0, "mean_acc": 0.9, "final_outer_step": 3},
                {"seed": 1, "mean_acc": 0.4, "final_outer_step": 3},
            ]
        )
        histories = _histories(
            [{"seed": 0, "outer_step": s, "test_acc": 0.5 + 0.1 * s} for s in range(4)]
        )
        for col in ("optimizer", "dataset", "eps", "T", "arch_label"):
            histories[col] = scalars[col].iloc[0]
        histories["test_loss"] = 0.0

        out = crp.read_off_at_common_step(scalars, histories).set_index("seed")

        assert out.loc[0, "mean_acc"] == pytest.approx(0.8)
        assert out.loc[1, "mean_acc"] == pytest.approx(0.4)


class TestCaption:
    """Captions must report what was actually plotted, not a hardcoded n."""

    def test_uniform_cells_report_a_single_n_and_step(self):
        caption = crp.ci_caption(pd.Series([8, 8, 8]), pd.Series([1000, 1000, 1000]))
        assert caption == "shaded = 95% CI (n = 8 seeds; read off at outer step 1000)"

    def test_ragged_cells_report_the_range(self):
        caption = crp.ci_caption(pd.Series([5, 8, 6]), pd.Series([850, 1000, 900]))
        assert caption == "shaded = 95% CI (n = 5–8 seeds; read off at outer steps 850–1000)"

    def test_step_clause_is_dropped_when_unknown(self):
        # A cache fetched before final_outer_step existed has no step to report.
        caption = crp.ci_caption(pd.Series([8, 8]), pd.Series([float("nan")] * 2))
        assert caption == "shaded = 95% CI (n = 8 seeds)"
