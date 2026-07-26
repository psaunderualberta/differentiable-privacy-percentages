import pandas as pd
import pytest

from transfer_plot import nearest_source, overlay_cells, overlay_stats, transfer_matrix


def _rows(source_id, seeds_accs, producer="curve", target="mnist", t_eps=1.0, t_T=200):
    """Per-seed transfer rows for one source×target cell (schema of util.transfer)."""
    return pd.DataFrame(
        {
            "producer": producer,
            "source_id": source_id,
            "source_dataset": "eyepacs",
            "source_eps": 1.0,
            "source_delta": 1e-7,
            "source_T": 200,
            "source_p": 0.01,
            "source_arch": "cnn",
            "target": target,
            "target_eps": t_eps,
            "target_delta": 1e-7,
            "target_T": t_T,
            "target_arch": "cnn",
            "seed": [s for s, _ in seeds_accs],
            "accuracy": [a for _, a in seeds_accs],
            "loss": 0.5,
        }
    )


class TestTransferMatrixIsReadOff:
    """The descriptive matrix is read off, not selected (ADR 0008): every source
    policy becomes a matrix row, and each cell reports the spread across its
    transferred seeds (generalization consistency) — never a selected best seed."""

    def test_every_source_kept_with_seed_spread_not_a_selected_best(self):
        assembled = pd.concat(
            [
                _rows("runA", [(0, 0.80), (1, 0.90)]),
                _rows("runB", [(0, 0.60), (1, 0.60)]),
            ],
            ignore_index=True,
        )

        matrix = transfer_matrix(assembled)

        # One row per (source_id, target cell) — every source is kept.
        cells = matrix.set_index("source_id")
        assert set(cells.index) == {"runA", "runB"}
        # Cell value is the mean across seeds, and the spread is reported.
        assert cells.loc["runA", "mean_acc"] == pytest.approx(0.85)
        assert cells.loc["runA", "n"] == 2
        assert cells.loc["runA", "spread"] > 0.0
        # runB's seeds agree, so its generalization spread is zero.
        assert cells.loc["runB", "spread"] == 0.0


def _src_at(source_id, s_eps, s_T):
    """One transfer row whose source regime is (s_eps, s_T); target fixed at (1.0, 200)."""
    df = _rows(source_id, [(0, 0.5)], t_eps=1.0, t_T=200)
    df["source_eps"] = s_eps
    df["source_T"] = s_T
    return df


class TestNearestSource:
    """Each target column is annotated with the source nearest in (ε, T) (ADR 0008).

    'Nearest' is relative distance in (ε, T) so the two axes are comparable despite
    their different scales."""

    def test_picks_the_source_closest_in_relative_eps_T(self):
        assembled = pd.concat(
            [
                _src_at("close", 1.0, 220),  # rel dist = 20/200 = 0.10
                _src_at("far", 2.0, 200),  # rel dist = 1.0/1.0 = 1.00
            ],
            ignore_index=True,
        )

        assert nearest_source(assembled, target_eps=1.0, target_T=200) == "close"

    def test_equidistant_sources_break_ties_on_source_id(self):
        # Both sources sit the same relative distance from the target, so the
        # annotation must be deterministic regardless of row order: sorted source_id.
        assembled = pd.concat(
            [
                _src_at("zebra", 1.0, 220),
                _src_at("alpha", 1.0, 180),
            ],
            ignore_index=True,
        )

        assert nearest_source(assembled, target_eps=1.0, target_T=200) == "alpha"


class TestOverlayCells:
    """The curve-vs-equation overlay joins on the source REGIME, not the source
    policy (ADR 0015): the two producers have different row granularity — curve's
    unit is one seed's policy, equation's is a whole distilled condition — and a
    condition has no per-seed identity to match on. Reference cells never
    participate, and a cell is drawn only when both producers have a record."""

    def test_producers_join_on_the_source_regime_not_the_source_id(self):
        # The two sides can never share a source_id: curve's is a W&B run id,
        # equation's is a condition slug. They DO share the regime the policies
        # were learned in, which is exactly what a condition is defined by.
        producers = {
            "curve": pd.concat(
                [
                    _rows("wandb_run_1", [(0, 0.7)], producer="curve"),
                    _rows("wandb_run_2", [(0, 0.7)], producer="curve"),
                ],
                ignore_index=True,
            ),
            "equation": _rows("eyepacs_eps1_T200_cnn_cat1", [(0, 0.8)], producer="equation"),
            # A reference producer is present but must never enter the overlay set.
            "reference": _rows("Constant", [(0, 0.6)], producer="reference"),
        }

        cells = overlay_cells(producers)

        # One cell — the shared regime — despite three distinct source_ids.
        assert cells == [("eyepacs", 1.0, 200, "cnn", "mnist", 1.0, 200)]

    def test_a_regime_only_one_producer_has_is_not_drawn(self):
        curve = _rows("wandb_run_1", [(0, 0.7)], producer="curve")
        equation = _rows("cond_cat1", [(0, 0.8)], producer="equation")
        equation["source_arch"] = "mlp"  # a different source regime

        assert overlay_cells({"curve": curve, "equation": equation}) == []

    def test_empty_when_equation_producer_absent(self):
        # Curve can run without the SR pipeline, so no equation cells means no overlay.
        producers = {"curve": _rows("runA", [(0, 0.7)], producer="curve")}

        assert overlay_cells(producers) == []


class TestOverlayStats:
    """Because the join is on the regime (ADR 0015), the curve side must be
    aggregated across its seed-policies before it can be compared with the single
    distilled condition. The figure therefore asserts 'this regime's policies,
    pooled, vs their distilled form' — not one policy vs its own distillation."""

    def test_curve_side_pools_every_policy_in_the_regime(self):
        assembled = pd.concat(
            [
                _rows("wandb_run_1", [(0, 0.4), (1, 0.6)], producer="curve"),
                _rows("wandb_run_2", [(0, 0.8), (1, 1.0)], producer="curve"),
            ],
            ignore_index=True,
        )
        cell = ("eyepacs", 1.0, 200, "cnn", "mnist", 1.0, 200)

        mean, spread = overlay_stats(assembled, cell)

        # Pooled over all four rows, not averaged per-policy-then-averaged.
        assert mean == pytest.approx(0.7)
        # The spread now mixes seed noise AND across-policy spread within the
        # regime — wider than either policy's own seed spread (0.1).
        assert spread > 0.1

    def test_only_the_requested_cell_contributes(self):
        assembled = pd.concat(
            [
                _rows("wandb_run_1", [(0, 0.4)], producer="curve", target="mnist"),
                _rows("wandb_run_2", [(0, 1.0)], producer="curve", target="cifar-10"),
            ],
            ignore_index=True,
        )

        mean, _ = overlay_stats(assembled, ("eyepacs", 1.0, 200, "cnn", "mnist", 1.0, 200))

        assert mean == pytest.approx(0.4)
