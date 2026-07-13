import pandas as pd
import pytest

from transfer_plot import nearest_source, overlay_cells, transfer_matrix


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
    """The curve-vs-equation overlay is drawn for a cell only when BOTH producers
    have a record for it (ADR 0008) — a presence-check join, the only place the
    compare-when-both-exist rule lives. Reference cells never participate."""

    def test_only_cells_present_in_both_curve_and_equation(self):
        producers = {
            "curve": pd.concat(
                [
                    _rows("shared", [(0, 0.7)], producer="curve", target="mnist"),
                    _rows("curve_only", [(0, 0.7)], producer="curve", target="mnist"),
                ],
                ignore_index=True,
            ),
            "equation": pd.concat(
                [
                    _rows("shared", [(0, 0.8)], producer="equation", target="mnist"),
                    _rows("eqn_only", [(0, 0.8)], producer="equation", target="mnist"),
                ],
                ignore_index=True,
            ),
            # A reference producer is present but must never enter the overlay set.
            "reference": _rows("Constant", [(0, 0.6)], producer="reference", target="mnist"),
        }

        cells = overlay_cells(producers)

        assert cells == [("shared", "mnist", 1.0, 200)]

    def test_empty_when_equation_producer_absent(self):
        # Curve can run without the SR pipeline, so no equation cells means no overlay.
        producers = {"curve": _rows("runA", [(0, 0.7)], producer="curve")}

        assert overlay_cells(producers) == []
