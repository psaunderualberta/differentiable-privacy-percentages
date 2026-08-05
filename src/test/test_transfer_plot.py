import pandas as pd
import pytest

from transfer_plot import nearest_source, overlay_cells, overlay_stats, transfer_matrix


def _rows(
    source_id,
    seeds_accs,
    producer="curve",
    target="mnist",
    t_eps=1.0,
    t_T=200,
    arm="sgd-m0.9",
):
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
            "source_arm": arm,
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
    """The descriptive matrix is read off, not selected (ADR 0008), and its row unit
    is the source **regime-arm** (ADR 0018): a cell pools every policy in the regime
    so its ± is generalization consistency — the spread across source policies —
    rather than evaluation noise, the spread across one policy's own reps."""

    def test_a_regimes_policies_pool_into_one_cell_whose_spread_is_across_them(self):
        # Two policies of the same regime-arm, each internally consistent. The cell
        # must report a spread driven by the gap BETWEEN them, which a per-policy
        # grouping would have reported as zero.
        assembled = pd.concat(
            [
                _rows("runA", [(0, 0.80), (1, 0.80)]),
                _rows("runB", [(0, 0.60), (1, 0.60)]),
            ],
            ignore_index=True,
        )

        matrix = transfer_matrix(assembled)

        # One row: one regime-arm × one target, not one row per policy.
        assert len(matrix) == 1
        (cell,) = matrix.to_dict("records")
        assert cell["mean_acc"] == pytest.approx(0.70)
        assert cell["n"] == 4
        # Both policies are internally identical, so any spread here is across them.
        assert cell["spread"] == pytest.approx(0.10)

    def test_the_cell_spread_excludes_each_policys_own_evaluation_noise(self):
        # Generalization consistency is the spread BETWEEN policies. Two policies
        # whose means are identical have zero of it, however noisy each policy's own
        # reps are — pooling raw rows would report that rep noise as the regime's
        # consistency, which is the quantity CONTEXT.md says the matrix must not mix.
        assembled = pd.concat(
            [
                _rows("runA", [(0, 0.5), (1, 0.9)]),  # mean 0.70, noisy
                _rows("runB", [(0, 0.6), (1, 0.8)]),  # mean 0.70, less noisy
            ],
            ignore_index=True,
        )

        (cell,) = transfer_matrix(assembled).to_dict("records")

        assert cell["mean_acc"] == pytest.approx(0.70)
        assert cell["spread"] == pytest.approx(0.0)
        assert cell["spread_of"] == "policies"
        assert cell["n_policies"] == 2
        assert cell["n"] == 4

    def test_a_single_unit_row_reports_its_rep_spread_instead(self):
        # A native reference has no regime — its row unit is the reference itself
        # (CONTEXT.md), so there are no sibling policies to spread across and the
        # honest ± is its evaluation noise. Falling through to zero would claim a
        # precision the eight reps do not have.
        assembled = _rows("Constant", [(0, 0.60), (1, 0.80)], producer="reference")

        (cell,) = transfer_matrix(assembled).to_dict("records")

        assert cell["mean_acc"] == pytest.approx(0.70)
        assert cell["spread"] == pytest.approx(0.10)
        assert cell["spread_of"] == "reps"
        assert cell["n_policies"] == 1

    def test_the_two_arms_of_one_regime_stay_separate_rows(self):
        # Pooling the arms would report an 8.5x median-sigma difference (ADR 0016) as
        # this regime's generalization consistency.
        assembled = pd.concat(
            [
                _rows("m09", [(0, 0.80)], arm="sgd-m0.9"),
                _rows("m00", [(0, 0.40)], arm="sgd-m0.0"),
            ],
            ignore_index=True,
        )

        matrix = transfer_matrix(assembled)

        assert sorted(matrix["source_arm"]) == ["sgd-m0.0", "sgd-m0.9"]
        assert sorted(matrix["mean_acc"]) == [pytest.approx(0.40), pytest.approx(0.80)]

    def test_the_three_references_stay_three_rows(self):
        # A native reference's source_* provenance mirrors its TARGET (there is no
        # source run), so all three share one degenerate "regime" and pooling on it
        # alone would silently average Constant, Dynamic-DPSGD and Median into a
        # single comparison row. A reference is a mechanism, not a regime (CONTEXT.md):
        # its source_id IS its row unit.
        assembled = pd.concat(
            [
                _rows("Constant", [(0, 0.50)], producer="reference", arm=""),
                _rows("Dynamic-DPSGD", [(0, 0.60)], producer="reference", arm=""),
                _rows("Median", [(0, 0.70)], producer="reference", arm=""),
            ],
            ignore_index=True,
        )

        matrix = transfer_matrix(assembled)

        assert len(matrix) == 3
        assert sorted(matrix["source_label"]) == ["Constant", "Dynamic-DPSGD", "Median"]

    def test_a_curve_cells_label_names_its_regime_arm_not_a_run_id(self):
        # The row unit is the regime-arm, so the label must be readable as one — a
        # W&B run id would name only one of the policies pooled into the cell.
        assembled = _rows("wandb_run_1", [(0, 0.7)], producer="curve", arm="sgd-m0.9")

        (label,) = transfer_matrix(assembled)["source_label"]

        assert "wandb_run_1" not in label
        for part in ("eyepacs", "sgd-m0.9", "200"):
            assert part in label


class TestPolicyMatrixIsEvaluationNoise:
    """The per-policy view survives alongside the pooled one, because the two ± are
    different quantities and CONTEXT.md requires naming which one a bar shows: here
    the spread is across one policy's evaluation reps — DP-SGD's own run-to-run
    variance — not across the regime's policies."""

    def test_every_source_policy_is_its_own_row_with_its_own_rep_spread(self):
        from transfer_plot import policy_matrix

        assembled = pd.concat(
            [
                _rows("runA", [(0, 0.80), (1, 0.90)]),
                _rows("runB", [(0, 0.60), (1, 0.60)]),
            ],
            ignore_index=True,
        )

        matrix = policy_matrix(assembled).set_index("source_id")

        assert set(matrix.index) == {"runA", "runB"}
        assert matrix.loc["runA", "mean_acc"] == pytest.approx(0.85)
        assert matrix.loc["runA", "n"] == 2
        assert matrix.loc["runA", "spread"] > 0.0
        # runB's reps agree, so its evaluation noise is zero.
        assert matrix.loc["runB", "spread"] == 0.0


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
        assert cells == [("eyepacs", 1.0, 200, "cnn", "sgd-m0.9", "mnist", 1.0, 200)]

    def test_a_curve_regime_from_the_other_arm_never_overlays(self):
        # Syntheses are scoped to one arm (ADR 0016), so the m0.0 sources have no
        # equation counterpart. Without the arm in the join key they would silently
        # overlay the m0.9 synthesis — comparing a curve against a closed form
        # distilled from a different arm's shapes entirely.
        producers = {
            "curve": _rows("wandb_m00", [(0, 0.7)], producer="curve", arm="sgd-m0.0"),
            "equation": _rows(
                "eyepacs_eps1_T200_cnn_cat1", [(0, 0.8)], producer="equation", arm="sgd-m0.9"
            ),
        }

        assert overlay_cells(producers) == []

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
        cell = ("eyepacs", 1.0, 200, "cnn", "sgd-m0.9", "mnist", 1.0, 200)

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

        mean, _ = overlay_stats(
            assembled, ("eyepacs", 1.0, 200, "cnn", "sgd-m0.9", "mnist", 1.0, 200)
        )

        assert mean == pytest.approx(0.4)
