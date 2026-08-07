import numpy as np
import pandas as pd
import pytest

from transfer_plot import (
    nearest_source,
    overlay_cells,
    overlay_stats,
    reference_rules,
    scope_to_arm,
    source_t_profile,
    transfer_matrix,
)


def _rows(
    source_id,
    seeds_accs,
    producer="curve",
    target="mnist",
    t_eps=1.0,
    t_T=200,
    arm="sgd-m0.9",
    tuned_scale=1.0,
    tuned_constants="",
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
            "tuned_scale": tuned_scale,
            "tuned_constants": tuned_constants,
        }
    )


class TestTheMatrixReportsWhatEachCellWasTunedTo:
    """ADR 0024: a cell's accuracy is the accuracy of a *tuned* schedule, so the matrix
    has to say which knobs won it — otherwise a heatmap silently compares schedules
    adapted by different amounts, and 'which scale did each target prefer' is lost."""

    def test_a_cell_carries_the_knobs_its_rows_won_under(self):
        assembled = _rows("run-a", [(0, 0.80), (1, 0.82)], tuned_scale=0.25)

        cell = transfer_matrix(assembled).iloc[0]

        assert cell["tuned"] == "scale=0.25"

    def test_an_untuned_cell_says_so_rather_than_going_blank(self):
        assembled = _rows("run-a", [(0, 0.80)])

        assert transfer_matrix(assembled).iloc[0]["tuned"] == "scale=1"

    def test_shape_constants_are_named_alongside_the_scale(self):
        assembled = _rows(
            "cat31",
            [(0, 0.80)],
            producer="equation",
            tuned_scale=4.0,
            tuned_constants="sigma.p2=1.5",
        )

        assert transfer_matrix(assembled).iloc[0]["tuned"] == "scale=4 sigma.p2=1.5"

    def test_a_cell_pooling_differently_tuned_policies_is_flagged_not_averaged(self):
        # Stage A is tuned per (target x arm) and shared across the sources in a cell,
        # so its policies must agree. If they ever do not, the cell mean is a mean over
        # different schedules and the label has to admit it rather than pick one.
        assembled = pd.concat(
            [
                _rows("run-a", [(0, 0.80)], tuned_scale=0.25),
                _rows("run-b", [(0, 0.90)], tuned_scale=4.0),
            ],
            ignore_index=True,
        )

        assert transfer_matrix(assembled).iloc[0]["tuned"] == "mixed"

    def test_a_cell_written_before_adr_0024_still_reads_as_untuned(self):
        # The reference cells already on disk carry neither column, and ADR 0024 does
        # not invalidate them (a native reference was always tuned on its target). The
        # assembler must keep reading them rather than dying on a missing column.
        legacy = _rows("Constant", [(0, 0.80)], producer="reference").drop(
            columns=["tuned_scale", "tuned_constants"]
        )

        assert transfer_matrix(legacy).iloc[0]["tuned"] == "scale=1"


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


def _cell(
    source_dataset,
    source_eps,
    source_T,
    mean_acc,
    target="imagenet",
    target_T=7000,
    arm="sgd-m0.9",
    producer="curve",
):
    """One row of the collapsed cell matrix — the shape ``transfer_matrix`` returns and
    ``matrix_curve.csv`` stores: one source regime-arm × one target regime."""
    return {
        "producer": producer,
        "source_dataset": source_dataset,
        "source_eps": source_eps,
        "source_T": source_T,
        "source_arch": "cnn-16x32-head32",
        "source_arm": arm,
        "source_label": f"{source_dataset} e{source_eps} T{source_T}",
        "target": target,
        "target_eps": 10.0,
        "target_T": target_T,
        "mean_acc": mean_acc,
        "spread": 0.1,
        "n_policies": 4,
        "n": 12,
        "spread_of": "policies",
    }


class TestSourceTProfile:
    """ADR 0022 replaces the 63-row heatmap with a line over source T, because source T
    explains 80-91% of the within-arm variance while source dataset explains 0-1%. The
    profile is what that line and its band are read from."""

    def test_the_band_is_the_spread_across_regimes_sharing_a_source_T(self):
        # Four source regimes at one source T, differing only in provenance. The line is
        # their mean and the band is the sd ACROSS them — the "source-regime spread" of
        # CONTEXT.md, which is precisely the quantity that says provenance does not
        # matter. Averaging the cells' own ± instead would report generalization
        # consistency, a different and smaller number.
        matrix = pd.DataFrame(
            [
                _cell("mnist", 3.0, 2000, 10.0),
                _cell("mnist", 10.0, 2000, 12.0),
                _cell("fashion-mnist", 3.0, 2000, 14.0),
                _cell("fashion-mnist", 10.0, 2000, 16.0),
            ]
        )

        profile = source_t_profile(matrix)

        assert len(profile) == 1
        row = profile.iloc[0]
        assert row["mean_acc"] == pytest.approx(13.0)
        assert row["spread"] == pytest.approx(np.std([10.0, 12.0, 14.0, 16.0], ddof=0))
        assert row["n_regimes"] == 4

    def test_the_two_arms_never_pool_into_one_series(self):
        # After ADR 0021 the arms differ in their *target* configuration, so their
        # accuracies are not on a common scale — pooling is a units error, not merely a
        # confounding one. The arm is a grouping key rather than a pre-filter so that
        # a caller who forgets to filter still cannot silently average the two.
        matrix = pd.DataFrame(
            [
                _cell("mnist", 3.0, 2000, 17.0, arm="sgd-m0.9"),
                _cell("mnist", 3.0, 2000, 3.0, arm="sgd-m0.0"),
            ]
        )

        profile = source_t_profile(matrix)

        assert len(profile) == 2
        assert dict(zip(profile["source_arm"], profile["mean_acc"])) == {
            "sgd-m0.9": 17.0,
            "sgd-m0.0": 3.0,
        }

    def test_splitting_by_source_dataset_gives_the_overplotted_series(self):
        # Source dataset explains 0-1% of the variance, and the figure makes that
        # visible by drawing each provenance as its own marker at the same x. So the
        # profile has to be reducible along that axis without changing anything else.
        matrix = pd.DataFrame(
            [
                _cell("mnist", 3.0, 2000, 10.0),
                _cell("mnist", 10.0, 2000, 12.0),
                _cell("fashion-mnist", 3.0, 2000, 20.0),
            ]
        )

        profile = source_t_profile(matrix, split=("source_dataset",))

        by_dataset = dict(zip(profile["source_dataset"], profile["mean_acc"]))
        assert by_dataset == {"mnist": pytest.approx(11.0), "fashion-mnist": 20.0}


def _ref(mechanism, mean_acc, target="imagenet", target_T=7000, arm="sgd-m0.9"):
    """One native-reference cell, as ``matrix_reference.csv`` stores it."""
    row = _cell(target, 10.0, target_T, mean_acc, target, target_T, arm, producer="reference")
    row["source_label"] = mechanism
    return row


class TestReferenceRules:
    """ADR 0022 draws TWO reference rules, because the choice of rule changes the claim.
    Against Constant alone the method wins everywhere by up to +11.4pp; against the best
    of the three it wins on ImageNet-32 and ties on CheXpert. One rule alone would
    either overstate the result or hide that the *adaptive* baselines are the hard ones."""

    def test_the_best_of_three_is_named_not_just_valued(self):
        # The winning mechanism differs per panel (Dynamic at 4 of 6, Median at 2), so
        # the rule is annotated rather than merely drawn — an unnamed best-of-3 line
        # silently changes what it means between facets.
        matrix = pd.DataFrame(
            [
                _ref("Constant", 5.57),
                _ref("Dynamic-DPSGD", 15.17),
                _ref("Median", 12.17),
            ]
        )

        rules = reference_rules(matrix)

        assert len(rules) == 1
        row = rules.iloc[0]
        assert row["best_mechanism"] == "Dynamic-DPSGD"
        assert row["best_acc"] == pytest.approx(15.17)

    def test_constant_is_reported_even_when_it_is_not_the_best(self):
        # The faint dashed rule. Constant sits 6-9pp below the adaptive references on
        # ImageNet-32, which is itself the interesting statement; collapsing to
        # best-of-3 would delete it.
        matrix = pd.DataFrame(
            [_ref("Constant", 5.57), _ref("Dynamic-DPSGD", 15.17), _ref("Median", 12.17)]
        )

        row = reference_rules(matrix).iloc[0]

        assert row["constant_acc"] == pytest.approx(5.57)

    def test_each_arm_gets_its_own_bar(self):
        # ADR 0021: a reference is native to its target, so an m=0.9-tuned one is not a
        # baseline for an m=0.0 target. The rules must therefore be per arm, or one
        # arm's panel would be drawn against the other's bar.
        matrix = pd.DataFrame(
            [
                _ref("Constant", 5.5, arm="sgd-m0.9"),
                _ref("Dynamic-DPSGD", 15.2, arm="sgd-m0.9"),
                _ref("Constant", 2.1, arm="sgd-m0.0"),
                _ref("Dynamic-DPSGD", 4.4, arm="sgd-m0.0"),
            ]
        )

        rules = reference_rules(matrix)

        assert dict(zip(rules["source_arm"], rules["best_acc"])) == {
            "sgd-m0.9": pytest.approx(15.2),
            "sgd-m0.0": pytest.approx(4.4),
        }

    def test_an_arm_with_no_references_yet_yields_an_empty_frame_not_a_crash(self):
        # The real state between the arm fix and the ADR 0021 re-run: sgd-m0.0's curve
        # cells exist but its references have not been swept. Its panel must draw with
        # no bars — a figure that renders without a baseline is honest; one that raises
        # takes the *other* arm's finished figure down with it.
        rules = reference_rules(pd.DataFrame(columns=list(_ref("Constant", 1.0))))

        assert rules.empty
        assert "best_acc" in rules.columns and "source_arm" in rules.columns


class TestScopeToArm:
    """One figure per arm (ADR 0022), so everything drawn on it has to be narrowed to
    that arm first — and two known-bad row groups dropped rather than plotted."""

    def test_only_the_requested_arms_rows_survive(self):
        matrix = pd.DataFrame(
            [
                _cell("mnist", 3.0, 2000, 17.0, arm="sgd-m0.9"),
                _cell("mnist", 3.0, 2000, 3.0, arm="sgd-m0.0"),
            ]
        )

        scoped = scope_to_arm(matrix, "sgd-m0.9")

        assert list(scoped["mean_acc"]) == [17.0]

    def test_eyepacs_is_dropped(self):
        # ADR 0020 dropped EyePACS as a target for having no schedule-resolving power:
        # it floors at 73.982% even non-privately, so every schedule scores the same
        # there. One stray cell survives in the cache; it is filtered, not plotted.
        matrix = pd.DataFrame(
            [
                _cell("mnist", 3.0, 2000, 17.0, target="imagenet"),
                _cell("mnist", 3.0, 2000, 74.0, target="eyepacs"),
            ]
        )

        scoped = scope_to_arm(matrix, "sgd-m0.9")

        assert set(scoped["target"]) == {"imagenet"}

    def test_rows_predating_the_arm_column_are_dropped_not_silently_grouped(self):
        # Three cells come from a parquet older than ADR 0011 and carry no arm. NaN
        # never compares equal to itself, so they would neither join nor group — they
        # would just appear as a fourth phantom series.
        rows = [_cell("mnist", 3.0, 2000, 17.0), _cell("mnist", 5.0, 2000, 9.0)]
        rows[1]["source_arm"] = float("nan")
        matrix = pd.DataFrame(rows)

        scoped = scope_to_arm(matrix, "sgd-m0.9")

        assert list(scoped["mean_acc"]) == [17.0]

    def test_legacy_armless_references_are_attributed_to_the_momentum_they_ran_at(self):
        # The 18 reference cells written before ADR 0021 carry arm="" — but the bug
        # pinned *every* target to momentum 0.9, so they are provably the sgd-m0.9
        # arm's references. Attributing them is what makes the matched half of the
        # existing batch plottable before the re-run lands; the other arm gets nothing
        # rather than a borrowed bar.
        matrix = pd.DataFrame(
            [_ref("Constant", 5.57, arm=""), _ref("Dynamic-DPSGD", 15.17, arm="")]
        )

        assert len(scope_to_arm(matrix, "sgd-m0.9", legacy_arm="sgd-m0.9")) == 2
        assert len(scope_to_arm(matrix, "sgd-m0.0", legacy_arm="sgd-m0.9")) == 0
