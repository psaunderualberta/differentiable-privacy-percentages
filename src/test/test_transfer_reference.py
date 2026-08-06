import contextlib
import math

import pandas as pd

from transfer_reference import baseline_data_to_results, reference_slugs, reference_source
from util.transfer import SourcePolicy, TargetSpec


class TestBaselineDataToResults:
    """Split Baseline.generate_baseline_data's multi-regime df into per-regime results.

    generate_baseline_data concatenates the three native references into one frame,
    tagged by the ``type`` column. The producer regroups them into a clean-slug →
    ``(seed, accuracy, loss)`` mapping; each regime's reps are seed-indexed 0..N-1.
    """

    def test_splits_multi_regime_df_into_per_regime_seed_results(self):
        df = pd.DataFrame(
            {
                "type": [
                    "Constant σ/clip",
                    "Constant σ/clip",
                    "Dynamic-DPSGD",
                    "Adaptive Clip (Andrew et al.)",
                    "Adaptive Clip (Andrew et al.)",
                ],
                "step": [0] * 5,
                "loss": [0.4, 0.5, 0.3, 0.6, 0.55],
                "accuracy": [0.8, 0.78, 0.85, 0.7, 0.72],
                "losses": [[]] * 5,
                "accuracies": [[]] * 5,
            }
        )

        results = baseline_data_to_results(df)

        # The three native regimes get clean, stable slugs.
        assert set(results) == {"Constant", "Dynamic-DPSGD", "Median"}
        # Reps within a regime are seed-indexed in order; accuracy/loss carried through.
        assert results["Constant"] == [(0, 0.8, 0.4), (1, 0.78, 0.5)]
        assert results["Dynamic-DPSGD"] == [(0, 0.85, 0.3)]
        assert results["Median"] == [(0, 0.7, 0.6), (1, 0.72, 0.55)]


class TestRegimeSlugsAreFilesystemSafe:
    """Regime slugs become a cell's source_id, which write_transfer_cell embeds in a
    filename. The raw ``type`` strings carry a path separator and whitespace ("Constant
    σ/clip", "Adaptive Clip (Andrew et al.)"), so every slug must be path-safe."""

    def test_no_slug_contains_a_path_separator_or_whitespace(self):
        for slug in reference_slugs():
            assert "/" not in slug
            assert not any(ch.isspace() for ch in slug)


class TestReferenceSource:
    """A native reference has no learned source; its SourcePolicy IS the target regime.

    So the source_* provenance mirrors the target (dataset, eps, delta, T, arch), with
    the regime slug as the run_id and p unknown (NaN) — there is no source run to read
    a sampling rate from."""

    def test_source_policy_mirrors_the_target_regime(self):
        target = TargetSpec(name="eyepacs", eps=1.0, delta=1e-7, T=200, arch="cnn-32x64-head256")

        source = reference_source("Constant", target, arm="sgd-m0.9")

        assert isinstance(source, SourcePolicy)
        assert source.run_id == "Constant"
        assert source.dataset == "eyepacs"
        assert source.eps == 1.0
        assert source.delta == 1e-7
        assert source.T == 200
        assert source.arch == "cnn-32x64-head256"
        # No source run to borrow a sampling rate from.
        assert math.isnan(source.p)

    def test_a_reference_carries_the_target_momentum_it_was_tuned_at(self):
        # ADR 0021 widens the arm from "which arm the source was learned in" to a
        # property of the whole transfer. A reference is swept and evaluated natively
        # on the target, so its arm is the target's momentum — and it must be recorded,
        # because an m=0.9-tuned reference is not a baseline for an m=0.0 target.
        target = TargetSpec(name="eyepacs", eps=1.0, delta=1e-7, T=200, arch="cnn")

        assert reference_source("Constant", target, arm="sgd-m0.0").arm == "sgd-m0.0"


@contextlib.contextmanager
def _baseline(T=8):
    """A Baseline with real privacy params but no env: enough to build candidates.

    Wrapped in both config scopes — the stateful median-gradient schedule reads
    ``batch_size`` off the singleton at construction, and outside a scope that
    re-parses ``sys.argv`` (which under pytest is the pytest command line).
    """
    from jax import random as jr

    from conf.scope import RunContext, using
    from conf.singleton_conf import SingletonConfig
    from privacy.gdp_privacy import GDPPrivacyParameters
    from util.baselines import Baseline
    from util.transfer import TargetSpec, build_target_config

    config = build_target_config(
        TargetSpec(name="mnist", eps=1.0, delta=1e-5, T=T, arch=""), 250, arm="sgd-m0.9"
    )
    privacy_params = GDPPrivacyParameters(eps=1.0, delta=1e-5, p=0.01, T=T)
    with SingletonConfig.override(config), using(RunContext(config)):
        yield Baseline(None, privacy_params, jr.PRNGKey(0), num_reps=3)


class TestCandidateEnumeration:
    """ADR 0019 splits the reference sweep into one SLURM task per candidate, so the
    candidate a task builds must be a pure function of (reference, master key, index)
    — identical to the one the monolithic 20-candidate sweep would have evaluated at
    that position. Otherwise splitting the job changes which hyperparameters are
    searched, and the references stop being the honestly-tuned comparison the
    chapter's claim rests on."""

    def _drawn(self, schedule):
        """The hyperparameters a candidate was drawn with.

        Compared rather than the σ/clip arrays because a runtime-adaptive reference
        (StatefulMedianGradient) has no schedule arrays until it is trained — what
        the sweep searches over, and what a split task must reproduce, is the draw.
        """
        import numpy as np

        from util.baselines import describe_schedule

        return {k: float(np.asarray(v)) for k, v in describe_schedule(schedule).items()}

    def test_a_candidate_is_the_same_whether_you_enumerate_20_or_just_reach_it(self):
        from jax import random as jr

        from util.baselines import REFERENCES

        with _baseline() as baseline:
            for reference in REFERENCES:
                full = baseline.candidate_schedules(reference, jr.PRNGKey(4), num_candidates=20)
                short = baseline.candidate_schedules(reference, jr.PRNGKey(4), num_candidates=8)

                assert len(full) == 20 and len(short) == 8
                for index in range(8):
                    assert self._drawn(full[index]) == self._drawn(short[index]), (
                        f"{reference} candidate {index}"
                    )

    def test_candidates_actually_differ_from_one_another(self):
        # A search that drew the same point 20 times would satisfy the test above
        # while searching nothing at all.
        from jax import random as jr

        from util.baselines import REFERENCES

        with _baseline() as baseline:
            for reference in REFERENCES:
                drawn = [
                    self._drawn(schedule)
                    for schedule in baseline.candidate_schedules(
                        reference, jr.PRNGKey(4), num_candidates=4
                    )
                ]
                assert drawn[0] != drawn[1], reference


class TestCandidateRecords:
    """A per-candidate score is an *intermediate* artifact, not a transfer cell (ADR
    0019). Only the selector's output carries ``producer="reference"``; if a candidate
    row reached the assembler it would appear in the matrix as 19 extra reference
    columns of deliberately under-evaluated schedules."""

    def _target(self):
        return TargetSpec(name="eyepacs", eps=8.0, delta=1e-7, T=5000, arch="cnn")

    def test_a_record_round_trips_for_its_own_reference_and_target(self, tmp_path):
        from util.transfer import read_candidate_records, write_candidate_record

        target = self._target()
        write_candidate_record("Constant", target, 3, mean_accuracy=0.61, n=3, cache_root=tmp_path)
        write_candidate_record("Constant", target, 7, mean_accuracy=0.55, n=3, cache_root=tmp_path)
        # A different reference on the same target must not leak into the read.
        write_candidate_record("Median", target, 3, mean_accuracy=0.90, n=3, cache_root=tmp_path)

        records = read_candidate_records("Constant", target, tmp_path)

        assert [(r["candidate"], r["mean_accuracy"]) for r in records] == [(3, 0.61), (7, 0.55)]

    def test_the_two_arms_sweeps_stay_separate(self, tmp_path):
        """ADR 0021 re-runs each reference's sweep at the new target momentum, so a
        record belongs to one arm. Sharing a filename would both lose one arm's score
        and hand the selector a pool it thinks is a single sweep."""
        from util.transfer import read_candidate_records, write_candidate_record

        target = self._target()
        write_candidate_record("Constant", target, 3, 0.61, 3, tmp_path, arm="sgd-m0.9")
        write_candidate_record("Constant", target, 3, 0.42, 3, tmp_path, arm="sgd-m0.0")

        m09 = read_candidate_records("Constant", target, tmp_path, arm="sgd-m0.9")
        m00 = read_candidate_records("Constant", target, tmp_path, arm="sgd-m0.0")

        assert [r["mean_accuracy"] for r in m09] == [0.61]
        assert [r["mean_accuracy"] for r in m00] == [0.42]

    def test_candidate_records_are_invisible_to_the_assembler(self, tmp_path):
        from transfer_plot import load_producers
        from util.transfer import write_candidate_record

        write_candidate_record("Constant", self._target(), 0, 0.61, 3, tmp_path)

        assert load_producers(tmp_path) == {}


class TestCandidateSelection:
    """The selector picks the sweep winner off the per-candidate scores and then runs
    the final evaluation itself (ADR 0019). Selection is by 3-run mean — noisier than
    the old 10-run mean, but unbiased."""

    def test_the_highest_mean_accuracy_wins(self):
        from transfer_reference import select_candidate

        records = [
            {"candidate": 0, "mean_accuracy": 0.55},
            {"candidate": 1, "mean_accuracy": 0.71},
            {"candidate": 2, "mean_accuracy": 0.60},
        ]

        assert select_candidate(records) == 1

    def test_ties_break_on_the_lowest_candidate_index(self):
        # Tasks finish in arbitrary order, so the winner must not depend on which
        # score file happened to be written first.
        from transfer_reference import select_candidate

        records = [
            {"candidate": 5, "mean_accuracy": 0.71},
            {"candidate": 2, "mean_accuracy": 0.71},
        ]

        assert select_candidate(records) == 2

    def test_selecting_with_no_scores_is_an_error_not_a_default_winner(self):
        # A selector whose candidates all died would otherwise silently report
        # candidate 0 — an untuned reference — as the tuned baseline.
        import pytest

        from transfer_reference import select_candidate

        with pytest.raises(SystemExit):
            select_candidate([])


class TestReferenceSweepKeys:
    """Each native reference is swept in its own SLURM job, so its PRNG key must be
    the one it would have received inside the combined three-reference sweep —
    otherwise splitting the job changes the results it produces."""

    def test_each_reference_gets_its_key_from_the_original_sequential_split(self):
        from jax import random as jr

        from util.baselines import reference_sweep_keys

        # The order generate_baseline_data has always split in: median, dynamic,
        # then constant, each from the running key.
        key = jr.PRNGKey(0)
        expected = {}
        for name in ("Adaptive Clip (Andrew et al.)", "Dynamic-DPSGD", "Constant σ/clip"):
            key, sweep_key = jr.split(key)
            expected[name] = sweep_key

        keys = reference_sweep_keys(jr.PRNGKey(0))

        assert set(keys) == set(expected)
        for name, expected_key in expected.items():
            assert (keys[name] == expected_key).all(), name

    def test_selecting_one_reference_does_not_change_its_key(self):
        from jax import random as jr

        from util.baselines import reference_sweep_keys

        # The dict is a pure function of the master key, so a job that sweeps only
        # Dynamic-DPSGD uses exactly the key the full sweep would have given it.
        all_keys = reference_sweep_keys(jr.PRNGKey(3))
        assert (
            all_keys["Dynamic-DPSGD"] == reference_sweep_keys(jr.PRNGKey(3))["Dynamic-DPSGD"]
        ).all()
