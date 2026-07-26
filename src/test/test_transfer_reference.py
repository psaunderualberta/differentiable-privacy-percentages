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

        source = reference_source("Constant", target)

        assert isinstance(source, SourcePolicy)
        assert source.run_id == "Constant"
        assert source.dataset == "eyepacs"
        assert source.eps == 1.0
        assert source.delta == 1e-7
        assert source.T == 200
        assert source.arch == "cnn-32x64-head256"
        # No source run to borrow a sampling rate from.
        assert math.isnan(source.p)


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
