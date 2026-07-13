import math

import pandas as pd

from transfer_reference import baseline_data_to_results, reference_source, regime_slugs
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
                    "Clip to Median Gradient Norm",
                    "Clip to Median Gradient Norm",
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
    σ/clip", "Clip to Median Gradient Norm"), so every slug must be path-safe."""

    def test_no_slug_contains_a_path_separator_or_whitespace(self):
        for slug in regime_slugs():
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
