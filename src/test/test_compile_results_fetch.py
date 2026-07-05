import pytest

from compile_results_fetch import DATASET_SHAPES, assert_shapes_consistent


class TestDatasetShapesConsistency:
    """DATASET_SHAPES is a fetch-side mirror of the dataloader's cached layout.

    It silently drifted once (eyepacs cached 256x256/5-class but the mirror said
    224x224/2-class); these tests pin the ground truth and guard the mirror.
    """

    def test_eyepacs_matches_cached_layout(self):
        # Ground truth: _eyepacs_download_and_cache writes (N, 3, 256, 256) uint8
        # with 5-class one-hot labels.
        assert DATASET_SHAPES["eyepacs"] == ((3, 256, 256), 5)

    def test_assert_shapes_consistent_passes_on_current_table(self):
        assert_shapes_consistent()  # should not raise

    def test_assert_shapes_consistent_catches_drift(self, monkeypatch):
        monkeypatch.setitem(DATASET_SHAPES, "eyepacs", ((3, 224, 224), 2))
        with pytest.raises(AssertionError, match="eyepacs"):
            assert_shapes_consistent()
