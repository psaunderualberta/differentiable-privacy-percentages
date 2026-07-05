import contextlib
import pathlib
import tempfile

import numpy as np
import pytest

from util.dataloaders import (
    DatasetLoader,
    _chexpert_binary_onehot,
    _chexpert_frontal_mask,
    _get_sample_shape,
    _imagenet100_select,
    _preprocess,
)


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@pytest.fixture
def datasets():
    with temp_seed(42):
        # 1-channel
        n_train = 100
        n_val = 20
        sample_shape = (1, 28, 28)
        label_shape = (1,)
        X_train = np.random.random((n_train, *sample_shape[1:]))
        Y_train = np.random.random((n_train, *label_shape))
        X_val = np.random.random((n_val, *sample_shape[1:]))
        Y_val = np.random.random((n_train, *label_shape))

        yield X_train, Y_train, X_val, Y_val


@pytest.fixture
def loader(datasets):
    X_train, Y_train, X_val, Y_val = datasets
    with tempfile.TemporaryDirectory() as d:
        train_path = pathlib.Path(d) / "train.npy"
        train_path_labels = pathlib.Path(d) / "train-labels.npy"
        val_path = pathlib.Path(d) / "val.npy"
        val_path_labels = pathlib.Path(d) / "val-labels.npy"

        np.save(train_path, X_train)
        np.save(val_path, X_val)
        np.save(train_path_labels, Y_train)
        np.save(val_path_labels, Y_val)
        n_val = X_val.shape[0] // 2
        n_test = X_val.shape[0] // 2

        yield DatasetLoader(
            x_path=str(train_path),
            y_path=str(train_path_labels),
            val_x_path=str(val_path),
            val_y_path=str(val_path_labels),
            n_train=X_train.shape[0],
            n_val=n_val,
            n_test=n_test,
            sample_shape=X_train.shape[1:],
            label_shape=Y_train.shape[1:],
            dataset_name="mnist",
            val_chunk_size=5,
        )


class TestNewTargetPreprocessing:
    """CheXpert (grayscale CHW) and ImageNet-32 (HWC→CHW) surrogate targets."""

    def test_chexpert_preprocess_keeps_chw_and_normalizes(self):
        # CheXpert mirrors eyepacs: cached channels-first uint8, 1 grayscale channel.
        x_raw = np.full((3, 1, 64, 64), 255, dtype=np.uint8)
        y_raw = np.eye(2, dtype=np.float32)[[0, 1, 0]]
        x, _ = _preprocess(x_raw, y_raw, "chexpert")
        assert x.shape == (3, 1, 64, 64)
        assert x.dtype == np.float32
        assert np.allclose(x, 1.0)

    def test_imagenet_preprocess_transposes_hwc_to_chw(self):
        # ImageNet-32 cached HWC uint8 like cifar-10.
        x_raw = np.zeros((3, 32, 32, 3), dtype=np.uint8)
        y_raw = np.eye(100, dtype=np.float32)[[0, 1, 2]]
        x, _ = _preprocess(x_raw, y_raw, "imagenet")
        assert x.shape == (3, 3, 32, 32)
        assert x.dtype == np.float32

    def test_chexpert_sample_shape_unchanged(self):
        assert _get_sample_shape((1, 64, 64), "chexpert") == (1, 64, 64)

    def test_imagenet_sample_shape_hwc_to_chw(self):
        assert _get_sample_shape((32, 32, 3), "imagenet") == (3, 32, 32)


class TestChexpertLabels:
    """U-Zeros convention: only an explicit positive (1.0) counts as positive."""

    def test_u_zeros_maps_positive_only(self):
        # Values: positive, negative, uncertain(-1), blank(NaN)
        col = np.array([1.0, 0.0, -1.0, np.nan])
        onehot = _chexpert_binary_onehot(col)
        # column 1 is the positive class
        assert onehot.shape == (4, 2)
        assert onehot.dtype == np.float32
        np.testing.assert_array_equal(onehot[:, 1], [1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(onehot[:, 0], [0.0, 1.0, 1.0, 1.0])

    def test_frontal_mask_selects_frontal_only(self):
        col = np.array(["Frontal", "Lateral", "Frontal"])
        np.testing.assert_array_equal(_chexpert_frontal_mask(col), [True, False, True])


class TestImagenet100Subset:
    """Filter the 1000-class source to the published 100-wnid subset and remap labels."""

    def test_select_masks_and_remaps_to_subset_order(self):
        sample_ids = np.array(["a", "c", "b", "d", "a"])
        subset = ["a", "b", "c"]  # labels remap to this order: a→0, b→1, c→2
        mask, labels = _imagenet100_select(sample_ids, subset)
        np.testing.assert_array_equal(mask, [True, True, True, False, True])
        # labels are given only for the selected rows, in original order
        np.testing.assert_array_equal(labels, [0, 2, 1, 0])


class TestDataloaderTestChunks:
    def test_loaded_test_data_is_offset_correctly(self, loader):
        loaded_x, loaded_y = loader.load_test_chunk(np.array([0]))
        # Test chunk index 0 maps to the (n_val)-th entry of the permuted pool.
        expected_location = int(loader._val_test_perm()[loader.n_val])
        expected_x, expected_y = _preprocess(
            np.load(loader.val_x_path)[expected_location : expected_location + 1],
            np.load(loader.val_y_path)[expected_location : expected_location + 1],
            "mnist",
        )

        assert np.allclose(loaded_x, expected_x, atol=1e-5)
        assert np.allclose(loaded_y, expected_y, atol=1e-5)

    def test_n_val_plus_n_test_gt_nrows_throws_value_error(self, loader):
        with pytest.raises(ValueError, match="must be <="):
            DatasetLoader(
                x_path=loader.x_path,
                y_path=loader.y_path,
                val_x_path=loader.val_x_path,
                val_y_path=loader.val_y_path,
                n_train=loader.n_train,
                n_val=loader.n_val,
                n_test=100,
                sample_shape=loader.sample_shape,
                label_shape=loader.label_shape,
                dataset_name=loader.dataset_name,
                val_chunk_size=loader.val_chunk_size,
            )


class TestDataloaderValChunks:
    def test_loaded_val_data_is_offset_correctly(self, loader):
        loaded_x, loaded_y = loader.load_val_chunk(np.array([0]))
        # Val chunk index 0 maps to the 0-th entry of the permuted pool.
        expected_location = int(loader._val_test_perm()[0])
        expected_x, expected_y = _preprocess(
            np.load(loader.val_x_path)[expected_location : expected_location + 1],
            np.load(loader.val_y_path)[expected_location : expected_location + 1],
            "mnist",
        )

        assert np.allclose(loaded_x, expected_x, atol=1e-5)
        assert np.allclose(loaded_y, expected_y, atol=1e-5)

    def test_val_and_test_indices_are_disjoint(self, loader):
        perm = loader._val_test_perm()
        val_rows = set(perm[np.arange(loader.n_val)].tolist())
        test_rows = set(perm[np.arange(loader.n_test) + loader.n_val].tolist())
        assert val_rows.isdisjoint(test_rows)
