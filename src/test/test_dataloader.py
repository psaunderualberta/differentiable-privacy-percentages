import contextlib
import pathlib
import tempfile

import numpy as np
import pytest

from util.dataloaders import DatasetLoader, _preprocess


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


class TestDataloaderTestChunks:
    def test_loaded_test_data_is_offset_correctly(self, loader):
        loaded_x, loaded_y = loader.load_test_chunk(np.array([0]))
        expected_location = loader.n_val
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
        expected_location = 0
        expected_x, expected_y = _preprocess(
            np.load(loader.val_x_path)[expected_location : expected_location + 1],
            np.load(loader.val_y_path)[expected_location : expected_location + 1],
            "mnist",
        )

        assert np.allclose(loaded_x, expected_x, atol=1e-5)
        assert np.allclose(loaded_y, expected_y, atol=1e-5)
