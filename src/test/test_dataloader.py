import contextlib
import pathlib
import tempfile

import numpy as np
import pytest

from conf import scope
from conf.config import Config, EnvConfig, ScheduleOptimizerConfig, SweepConfig, WandbConfig
from conf.singleton_conf import SingletonConfig
from util import dataloaders
from util.dataloaders import (
    _IMAGENET100_NAMES,
    _IMAGENET100_WNIDS,
    DatasetLoader,
    _chexpert_binary_onehot,
    _chexpert_frontal_mask,
    _chexpert_relpath,
    _get_sample_shape,
    _imagenet100_select,
    _preprocess,
    dataset_dir,
    get_dataset_loader,
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

    def test_relpath_strips_the_archive_root(self):
        """train.csv roots every Path at the archive dir, but the Kaggle mirror
        (`ashery/chexpert`) unzips `train/` straight into datadir — so the CSV path
        joined onto datadir points at nothing."""
        assert (
            _chexpert_relpath("CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg")
            == "train/patient00001/study1/view1_frontal.jpg"
        )
        assert (
            _chexpert_relpath("CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg")
            == "valid/patient64541/study1/view1_frontal.jpg"
        )

    def test_relpath_leaves_an_already_relative_path_alone(self):
        """The official archive layout must keep working, so this is idempotent."""
        p = "train/patient00001/study1/view1_frontal.jpg"
        assert _chexpert_relpath(p) == p == _chexpert_relpath(_chexpert_relpath(p))

    def test_relpath_passes_through_an_unrecognised_layout(self):
        """No split component to anchor on — better to fail loudly at open() with the
        original path than to silently mangle it."""
        assert _chexpert_relpath("some/other/thing.jpg") == "some/other/thing.jpg"


class TestImagenet100Subset:
    """Filter the 1000-class source to the published 100-wnid subset and remap labels."""

    def test_select_masks_and_remaps_to_subset_order(self):
        sample_ids = np.array(["a", "c", "b", "d", "a"])
        subset = ["a", "b", "c"]  # labels remap to this order: a→0, b→1, c→2
        mask, labels = _imagenet100_select(sample_ids, subset)
        np.testing.assert_array_equal(mask, [True, True, True, False, True])
        # labels are given only for the selected rows, in original order
        np.testing.assert_array_equal(labels, [0, 2, 1, 0])

    def test_subset_names_align_index_for_index_with_wnids(self):
        """The available ImageNet-32 mirrors label by synset *name*, not wnid, so
        selection goes through ``_IMAGENET100_NAMES``. The wnid list stays the
        citable identity of the subset (ADR 0007), which only holds if the two are
        the same classes in the same order — a silent misalignment would relabel
        every sample rather than fail."""
        assert len(_IMAGENET100_NAMES) == len(_IMAGENET100_WNIDS) == 100
        assert len(set(_IMAGENET100_NAMES)) == 100
        # Spot-check the published CMC head: n02869837=bonnet, n01749939=green mamba,
        # n02488291=langur, n02107142=Doberman.
        assert _IMAGENET100_WNIDS[:4] == ["n02869837", "n01749939", "n02488291", "n02107142"]
        assert _IMAGENET100_NAMES[:4] == [
            "bonnet, poke bonnet",
            "green mamba",
            "langur",
            "Doberman, Doberman pinscher",
        ]


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


@pytest.fixture
def roots(monkeypatch, tmp_path):
    """Stand-ins for the two dataset roots: the repo's ``src/data`` and ``$SCRATCH``.

    The real ``src/data`` is whatever the developer happens to have downloaded, so
    resolution tests must not read it — the size rule and the already-cached rule
    would give different answers on different machines.
    """
    home, scratch = tmp_path / "repo-data", tmp_path / "scratch"
    monkeypatch.setattr(dataloaders, "__DATA_DIR", str(home))
    monkeypatch.setenv("SCRATCH", str(scratch))
    yield home, scratch / "data"


def _write_cache(directory: pathlib.Path, name: str) -> pathlib.Path:
    """Make ``directory`` look like an already-downloaded dataset cache."""
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{name}-train.npy", np.zeros(1))
    return directory


class TestDatasetDirResolution:
    """Where a dataset's .npy cache lives, on a cluster with $SCRATCH and without."""

    def test_without_scratch_everything_lives_under_the_repo(self, monkeypatch):
        monkeypatch.delenv("SCRATCH", raising=False)
        repo_data = pathlib.Path(dataloaders.__file__).resolve().parent.parent / "data"
        assert dataset_dir("eyepacs") == str(repo_data / "eyepacs")

    def test_with_scratch_an_uncached_large_dataset_lives_under_scratch(self, roots):
        _, scratch = roots
        assert dataset_dir("eyepacs") == str(scratch / "eyepacs")

    def test_with_scratch_a_small_dataset_still_lives_under_the_repo(self, roots):
        home, _ = roots
        assert dataset_dir("mnist") == str(home / "mnist")

    def test_a_small_dataset_already_cached_on_scratch_is_read_from_there(self, roots):
        _, scratch = roots
        cached = _write_cache(scratch / "mnist", "mnist")
        assert dataset_dir("mnist") == str(cached)

    def test_a_large_dataset_already_cached_in_the_repo_is_not_moved_to_scratch(self, roots):
        home, _ = roots
        cached = _write_cache(home / "eyepacs", "eyepacs")
        assert dataset_dir("eyepacs") == str(cached)

    def test_an_empty_directory_does_not_count_as_a_cache(self, roots):
        home, scratch = roots
        (home / "eyepacs").mkdir(parents=True)
        assert dataset_dir("eyepacs") == str(scratch / "eyepacs")


class TestGetDatasetLoaderUsesResolvedDir:
    """get_dataset_loader must read the cache from wherever dataset_dir puts it."""

    @staticmethod
    def _write_mnist(directory: pathlib.Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "mnist-train.npy", np.zeros((50, 28, 28), dtype=np.uint8))
        np.save(directory / "mnist-labels-train.npy", np.zeros((50, 10), dtype=np.float32))
        np.save(directory / "mnist-test.npy", np.zeros((40, 28, 28), dtype=np.uint8))
        np.save(directory / "mnist-labels-test.npy", np.zeros((40, 10), dtype=np.float32))

    @contextlib.contextmanager
    def _mnist_run(self, monkeypatch):
        """Scope a mnist RunContext, and make any download attempt a hard failure."""

        def _no_download(*args, **kwargs):
            raise AssertionError("get_dataset_loader tried to download an already-cached dataset")

        monkeypatch.setattr(dataloaders, "load_dataset", _no_download)
        config = Config(
            wandb_conf=WandbConfig(),
            sweep=SweepConfig(
                dataset="mnist",
                env=EnvConfig(batch_size=8),
                schedule_optimizer=ScheduleOptimizerConfig(max_sigma=10.0),
            ),
        )
        with SingletonConfig.override(config), scope.using(scope.RunContext(config)):
            yield

    def test_reads_the_cache_from_the_repo_root(self, roots, monkeypatch):
        home, _ = roots
        self._write_mnist(home / "mnist")
        with self._mnist_run(monkeypatch):
            loader = get_dataset_loader()
        assert loader.x_path == str(home / "mnist" / "mnist-train.npy")

    def test_reads_the_cache_from_scratch_when_that_is_where_it_lives(self, roots, monkeypatch):
        _, scratch = roots
        self._write_mnist(scratch / "mnist")
        with self._mnist_run(monkeypatch):
            loader = get_dataset_loader()
        assert loader.x_path == str(scratch / "mnist" / "mnist-train.npy")

    def test_the_legacy_full_load_helper_also_follows_scratch(self, roots, monkeypatch):
        _, scratch = roots

        def _no_download(*args, **kwargs):
            raise AssertionError("legacy loader tried to download an already-cached dataset")

        monkeypatch.setattr(dataloaders, "_eyepacs_download_and_cache", _no_download)
        cache = scratch / "eyepacs"
        cache.mkdir(parents=True)
        images = np.full((2, 3, 4, 4), 255, dtype=np.uint8)
        np.save(cache / "eyepacs-train.npy", images)
        np.save(cache / "eyepacs-labels-train.npy", np.eye(2, 5, dtype=np.float32))
        np.save(cache / "eyepacs-val.npy", images)
        np.save(cache / "eyepacs-labels-val.npy", np.eye(2, 5, dtype=np.float32))

        loaded_images, _ = dataloaders._dataloader_eyepacs()
        assert np.allclose(loaded_images, 1.0)
