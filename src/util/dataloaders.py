import os

import chex
import jax.numpy as jnp
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures

from conf.singleton_conf import SingletonConfig

__DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def image_ds_saver(ds, x_file, y_file):
    """Save a HuggingFace image dataset split to .npy files.

    Converts images to JAX arrays and one-hot encodes labels before saving.

    Args:
        ds: HuggingFace dataset split (train or test).
        x_file: Path to save the images array.
        y_file: Path to save the one-hot labels array.
    """
    # https://huggingface.co/docs/datasets/en/use_with_jax
    dds = ds.with_format("jax")
    image_key = "image" if "image" in dds.features else "img"
    images = dds[image_key]
    labels = pd.Series(dds["label"]).astype(jnp.int32)
    labels = jnp.asarray(pd.get_dummies(labels).values).astype(jnp.float32)

    jnp.save(x_file, images)  # type: ignore
    jnp.save(y_file, labels)  # type: ignore


_EYEPACS_IMG_SIZE = 256
_EYEPACS_TRAIN_SPLIT = 0.8


def _eyepacs_download_and_cache(datadir: str) -> None:
    """Download EyePACS from Kaggle, resize images to 256x256 RGB, and save as .npy.

    Requires the Kaggle API token at ~/.kaggle/kaggle.json and the kaggle
    package to be installed. Images are written one-by-one via np.memmap so
    the full dataset is never held in RAM simultaneously.

    Produces four files:
        eyepacs-train.npy         (N_train, 3, 256, 256) uint8
        eyepacs-labels-train.npy  (N_train, 5) float32  one-hot over grades 0-4
        eyepacs-val.npy           (N_val,   3, 256, 256) uint8
        eyepacs-labels-val.npy    (N_val,   5) float32
    """
    import csv
    import zipfile

    from PIL import Image

    zip_path = os.path.join(datadir, "diabetic-retinopathy-detection.zip")
    train_img_dir = os.path.join(datadir, "train")

    # --- Download via Kaggle API ---
    if not os.path.exists(zip_path) and not os.path.exists(train_img_dir):
        print("Downloading EyePACS from Kaggle (this may take a while)...")
        import kaggle  # noqa: F401 — validates token exists before we start

        ret = os.system(
            f"kaggle competitions download -c diabetic-retinopathy-detection -p {datadir}"
        )
        print(datadir)
        if ret != 0:
            raise Exception("Download failed")

    # --- Extract zip ---
    if not os.path.exists(train_img_dir):
        print("Extracting EyePACS zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            # The competition zip contains nested zips; extract the train one
            train_zip_name = next(n for n in zf.namelist() if "train" in n and n.endswith(".zip"))
            zf.extract(train_zip_name, datadir)
        train_zip_path = os.path.join(datadir, train_zip_name)
        with zipfile.ZipFile(train_zip_path, "r") as zf:
            zf.extractall(datadir)

    # --- Read labels CSV (may also be nested in zip) ---
    labels_csv = os.path.join(datadir, "trainLabels.csv")
    if not os.path.exists(labels_csv):
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_name = next(n for n in zf.namelist() if "trainLabels" in n)
            zf.extract(csv_name, datadir)

    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Shuffle deterministically so the split is reproducible
    rng = np.random.default_rng(seed=0)
    rng.shuffle(rows)

    n_total = len(rows)
    n_train = int(n_total * _EYEPACS_TRAIN_SPLIT)
    splits = {"train": rows[:n_train], "val": rows[n_train:]}

    size = _EYEPACS_IMG_SIZE
    n_classes = 5

    for split_name, split_rows in splits.items():
        n = len(split_rows)
        img_file = os.path.join(datadir, f"eyepacs-{split_name}.npy")
        lbl_file = os.path.join(datadir, f"eyepacs-labels-{split_name}.npy")

        print(f"Processing {split_name} split ({n} images)...")
        mmap = np.memmap(img_file + ".tmp", dtype=np.uint8, mode="w+", shape=(n, 3, size, size))
        labels_arr = np.zeros((n, n_classes), dtype=np.float32)

        for i, row in enumerate(split_rows):
            img_path = os.path.join(train_img_dir, row["image"] + ".jpeg")
            with Image.open(img_path) as img:
                img_resized = img.convert("RGB").resize((size, size), Image.BILINEAR)
                mmap[i] = np.asarray(img_resized, dtype=np.uint8).transpose(2, 0, 1)
            labels_arr[i, int(row["level"])] = 1.0
            if i % 5000 == 0:
                print(f"  {i}/{n}")
                mmap.flush()

        mmap.flush()
        del mmap
        os.replace(img_file + ".tmp", img_file)
        np.save(lbl_file, labels_arr)
        print(f"  Saved {split_name} split.")


def _dataloader_eyepacs(_=None, test=False) -> tuple[chex.Array, chex.Array]:
    """Load EyePACS retinal images, downloading and caching as .npy files if necessary.

    Downloads from Kaggle on first use (requires ~/.kaggle/kaggle.json and the
    kaggle package). Images are resized to (3, 256, 256) RGB and normalised to [0, 1].
    The Kaggle train set is split 80/20 into train and val splits.

    Args:
        _: Unused (polynomial degree placeholder for API consistency).
        test: If True, return the validation split instead of the training split.

    Returns:
        Tuple of (images, labels) with shapes (N, 3, 256, 256) and (N, 5).
    """
    eyepacs_datadir = os.path.join(__DATA_DIR, "eyepacs")
    os.makedirs(eyepacs_datadir, exist_ok=True)
    image_train_file = os.path.join(eyepacs_datadir, "eyepacs-train.npy")
    label_train_file = os.path.join(eyepacs_datadir, "eyepacs-labels-train.npy")
    image_val_file = os.path.join(eyepacs_datadir, "eyepacs-val.npy")
    label_val_file = os.path.join(eyepacs_datadir, "eyepacs-labels-val.npy")

    cached = all(
        os.path.exists(p)
        for p in [image_train_file, label_train_file, image_val_file, label_val_file]
    )
    if not cached:
        _eyepacs_download_and_cache(eyepacs_datadir)

    img_file = image_val_file if test else image_train_file
    lbl_file = label_val_file if test else label_train_file

    images = jnp.asarray(np.load(img_file)) / 255.0  # (N, 3, 256, 256) float
    labels = jnp.asarray(np.load(lbl_file))  # (N, 5)
    return images, labels


def _dataloader_california(degree=1):
    """Load the California Housing dataset as a binary classification problem.

    Applies polynomial feature expansion and z-score normalization.
    The regression target is binarised by comparing each value to the median.

    Args:
        degree: Polynomial degree for feature expansion (default 1 = no expansion).

    Returns:
        Tuple of (X, y) JAX arrays with shapes (N, features) and (N, 2).
    """
    X, y = fetch_california_housing(return_X_y=True)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(X)

    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y_classes = y < np.median(y)
    y = pd.get_dummies(y_classes).values.astype(np.int32)
    return jnp.asarray(X), jnp.asarray(y)


def _dataloader_mnist(_=None, test=False) -> tuple[chex.Array, chex.Array]:
    """Load MNIST, downloading and caching as .npy files if necessary.

    Args:
        _: Unused (polynomial degree placeholder for API consistency).
        test: If True, return the test split instead of the training split.

    Returns:
        Tuple of (images, labels) with shapes (N, 1, 28, 28) and (N, 10).
    """
    mnist_datadir = os.path.join(__DATA_DIR, "mnist")
    os.makedirs(mnist_datadir, exist_ok=True)
    image_train_file = os.path.join(mnist_datadir, "mnist-train.npy")
    label_train_file = os.path.join(mnist_datadir, "mnist-labels-train.npy")
    image_test_file = os.path.join(mnist_datadir, "mnist-test.npy")
    label_test_file = os.path.join(mnist_datadir, "mnist-labels-test.npy")
    if not os.path.exists(image_train_file) or not os.path.exists(label_train_file):
        print("Downloading MNIST dataset...")
        ds = load_dataset("ylecun/mnist", split="train")
        image_ds_saver(ds, image_train_file, label_train_file)

    if not os.path.exists(image_test_file) or not os.path.exists(label_test_file):
        print("Downloading MNIST test dataset...")
        ds = load_dataset("ylecun/mnist", split="test")
        image_ds_saver(ds, image_test_file, label_test_file)
    if test:
        images = image_test_file
        labels = label_test_file
    else:
        images = image_train_file
        labels = label_train_file

    images = jnp.load(images)
    labels = jnp.load(labels)
    # convert 'labels' to one-hot encoding

    # normalize & flatten images
    images = images / 255.0

    # Add channel dimension in second position
    # (ndatapoints, nchannels, *image_shape)
    images = jnp.expand_dims(images, 1)

    return images, labels


def _dataloader_cifar_10(_=None, test=False) -> tuple[chex.Array, chex.Array]:
    """Load CIFAR-10, downloading and caching as .npy files if necessary.

    Images are converted from (N, H, W, C) to (N, C, H, W) channel-first format
    and normalised to [0, 1].

    Args:
        _: Unused (polynomial degree placeholder for API consistency).
        test: If True, return the test split instead of the training split.

    Returns:
        Tuple of (images, labels) with shapes (N, 3, 32, 32) and (N, 10).
    """
    ds = load_dataset("uoft-cs/cifar10", split="train")
    # https://huggingface.co/docs/datasets/en/use_with_jax
    cifar_datadir = os.path.join(__DATA_DIR, "cifar-10")
    os.makedirs(cifar_datadir, exist_ok=True)
    image_train_file = os.path.join(cifar_datadir, "cifar-10-train.npy")
    label_train_file = os.path.join(cifar_datadir, "cifar-10-labels-train.npy")
    image_test_file = os.path.join(cifar_datadir, "cifar-10-test.npy")
    label_test_file = os.path.join(cifar_datadir, "cifar-10-labels-test.npy")
    if not os.path.exists(image_train_file) or not os.path.exists(label_train_file):
        print("Downloading CIFAR-10 dataset...")
        ds = load_dataset("uoft-cs/cifar10", split="train")
        image_ds_saver(ds, image_train_file, label_train_file)

    if not os.path.exists(image_test_file) or not os.path.exists(label_test_file):
        print("Downloading CIFAR-10 test dataset...")
        ds = load_dataset("uoft-cs/cifar10", split="test")
        image_ds_saver(ds, image_test_file, label_test_file)
    if test:
        image_train_file = image_test_file
        label_train_file = label_test_file

    images = jnp.load(image_train_file)
    labels = jnp.load(label_train_file)

    # normalize & flatten images
    images = images / 255.0

    # images are n x n x c, should be c x n x n
    images = images.transpose((0, 3, 1, 2))

    return images, labels


def _dataloader_fashion_mnist(_=None, test=False) -> tuple[chex.Array, chex.Array]:
    """Load Fashion-MNIST, downloading and caching as .npy files if necessary.

    Args:
        _: Unused (polynomial degree placeholder for API consistency).
        test: If True, return the test split instead of the training split.

    Returns:
        Tuple of (images, labels) with shapes (N, 1, 28, 28) and (N, 10).
    """
    mnist_datadir = os.path.join(__DATA_DIR, "fashion-mnist")
    os.makedirs(mnist_datadir, exist_ok=True)
    image_train_file = os.path.join(mnist_datadir, "fashion-mnist-train.npy")
    label_train_file = os.path.join(mnist_datadir, "fashion-mnist-labels-train.npy")
    image_test_file = os.path.join(mnist_datadir, "fashion-mnist-test.npy")
    label_test_file = os.path.join(mnist_datadir, "fashion-mnist-labels-test.npy")
    if not os.path.exists(image_train_file) or not os.path.exists(label_train_file):
        print("Downloading fashion MNIST dataset...")
        ds = load_dataset("zalando-datasets/fashion_mnist", split="train")
        image_ds_saver(ds, image_train_file, label_train_file)

    if not os.path.exists(image_test_file) or not os.path.exists(label_test_file):
        print("Downloading fashion MNIST test dataset...")
        ds = load_dataset("zalando-datasets/fashion_mnist", split="test")
        image_ds_saver(ds, image_test_file, label_test_file)
    if test:
        image_train_file = image_test_file
        label_train_file = label_test_file

    images = jnp.load(image_train_file)
    labels = jnp.load(label_train_file)
    # convert 'labels' to one-hot encoding

    # normalize & flatten images
    images = images / 255.0

    # Add channel dimension in second position
    # (ndatapoints, nchannels, *image_shape)
    images = jnp.expand_dims(images, 1)

    return images, labels


DATALOADERS = {
    "california": _dataloader_california,
    "mnist": _dataloader_mnist,
    "cifar-10": _dataloader_cifar_10,
    "fashion-mnist": _dataloader_fashion_mnist,
    "eyepacs": _dataloader_eyepacs,
}


def get_datasets():
    """Load train and test splits for the dataset specified in the singleton config.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test) JAX arrays.
    """
    sweep_config = SingletonConfig.get_sweep_config_instance()
    X, y = DATALOADERS[sweep_config.dataset](sweep_config.dataset_poly_d)
    X_test, y_test = DATALOADERS[sweep_config.dataset](
        sweep_config.dataset_poly_d,
        test=True,
    )
    return X, y, X_test, y_test


def get_dataset_shapes():
    """Return the shapes of the train and validation splits from the configured dataset.

    Returns:
        Tuple of (X_shape, y_shape, valX_shape, valy_shape).
    """
    X, y, valX, valy = get_datasets()
    return X.shape, y.shape, valX.shape, valy.shape


if __name__ == "__main__":
    _dataloader_mnist()
