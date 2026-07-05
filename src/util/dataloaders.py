import dataclasses
import os

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures

from conf.scope import current

__DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# ---------------------------------------------------------------------------
# Module-level memmap cache: avoids re-opening the file header on every callback
# call.  Keyed by absolute path; values are read-only np.memmap objects.
# ---------------------------------------------------------------------------
_MMAP_CACHE: dict[str, np.ndarray] = {}


def _get_mmap(path: str) -> np.ndarray:
    if path not in _MMAP_CACHE:
        _MMAP_CACHE[path] = np.load(path, mmap_mode="r")
    return _MMAP_CACHE[path]


# ---------------------------------------------------------------------------
# Per-dataset preprocessing applied inside load callbacks
# ---------------------------------------------------------------------------


def _preprocess(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply per-dataset preprocessing to raw (uint8) arrays loaded from .npy files.

    Returns float32 arrays with channels-first image layout where applicable.
    """
    if dataset_name in ("mnist", "fashion-mnist"):
        # Stored as (N, H, W) uint8 → (N, 1, H, W) float32
        x = x_raw.astype(np.float32)[:, np.newaxis] / 255.0
    elif dataset_name == "cifar-10":
        # Stored as (N, H, W, C) uint8 → (N, C, H, W) float32
        x = x_raw.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    elif dataset_name in ("eyepacs", "chexpert"):
        # Stored as (N, C, H, W) uint8 → (N, C, H, W) float32 (already channels-first)
        x = x_raw.astype(np.float32) / 255.0
    elif dataset_name == "imagenet":
        # Stored as (N, H, W, C) uint8 → (N, C, H, W) float32
        x = x_raw.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    elif dataset_name == "california":
        # Already stored as float32 with full preprocessing applied at cache time
        x = x_raw.astype(np.float32)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")
    y = y_raw.astype(np.float32)
    return x, y


def _get_sample_shape(raw_shape: tuple[int, ...], dataset_name: str) -> tuple[int, ...]:
    """Return the per-sample shape *after* preprocessing."""
    if dataset_name in ("mnist", "fashion-mnist"):
        return (1, *raw_shape)  # insert channel dim
    if dataset_name in ("cifar-10", "imagenet"):
        H, W, C = raw_shape
        return (C, H, W)  # HWC → CHW
    # eyepacs (already CHW), california (1-D features)
    return raw_shape


# ---------------------------------------------------------------------------
# DatasetLoader
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DatasetLoader:
    """Static (non-JAX-array) container for dataset file paths and metadata.

    Stored as a static field in DPTrainingParams (equinox.Module); never traced
    by JAX.  Provides load methods used inside jax.pure_callback.
    """

    x_path: str
    y_path: str
    val_x_path: str
    val_y_path: str
    n_train: int
    n_val: int
    n_test: int
    """Number of validation samples after trimming to a multiple of val_chunk_size."""
    sample_shape: tuple[int, ...]
    """Per-sample shape *after* preprocessing, e.g. (1, 28, 28) for MNIST."""
    label_shape: tuple[int, ...]
    """Per-label shape, e.g. (10,) for MNIST one-hot."""
    dataset_name: str
    val_chunk_size: int
    """Must divide n_val; determines the static batch shape inside the val scan."""
    split_seed: int = 0
    """Seed for the deterministic permutation used to split the val file into
    val/test.  The raw val file is often ordered (e.g. a difficulty gradient),
    so a contiguous slice would give a biased test set; permuting first makes
    val and test exchangeable draws from the same distribution."""

    def _val_test_perm(self) -> np.ndarray:
        """Deterministic permutation of the val/test pool rows ([0, n_val+n_test)).

        Recomputed on demand (cheap relative to the memmap reads) so the loader
        stays a hashable, array-free static field.  Indices into the val file are
        obtained by mapping a contiguous chunk index through this permutation.
        """
        pool = self.n_val + self.n_test
        return np.random.default_rng(self.split_seed).permutation(pool)

    def __post_init__(self) -> None:
        assert self.n_val % self.val_chunk_size == 0, (
            f"n_val ({self.n_val}) must be divisible by val_chunk_size "
            f"({self.val_chunk_size}).  Adjust val_chunk_size or trim n_val "
            f"before constructing DatasetLoader."
        )

        x_val_rows = _get_mmap(self.val_x_path).shape[0]
        if self.n_val + self.n_test > x_val_rows:
            raise ValueError(
                f"n_val ({self.n_val}) + n_test ({self.n_test}) must be <= rows(x_val) ({x_val_rows})"
            )

    def load_train_batch(
        self,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fetch and preprocess a training mini-batch by integer index array.

        Called from inside jax.pure_callback; must return plain numpy arrays.
        """
        x_mmap = _get_mmap(self.x_path)
        y_mmap = _get_mmap(self.y_path)
        return _preprocess(x_mmap[indices].copy(), y_mmap[indices].copy(), self.dataset_name)

    def _load_chunk(
        self,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_mmap = _get_mmap(self.val_x_path)
        y_mmap = _get_mmap(self.val_y_path)
        return _preprocess(x_mmap[indices].copy(), y_mmap[indices].copy(), self.dataset_name)

    def load_val_chunk(
        self,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fetch and preprocess a validation chunk by integer index array.

        The contiguous `indices` (in [0, n_val)) are mapped through the val/test
        permutation so val draws are spread across the whole file.
        """
        return self._load_chunk(self._val_test_perm()[indices])

    def load_test_chunk(
        self,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fetch and preprocess a test chunk by integer index array.

        The contiguous `indices` (in [0, n_test)) are offset past the val region
        and mapped through the same permutation, so val and test are disjoint
        i.i.d. draws from the shuffled pool.
        """
        return self._load_chunk(self._val_test_perm()[indices + self.n_val])


# ---------------------------------------------------------------------------
# File-creation helpers (unchanged logic; now also used by get_dataset_loader)
# ---------------------------------------------------------------------------


def image_ds_saver(ds, x_file, y_file):
    """Save a HuggingFace image dataset split to .npy files.

    Converts images to JAX arrays and one-hot encodes labels before saving.

    Args:
        ds: HuggingFace dataset split (train or test).
        x_file: Path to save the images array.
        y_file: Path to save the one-hot labels array.
    """
    import jax.numpy as jnp

    # https://huggingface.co/docs/datasets/en/use_with_jax
    dds = ds.with_format("jax")
    image_key = "image" if "image" in dds.features else "img"
    images = dds[image_key]
    labels = pd.Series(dds["label"]).astype(jnp.int32)
    labels = jnp.asarray(pd.get_dummies(labels).values).astype(jnp.float32)

    jnp.save(x_file, images)  # type: ignore
    jnp.save(y_file, labels)  # type: ignore


def _chexpert_binary_onehot(column: np.ndarray) -> np.ndarray:
    """Map a raw CheXpert finding column to binary one-hot under the U-Zeros convention.

    Under U-Zeros only an explicit positive (``1.0``) is treated as positive;
    negatives (``0.0``), uncertains (``-1.0``) and blanks (``NaN``) all map to
    the negative class.  Returns an ``(N, 2)`` float32 one-hot with column 1 the
    positive class.
    """
    col = np.asarray(column, dtype=np.float32)
    positive = col == 1.0
    onehot = np.zeros((col.shape[0], 2), dtype=np.float32)
    onehot[positive, 1] = 1.0
    onehot[~positive, 0] = 1.0
    return onehot


def _imagenet100_select(
    sample_ids: np.ndarray,
    subset_ids: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the rows belonging to ``subset_ids`` and remap their class labels.

    ``sample_ids`` is the per-sample class identifier from the 1000-class source
    (e.g. wnid). ``subset_ids`` is the ordered list of the published ImageNet-100
    identifiers. Returns ``(mask, labels)`` where ``mask`` marks the kept rows and
    ``labels`` are the remapped ``[0, len(subset_ids))`` class indices for the kept
    rows, in original order (position within ``subset_ids``).
    """
    remap = {wnid: i for i, wnid in enumerate(subset_ids)}
    sample_ids = np.asarray(sample_ids)
    mask = np.isin(sample_ids, subset_ids)
    labels = np.array([remap[s] for s in sample_ids[mask]], dtype=np.int64)
    return mask, labels


def _chexpert_frontal_mask(frontal_lateral: np.ndarray) -> np.ndarray:
    """Boolean mask selecting frontal-view rows from the ``Frontal/Lateral`` column."""
    return np.asarray(frontal_lateral) == "Frontal"


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


def _ensure_mnist_cached(datadir: str, variant: str = "mnist") -> tuple[str, str, str, str]:
    """Ensure MNIST or Fashion-MNIST .npy files exist; return their paths."""
    os.makedirs(datadir, exist_ok=True)
    if variant == "mnist":
        hf_train_name = "ylecun/mnist"
        prefix = "mnist"
    else:
        hf_train_name = "zalando-datasets/fashion_mnist"
        prefix = "fashion-mnist"

    x_train = os.path.join(datadir, f"{prefix}-train.npy")
    y_train = os.path.join(datadir, f"{prefix}-labels-train.npy")
    x_test = os.path.join(datadir, f"{prefix}-test.npy")
    y_test = os.path.join(datadir, f"{prefix}-labels-test.npy")

    if not os.path.exists(x_train) or not os.path.exists(y_train):
        print(f"Downloading {variant} dataset...")
        ds = load_dataset(hf_train_name, split="train")
        image_ds_saver(ds, x_train, y_train)
    if not os.path.exists(x_test) or not os.path.exists(y_test):
        print(f"Downloading {variant} test dataset...")
        ds = load_dataset(hf_train_name, split="test")
        image_ds_saver(ds, x_test, y_test)

    return x_train, y_train, x_test, y_test


def _ensure_cifar10_cached(datadir: str) -> tuple[str, str, str, str]:
    """Ensure CIFAR-10 .npy files exist; return their paths."""
    os.makedirs(datadir, exist_ok=True)
    x_train = os.path.join(datadir, "cifar-10-train.npy")
    y_train = os.path.join(datadir, "cifar-10-labels-train.npy")
    x_test = os.path.join(datadir, "cifar-10-test.npy")
    y_test = os.path.join(datadir, "cifar-10-labels-test.npy")

    if not os.path.exists(x_train) or not os.path.exists(y_train):
        print("Downloading CIFAR-10 dataset...")
        ds = load_dataset("uoft-cs/cifar10", split="train")
        image_ds_saver(ds, x_train, y_train)
    if not os.path.exists(x_test) or not os.path.exists(y_test):
        print("Downloading CIFAR-10 test dataset...")
        ds = load_dataset("uoft-cs/cifar10", split="test")
        image_ds_saver(ds, x_test, y_test)

    return x_train, y_train, x_test, y_test


def _ensure_eyepacs_cached(datadir: str) -> tuple[str, str, str, str]:
    """Ensure EyePACS .npy files exist; return their paths."""
    os.makedirs(datadir, exist_ok=True)
    x_train = os.path.join(datadir, "eyepacs-train.npy")
    y_train = os.path.join(datadir, "eyepacs-labels-train.npy")
    x_val = os.path.join(datadir, "eyepacs-val.npy")
    y_val = os.path.join(datadir, "eyepacs-labels-val.npy")

    if not all(os.path.exists(p) for p in [x_train, y_train, x_val, y_val]):
        _eyepacs_download_and_cache(datadir)

    return x_train, y_train, x_val, y_val


_CHEXPERT_IMG_SIZE = 64
_CHEXPERT_FINDING = "Pleural Effusion"

# Published ImageNet-100 wnid subset (Tian et al. 2019, CMC — imagenet100.txt).
# Fixed and citable; the surrogate ImageNet target is exactly these 100 classes.
_IMAGENET100_WNIDS: list[str] = [
    "n02869837",
    "n01749939",
    "n02488291",
    "n02107142",
    "n13037406",
    "n02091831",
    "n04517823",
    "n04589890",
    "n03062245",
    "n01773797",
    "n01735189",
    "n07831146",
    "n07753275",
    "n03085013",
    "n04485082",
    "n02105505",
    "n01983481",
    "n02788148",
    "n03530642",
    "n04435653",
    "n02086910",
    "n02859443",
    "n13040303",
    "n03594734",
    "n02085620",
    "n02099849",
    "n01558993",
    "n04493381",
    "n02109047",
    "n04111531",
    "n02877765",
    "n04429376",
    "n02009229",
    "n01978455",
    "n02106550",
    "n01820546",
    "n01692333",
    "n07714571",
    "n02974003",
    "n02114855",
    "n03785016",
    "n03764736",
    "n03775546",
    "n02087046",
    "n07836838",
    "n04099969",
    "n04592741",
    "n03891251",
    "n02701002",
    "n03379051",
    "n02259212",
    "n07715103",
    "n03947888",
    "n04026417",
    "n02326432",
    "n03637318",
    "n01980166",
    "n02113799",
    "n02086240",
    "n03903868",
    "n02483362",
    "n04127249",
    "n02089973",
    "n03017168",
    "n02093428",
    "n02804414",
    "n02396427",
    "n04418357",
    "n02172182",
    "n01729322",
    "n02113978",
    "n03787032",
    "n02089867",
    "n02119022",
    "n03777754",
    "n04238763",
    "n02231487",
    "n03032252",
    "n02138441",
    "n02104029",
    "n03837869",
    "n03494278",
    "n04136333",
    "n03794056",
    "n03492542",
    "n02018207",
    "n04067472",
    "n03930630",
    "n03584829",
    "n02123045",
    "n04229816",
    "n02100583",
    "n03642806",
    "n04336792",
    "n03259280",
    "n02116738",
    "n02108089",
    "n03424325",
    "n01855672",
    "n02090622",
]

# Chrabaszcz downsampled-ImageNet 32x32 mirror on the Hugging Face hub. The label
# feature's `names` must be wnids (n########); we map them onto the ImageNet-100
# subset above. Change this constant if a different mirror is used.
_IMAGENET32_HF_REPO = "Chrabaszcz/imagenet32"


def _imagenet32_download_and_cache(datadir: str) -> None:
    """Download the HF ImageNet-32 mirror, keep the 100-wnid subset, cache as .npy.

    Mirrors ``image_ds_saver`` but filters to ``_IMAGENET100_WNIDS`` and remaps the
    labels to ``[0, 100)`` via ``_imagenet100_select``. Images are cached HWC uint8
    (like cifar-10) so ``_preprocess('imagenet', ...)`` transposes to CHW.

    Produces:
        imagenet-train.npy         (N_train, 32, 32, 3) uint8
        imagenet-labels-train.npy  (N_train, 100) float32  one-hot
        imagenet-val.npy           (N_val,   32, 32, 3) uint8
        imagenet-labels-val.npy    (N_val,   100) float32
    """
    for split_name, hf_split in (("train", "train"), ("val", "validation")):
        ds = load_dataset(_IMAGENET32_HF_REPO, split=hf_split)
        wnid_names = ds.features["label"].names
        sample_wnids = np.array([wnid_names[label] for label in ds["label"]])

        mask, remapped = _imagenet100_select(sample_wnids, _IMAGENET100_WNIDS)

        image_key = "image" if "image" in ds.features else "img"
        images = np.stack(
            [np.asarray(img.convert("RGB"), dtype=np.uint8) for img in ds[image_key]]
        )[mask]
        labels = np.eye(len(_IMAGENET100_WNIDS), dtype=np.float32)[remapped]

        np.save(os.path.join(datadir, f"imagenet-{split_name}.npy"), images)
        np.save(os.path.join(datadir, f"imagenet-labels-{split_name}.npy"), labels)


def _ensure_imagenet32_cached(datadir: str) -> tuple[str, str, str, str]:
    """Ensure ImageNet-32 (100-class subset) .npy files exist; return their paths."""
    os.makedirs(datadir, exist_ok=True)
    x_train = os.path.join(datadir, "imagenet-train.npy")
    y_train = os.path.join(datadir, "imagenet-labels-train.npy")
    x_val = os.path.join(datadir, "imagenet-val.npy")
    y_val = os.path.join(datadir, "imagenet-labels-val.npy")

    if not all(os.path.exists(p) for p in [x_train, y_train, x_val, y_val]):
        _imagenet32_download_and_cache(datadir)

    return x_train, y_train, x_val, y_val


def _chexpert_download_and_cache(datadir: str) -> None:
    """Download Kaggle CheXpert-v1.0-small and cache a binary Pleural-Effusion probe.

    Mirrors ``_eyepacs_download_and_cache``. The surrogate regime (ADR 0007) is:
    single finding ``Pleural Effusion`` under the **U-Zeros** convention, **frontal
    view only**, **grayscale (1 channel)**, resized to 64x64. The tiny official
    ``valid.csv`` is unused; a val/test pool is carved out of ``train.csv`` and
    permuted by ``split_seed`` downstream (like california), so ``δ ≤ 1/N`` uses the
    reduced ``N_train``.

    Produces:
        chexpert-train.npy         (N_train, 1, 64, 64) uint8
        chexpert-labels-train.npy  (N_train, 2) float32  one-hot
        chexpert-val.npy           (N_val,   1, 64, 64) uint8
        chexpert-labels-val.npy    (N_val,   2) float32
    """
    import kaggle  # noqa: F401 — validates token exists before we start
    from PIL import Image

    # --- Download + unzip via Kaggle API ---
    csv_path = os.path.join(datadir, "train.csv")
    if not os.path.exists(csv_path):
        print("Downloading CheXpert-v1.0-small from Kaggle (this may take a while)...")
        ret = os.system(f"kaggle datasets download -d ashery/chexpert -p {datadir} --unzip")
        if ret != 0:
            raise Exception("Download failed")

    df = pd.read_csv(csv_path)

    # Frontal-only, then split into train / (val+test pool) mirroring eyepacs.
    df = df[_chexpert_frontal_mask(df["Frontal/Lateral"].to_numpy())].reset_index(drop=True)

    rng = np.random.default_rng(seed=0)
    order = rng.permutation(len(df))
    n_train = int(len(df) * _EYEPACS_TRAIN_SPLIT)
    splits = {"train": order[:n_train], "val": order[n_train:]}

    size = _CHEXPERT_IMG_SIZE
    for split_name, idx in splits.items():
        rows = df.iloc[idx].reset_index(drop=True)
        n = len(rows)
        img_file = os.path.join(datadir, f"chexpert-{split_name}.npy")
        lbl_file = os.path.join(datadir, f"chexpert-labels-{split_name}.npy")

        print(f"Processing {split_name} split ({n} images)...")
        mmap = np.memmap(img_file + ".tmp", dtype=np.uint8, mode="w+", shape=(n, 1, size, size))
        for i, rel_path in enumerate(rows["Path"]):
            img_path = os.path.join(datadir, rel_path)
            with Image.open(img_path) as img:
                img_resized = img.convert("L").resize((size, size), Image.BILINEAR)
                mmap[i, 0] = np.asarray(img_resized, dtype=np.uint8)
            if i % 5000 == 0:
                print(f"  {i}/{n}")
                mmap.flush()
        mmap.flush()
        del mmap
        os.replace(img_file + ".tmp", img_file)

        labels = _chexpert_binary_onehot(rows[_CHEXPERT_FINDING].to_numpy())
        np.save(lbl_file, labels)
        print(f"  Saved {split_name} split.")


def _ensure_chexpert_cached(datadir: str) -> tuple[str, str, str, str]:
    """Ensure CheXpert .npy files exist; return their paths."""
    os.makedirs(datadir, exist_ok=True)
    x_train = os.path.join(datadir, "chexpert-train.npy")
    y_train = os.path.join(datadir, "chexpert-labels-train.npy")
    x_val = os.path.join(datadir, "chexpert-val.npy")
    y_val = os.path.join(datadir, "chexpert-labels-val.npy")

    if not all(os.path.exists(p) for p in [x_train, y_train, x_val, y_val]):
        _chexpert_download_and_cache(datadir)

    return x_train, y_train, x_val, y_val


def _ensure_california_cached(datadir: str, poly_d: int | None) -> tuple[str, str, str, str]:
    """Ensure California Housing .npy files exist with an 80/20 train/val split."""
    os.makedirs(datadir, exist_ok=True)
    degree = poly_d if poly_d is not None else 1
    prefix = f"california-d{degree}"
    x_train = os.path.join(datadir, f"{prefix}-train.npy")
    y_train = os.path.join(datadir, f"{prefix}-labels-train.npy")
    x_val = os.path.join(datadir, f"{prefix}-val.npy")
    y_val = os.path.join(datadir, f"{prefix}-labels-val.npy")

    if all(os.path.exists(p) for p in [x_train, y_train, x_val, y_val]):
        return x_train, y_train, x_val, y_val

    X_raw, y_raw = fetch_california_housing(return_X_y=True)
    X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
    X = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(X)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y_classes = y_raw < np.median(y_raw)
    y = pd.get_dummies(y_classes).values.astype(np.float32)
    X = X.astype(np.float32)

    rng = np.random.default_rng(seed=42)
    indices = rng.permutation(len(X))
    n_train = int(len(X) * 0.8)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    np.save(x_train, X[train_idx])
    np.save(y_train, y[train_idx])
    np.save(x_val, X[val_idx])
    np.save(y_val, y[val_idx])

    return x_train, y_train, x_val, y_val


# ---------------------------------------------------------------------------
# Primary entry point: get_dataset_loader
# ---------------------------------------------------------------------------


def get_dataset_loader() -> DatasetLoader:
    """Ensure dataset files exist and return a DatasetLoader with file paths + metadata.

    Does NOT load any data into memory.  The DatasetLoader's load_* methods
    use numpy memmaps to fetch only the requested samples on demand.
    """
    sweep_config = current().config.sweep
    env_config = current().config.sweep.env
    dataset_name = sweep_config.dataset
    poly_d = sweep_config.dataset_poly_d
    batch_size = env_config.batch_size

    if dataset_name == "mnist":
        datadir = os.path.join(__DATA_DIR, "mnist")
        x_train, y_train, x_val, y_val = _ensure_mnist_cached(datadir, variant="mnist")
    elif dataset_name == "fashion-mnist":
        datadir = os.path.join(__DATA_DIR, "fashion-mnist")
        x_train, y_train, x_val, y_val = _ensure_mnist_cached(datadir, variant="fashion-mnist")
    elif dataset_name == "cifar-10":
        datadir = os.path.join(__DATA_DIR, "cifar-10")
        x_train, y_train, x_val, y_val = _ensure_cifar10_cached(datadir)
    elif dataset_name == "eyepacs":
        datadir = os.path.join(__DATA_DIR, "eyepacs")
        x_train, y_train, x_val, y_val = _ensure_eyepacs_cached(datadir)
    elif dataset_name == "chexpert":
        datadir = os.path.join(__DATA_DIR, "chexpert")
        x_train, y_train, x_val, y_val = _ensure_chexpert_cached(datadir)
    elif dataset_name == "imagenet":
        datadir = os.path.join(__DATA_DIR, "imagenet")
        x_train, y_train, x_val, y_val = _ensure_imagenet32_cached(datadir)
    elif dataset_name == "california":
        datadir = os.path.join(__DATA_DIR, "california")
        x_train, y_train, x_val, y_val = _ensure_california_cached(datadir, poly_d)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")

    # Read shapes from file headers only (no data loaded into RAM)
    x_mmap = np.load(x_train, mmap_mode="r")
    y_mmap = np.load(y_train, mmap_mode="r")
    xv_mmap = np.load(x_val, mmap_mode="r")

    n_train = x_mmap.shape[0]
    n_val_raw = xv_mmap.shape[0]
    raw_sample_shape = x_mmap.shape[1:]
    label_shape = y_mmap.shape[1:]

    sample_shape = _get_sample_shape(raw_sample_shape, dataset_name)

    # Trim val set to a multiple of batch_size so val_chunk_size divides (n_val + n_test) exactly,
    # and n_test / (n_val + n_test) \approx test_percentace
    val_test_split = sweep_config.val_test_split
    val_chunk_size = batch_size
    n_test_raw = round(n_val_raw * (1 - val_test_split))
    n_true_val_raw = round(n_val_raw * val_test_split)
    n_val = (n_true_val_raw // val_chunk_size) * val_chunk_size
    n_test = (n_test_raw // val_chunk_size) * val_chunk_size

    return DatasetLoader(
        x_path=x_train,
        y_path=y_train,
        val_x_path=x_val,
        val_y_path=y_val,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        sample_shape=sample_shape,
        label_shape=label_shape,
        dataset_name=dataset_name,
        val_chunk_size=val_chunk_size,
    )


# ---------------------------------------------------------------------------
# Backward-compatible helpers
# ---------------------------------------------------------------------------


def get_dataset_shapes() -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Return the shapes of the train and validation splits from the configured dataset.

    Returns:
        Tuple of (X_shape, y_shape, valX_shape, valy_shape).
    """
    loader = get_dataset_loader()
    X_shape = (loader.n_train, *loader.sample_shape)
    y_shape = (loader.n_train, *loader.label_shape)
    valX_shape = (loader.n_val, *loader.sample_shape)
    valy_shape = (loader.n_val, *loader.label_shape)
    return X_shape, y_shape, valX_shape, valy_shape


# ---------------------------------------------------------------------------
# Legacy full-load helpers (kept for external / one-off use; not used in the
# main training pipeline after the DatasetLoader refactor)
# ---------------------------------------------------------------------------


def _dataloader_eyepacs(_=None, test=False) -> tuple[np.ndarray, np.ndarray]:
    """Load EyePACS retinal images as float32 arrays (legacy full-load helper)."""
    eyepacs_datadir = os.path.join(__DATA_DIR, "eyepacs")
    x_train, y_train, x_val, y_val = _ensure_eyepacs_cached(eyepacs_datadir)
    img_file = x_val if test else x_train
    lbl_file = y_val if test else y_train
    images = np.load(img_file).astype(np.float32) / 255.0
    labels = np.load(lbl_file).astype(np.float32)
    return images, labels


def _dataloader_california(degree=1):
    """Load the California Housing dataset as a binary classification problem (legacy)."""
    california_datadir = os.path.join(__DATA_DIR, "california")
    poly_d = degree if degree != 1 else None
    x_train, y_train, _, _ = _ensure_california_cached(california_datadir, poly_d)
    return np.load(x_train).astype(np.float32), np.load(y_train).astype(np.float32)


def _dataloader_mnist(_=None, test=False) -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST as float32 arrays (legacy full-load helper)."""
    mnist_datadir = os.path.join(__DATA_DIR, "mnist")
    x_train, y_train, x_test, y_test = _ensure_mnist_cached(mnist_datadir, variant="mnist")
    x_file = x_test if test else x_train
    y_file = y_test if test else y_train
    images = np.load(x_file).astype(np.float32)[:, np.newaxis] / 255.0
    labels = np.load(y_file).astype(np.float32)
    return images, labels


def _dataloader_cifar_10(_=None, test=False) -> tuple[np.ndarray, np.ndarray]:
    """Load CIFAR-10 as float32 arrays (legacy full-load helper)."""
    cifar_datadir = os.path.join(__DATA_DIR, "cifar-10")
    x_train, y_train, x_test, y_test = _ensure_cifar10_cached(cifar_datadir)
    x_file = x_test if test else x_train
    y_file = y_test if test else y_train
    images = np.load(x_file).astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    labels = np.load(y_file).astype(np.float32)
    return images, labels


def _dataloader_fashion_mnist(_=None, test=False) -> tuple[np.ndarray, np.ndarray]:
    """Load Fashion-MNIST as float32 arrays (legacy full-load helper)."""
    fmnist_datadir = os.path.join(__DATA_DIR, "fashion-mnist")
    x_train, y_train, x_test, y_test = _ensure_mnist_cached(fmnist_datadir, variant="fashion-mnist")
    x_file = x_test if test else x_train
    y_file = y_test if test else y_train
    images = np.load(x_file).astype(np.float32)[:, np.newaxis] / 255.0
    labels = np.load(y_file).astype(np.float32)
    return images, labels


DATALOADERS = {
    "california": _dataloader_california,
    "mnist": _dataloader_mnist,
    "cifar-10": _dataloader_cifar_10,
    "fashion-mnist": _dataloader_fashion_mnist,
    "eyepacs": _dataloader_eyepacs,
}


if __name__ == "__main__":
    _dataloader_mnist()
