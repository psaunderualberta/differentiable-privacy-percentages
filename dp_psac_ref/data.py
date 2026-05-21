"""Standalone dataset loaders: MNIST and Fashion-MNIST via torchvision.

Caches into dp_psac_ref/data/. Does NOT touch src/util/dataloaders.py.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from torchvision import datasets

DATA_ROOT = Path(__file__).parent / "data"


def _to_arrays(ds) -> tuple[np.ndarray, np.ndarray]:
    x = ds.data.numpy().astype(np.float32) / 255.0
    y = np.array(ds.targets, dtype=np.int32)
    return x, y


def load(name: str, arch: str) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns (x_train, y_train, x_test, y_test) as jnp arrays.

    If arch == "mlp": images flattened to (N, 784).
    If arch == "cnn": images shaped (N, 1, 28, 28).
    """
    DATA_ROOT.mkdir(exist_ok=True)
    name = name.lower()
    if name == "mnist":
        cls = datasets.MNIST
    elif name in ("fashion-mnist", "fashion_mnist", "fmnist"):
        cls = datasets.FashionMNIST
    else:
        raise ValueError(f"unsupported dataset: {name}")

    train_ds = cls(root=str(DATA_ROOT), train=True, download=True)
    test_ds = cls(root=str(DATA_ROOT), train=False, download=True)

    x_train, y_train = _to_arrays(train_ds)
    x_test, y_test = _to_arrays(test_ds)

    # Normalize to roughly zero mean / unit variance using train stats.
    mu, sd = x_train.mean(), x_train.std() + 1e-8
    x_train = (x_train - mu) / sd
    x_test = (x_test - mu) / sd

    if arch == "mlp":
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
    elif arch == "cnn":
        x_train = x_train[:, None, :, :]
        x_test = x_test[:, None, :, :]
    else:
        raise ValueError(f"unknown arch: {arch}")

    return jnp.asarray(x_train), jnp.asarray(y_train), jnp.asarray(x_test), jnp.asarray(y_test)
