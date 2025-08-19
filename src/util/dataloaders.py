import os
from typing import Tuple

import chex
import jax.numpy as jnp
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures

__DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def _dataloader_california(degree=1):
    X, y = fetch_california_housing(return_X_y=True)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(X)

    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y_classes = y < np.median(y)
    y = pd.get_dummies(y_classes).values.astype(np.int32)
    return jnp.asarray(X), jnp.asarray(y)


def _dataloader_mnist(_=None) -> Tuple[chex.Array, chex.Array]:
    mnist_datadir = os.path.join(__DATA_DIR, "mnist")
    os.makedirs(mnist_datadir, exist_ok=True)
    image_file = os.path.join(mnist_datadir, "images.npy")
    label_file = os.path.join(mnist_datadir, "labels.npy")
    if not os.path.exists(image_file) or not os.path.exists(label_file):
        ds = load_dataset("ylecun/mnist", split="train")
        # https://huggingface.co/docs/datasets/en/use_with_jax
        dds = ds.with_format("jax")
        images = dds["image"]
        labels = dds["label"]
        print(labels)
        labels = jnp.asarray(pd.get_dummies(labels).values).astype(jnp.int32)

        jnp.save(image_file, images)
        jnp.save(label_file, labels)

    images = jnp.load(image_file)
    labels = jnp.load(label_file)
    # convert 'labels' to one-hot encoding

    # normalize & flatten images
    images = images / 255.0
    images = images.reshape((images.shape[0], -1))

    return images, labels


def _dataloader_cifar_10(_=None) -> Tuple[chex.Array, chex.Array]:
    ds = load_dataset("uoft-cs/cifar10", split="train")
    # https://huggingface.co/docs/datasets/en/use_with_jax
    dds = ds.with_format("jax")
    images = dds["img"]
    labels = dds["label"]

    # convert 'labels' to one-hot encoding
    labels = jnp.asarray(pd.get_dummies(labels).values).astype(jnp.int32)

    # normalize & flatten images
    images = images / 255.0

    # images are n x n x c, should be c x n x n
    images = images.transpose((0, 3, 1, 2))

    return images, labels


DATALOADERS = {
    "california": _dataloader_california,
    "mnist": _dataloader_mnist,
    "cifar10": _dataloader_cifar_10,
}

if __name__ == "__main__":
    _dataloader_mnist()
