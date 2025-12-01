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

def image_ds_saver(ds, x_file, y_file):
    # https://huggingface.co/docs/datasets/en/use_with_jax
    dds = ds.with_format("jax")
    image_key = "image" if "image" in dds.features else "img"
    images = dds[image_key]
    labels = dds["label"]
    pd_labels = pd.Series(labels).astype(int) # type: ignore
    labels = jnp.asarray(pd.get_dummies(pd_labels).values).astype(jnp.float32)

    jnp.save(x_file, images)  # type: ignore
    jnp.save(y_file, labels)  # type: ignore

def _dataloader_california(degree=1):
    X, y = fetch_california_housing(return_X_y=True)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    X = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(X)

    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y_classes = y < np.median(y)
    y = pd.get_dummies(y_classes).values.astype(np.int32)
    return jnp.asarray(X), jnp.asarray(y)


def _dataloader_mnist(_=None, test=False) -> Tuple[chex.Array, chex.Array]:
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


def _dataloader_cifar_10(_=None, test=False) -> Tuple[chex.Array, chex.Array]:
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


def _dataloader_fashion_mnist(_=None, test=False) -> Tuple[chex.Array, chex.Array]:
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
}

if __name__ == "__main__":
    _dataloader_mnist()
