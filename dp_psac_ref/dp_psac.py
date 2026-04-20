"""DP-PSAC (Xia et al. 2023) reference implementation.

Bare-bones. Accepts per-step sigma and clip schedules of arbitrary length T.
No dependencies on src/.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Float, PRNGKeyArray, PyTree

# Activation choice matters under DP. Papernot et al. 2021 ("Tempered Sigmoid
# Activations") show tanh >> ReLU for DP-SGD because per-sample gradient norms
# stay bounded. Tramèr & Boneh 2020 and the DP-PSAC paper both use tanh CNNs.
_ACT = jax.nn.tanh


class MLP(eqx.Module):
    layers: list

    def __init__(self, in_dim: int, hidden: int, out_dim: int, key: PRNGKeyArray):
        k1, k2, k3 = jr.split(key, 3)
        self.layers = [
            eqx.nn.Linear(in_dim, hidden, key=k1),
            eqx.nn.Linear(hidden, hidden, key=k2),
            eqx.nn.Linear(hidden, out_dim, key=k3),
        ]

    def __call__(self, x: Float[Array, " in_dim"]) -> Float[Array, " out_dim"]:  # noqa: F722
        x = _ACT(self.layers[0](x))
        x = _ACT(self.layers[1](x))
        return self.layers[2](x)


class CNN(eqx.Module):
    """Tramèr-Boneh 2020 CNN for MNIST/FashionMNIST — tanh activations."""

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, in_channels: int, out_dim: int, key: PRNGKeyArray):
        k1, k2, k3, k4 = jr.split(key, 4)
        # 28 --conv(k=8,s=2,p=3)--> 14 --pool(k=2,s=1)--> 13
        self.conv1 = eqx.nn.Conv2d(in_channels, 16, 8, stride=2, padding=3, key=k1)
        # 13 --conv(k=4,s=2,p=0)--> 5 --pool(k=2,s=1)--> 4
        self.conv2 = eqx.nn.Conv2d(16, 32, 4, stride=2, padding=0, key=k2)
        self.fc1 = eqx.nn.Linear(32 * 4 * 4, 32, key=k3)
        self.fc2 = eqx.nn.Linear(32, out_dim, key=k4)

    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, " out_dim"]:  # noqa: F722
        x = _ACT(self.conv1(x))
        x = eqx.nn.MaxPool2d(2, 1)(x)
        x = _ACT(self.conv2(x))
        x = eqx.nn.MaxPool2d(2, 1)(x)
        x = x.reshape(-1)
        x = _ACT(self.fc1(x))
        return self.fc2(x)


def sample_loss(model: eqx.Module, x: Array, y: Array) -> Array:
    logits = model(x)
    return -jax.nn.log_softmax(logits)[y]


def _flatten_per_sample(tree: PyTree) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.concatenate([leaf.reshape(leaf.shape[0], -1) for leaf in leaves], axis=1)


def psac_clip(per_sample_grads: PyTree, C: Array, r: float) -> PyTree:
    """Apply DP-PSAC per-sample clipping.

    g̃_i = C · g_i / (||g_i|| + r / (||g_i|| + r))
    """
    flat = _flatten_per_sample(per_sample_grads)
    norms = jnp.linalg.norm(flat, axis=1)
    denom = norms + r / (norms + r)
    scale = C / denom
    return jax.tree_util.tree_map(
        lambda g: g * scale.reshape((-1,) + (1,) * (g.ndim - 1)),
        per_sample_grads,
    )


def add_gaussian_noise(summed_grads: PyTree, C: Array, sigma: Array, key: PRNGKeyArray) -> PyTree:
    leaves, treedef = jax.tree_util.tree_flatten(summed_grads)
    keys = jr.split(key, len(leaves))
    noised = [g + C * sigma * jr.normal(k, g.shape) for g, k in zip(leaves, keys)]
    return jax.tree_util.tree_unflatten(treedef, noised)


def _sample_indices(key: PRNGKeyArray, n: int, batch_size: int) -> Array:
    # Uniform sampling with replacement (per paper Algorithm 1 line 3).
    return jr.randint(key, (batch_size,), 0, n)


def make_train_step(
    model_static: PyTree,
    optimizer: optax.GradientTransformation,
    batch_size: int,
    r: float,
    x_train: Array,
    y_train: Array,
) -> Callable:
    """Build a JIT-compiled single step closed over the dataset."""

    def loss_for_sample(params: PyTree, x: Array, y: Array) -> Array:
        model = eqx.combine(params, model_static)
        return sample_loss(model, x, y)

    per_sample_grad_fn = jax.vmap(jax.grad(loss_for_sample), in_axes=(None, 0, 0))

    @eqx.filter_jit
    def step(
        params: PyTree,
        opt_state: PyTree,
        sigma_t: Array,
        C_t: Array,
        key: PRNGKeyArray,
    ) -> tuple[PyTree, PyTree, Array]:
        sample_key, noise_key = jr.split(key)
        idx = _sample_indices(sample_key, x_train.shape[0], batch_size)
        xb, yb = x_train[idx], y_train[idx]

        per_sample = per_sample_grad_fn(params, xb, yb)
        clipped = psac_clip(per_sample, C_t, r)
        summed = jax.tree_util.tree_map(lambda g: g.sum(0), clipped)
        noisy = add_gaussian_noise(summed, C_t, sigma_t, noise_key)
        avg = jax.tree_util.tree_map(lambda g: g / batch_size, noisy)

        updates, opt_state = optimizer.update(avg, opt_state, params)
        params = eqx.apply_updates(params, updates)

        # Mean per-sample loss, for logging (computed on the same batch).
        train_loss = jax.vmap(loss_for_sample, in_axes=(None, 0, 0))(params, xb, yb).mean()
        return params, opt_state, train_loss

    return step


def train(
    model: eqx.Module,
    x_train: Array,
    y_train: Array,
    x_test: Array,
    y_test: Array,
    sigmas: Array,
    clips: Array,
    batch_size: int,
    lr: float,
    r: float,
    key: PRNGKeyArray,
    log_every: int = 50,
) -> tuple[eqx.Module, dict]:
    assert sigmas.shape == clips.shape and sigmas.ndim == 1
    T = int(sigmas.shape[0])

    params, model_static = eqx.partition(model, eqx.is_inexact_array)
    optimizer = optax.sgd(lr, momentum=0.9)
    opt_state = optimizer.init(params)

    step = make_train_step(model_static, optimizer, batch_size, r, x_train, y_train)

    keys = jr.split(key, T)
    train_losses = []
    for t in range(T):
        params, opt_state, train_loss = step(params, opt_state, sigmas[t], clips[t], keys[t])
        train_losses.append(float(train_loss))
        if (t + 1) % log_every == 0 or t == T - 1:
            print(f"step {t + 1:>5}/{T}  loss={train_loss:.4f}")

    trained = eqx.combine(params, model_static)
    test_acc = evaluate(trained, x_test, y_test)
    return trained, {
        "test_accuracy": test_acc,
        "final_train_loss": train_losses[-1] if train_losses else float("nan"),
        "train_losses": train_losses,
    }


@eqx.filter_jit
def evaluate(model: eqx.Module, x: Array, y: Array) -> Array:
    preds = jax.vmap(model)(x).argmax(axis=-1)
    return (preds == y).mean()
