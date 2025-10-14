from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import tree as jt
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PRNGKeyArray, PyTree, PyTreeDef
from conf.singleton_conf import SingletonConfig


@eqx.filter_jit
def sample_batch_uniform(
    x: Array,
    y: Array,
    idxs: Array,
    key: PRNGKeyArray,
):
    assert len(x.shape) > 0 and len(idxs.shape) > 0 and len(y.shape) > 0
    # get random subset of idxs for training
    key, _key = jr.split(key)
    probs = jr.uniform(_key, (x.shape[0],))

    # https://arxiv.org/abs/2206.14286 for implementation of approx_max_k
    # jitted_approx_max_k = jax.lax.approx_max_k, static_argnums=(1,))
    _, subsample_idxs = jax.lax.approx_max_k(probs, idxs.shape[0])

    return x[subsample_idxs], y[subsample_idxs]  # type: ignore


@eqx.filter_jit
def clip_grads_abadi(grads: eqx.Module, C: float) -> eqx.Module:
    # https://github.com/google-deepmind/optax/blob/main/optax/contrib/_privacy.py#L35#L87
    grads_flat, grads_treedef = jax.tree.flatten(grads)

    # computes "sum of the clipped per-example grads,"
    # per https://optax.readthedocs.io/en/latest/api/transformations.html#optax.per_example_global_norm_clip
    # NOTE: This computes the global norm over all layers, not layer-wise
    # sum_clipped, _ = per_example_global_norm_clip(grads_flat, C)

    # DP optimization as described in https://proceedings.neurips.cc/paper_files/paper/2023/file/8249b30d877c91611fd8c7aa6ac2b5fe-Paper-Conference.pdf
    grads = ensure_valid_pytree(grads, "grads in clip_grads")
    gamma = 0.01

    def get_multiplier(grad: eqx.Module):
        return 1 / (
            jnp.sqrt(
                1e-8 + sum(jnp.sum(jnp.abs(x) ** 2) for x in jax.tree.leaves(grad))
            )
            + gamma
        )


    batch_size = SingletonConfig.get_environment_config_instance().batch_size
    multipliers = jax.vmap(get_multiplier)(grads) / batch_size
    mean_clipped = jax.tree.map(
        lambda g: jnp.tensordot(multipliers, g, axes=1), grads_flat
    )

    return jax.tree.unflatten(grads_treedef, mean_clipped)


@eqx.filter_jit
def uniform_subsample_batch(
    x: Array,
    y: Array,
    key: PRNGKeyArray,
    idxs: Array,
):
    # get random subset of idxs for training
    key, _key = jr.split(key)
    probs = jr.uniform(_key, (x.shape[0],))

    # https://arxiv.org/abs/2206.14286 for implementation of approx_max_k
    # jitted_approx_max_k = jax.lax.approx_max_k, static_argnums=(1,))
    _, subsample_idxs = jax.lax.approx_max_k(probs, idxs.shape[0])
    _x = x[subsample_idxs]
    _y = y[subsample_idxs]

    return _x, _y


@eqx.filter_jit
def dp_mse_loss_poisson(
    model: eqx.Module,
    x: Array,
    y: Array,
    key: PRNGKeyArray,
    idxs: Array,
    C: float,
):
    # get random subset of idxs for training
    key, _key = jr.split(key)
    probs = jr.uniform(_key, (x.shape[0],))

    # https://arxiv.org/abs/2206.14286 for implementation of approx_max_k
    # jitted_approx_max_k = jax.lax.approx_max_k, static_argnums=(1,))
    _, subsample_idxs = jax.lax.approx_max_k(probs, idxs.shape[0])
    _x = x[subsample_idxs]
    _y = y[subsample_idxs]

    return dp_mse_loss(model, _x, _y, C)


# Create random PRNG keys w/ same pytree structure as model
@eqx.filter_jit
def pytree_keys(model: eqx.Module, key: PRNGKeyArray) -> PRNGKeyArray:
    treedef = jt.structure(model)
    keys = jr.split(key, treedef.num_leaves)
    return jt.unflatten(treedef, keys)


@eqx.filter_jit
def is_none(x: Array | None) -> bool:
    return x is None


@eqx.filter_jit
def reinit_model(model: eqx.Module, key: PRNGKeyArray) -> eqx.Module:
    return model.reinitialize(key)


@eqx.filter_jit
def get_spherical_noise(
    grads: eqx.Module, action: float, key: PRNGKeyArray
) -> eqx.Module:

    batch_size = SingletonConfig.get_environment_config_instance().batch_size

    def f(g: eqx.Module | None, k: PRNGKeyArray):
        if g is None:
            return g
        return action * jax.random.normal(k, g.shape, g.dtype) / batch_size

    return jt.map(f, grads, pytree_keys(grads, key))


@eqx.filter_jit
def add_pytrees(a: PyTree, b: PyTree) -> PyTree:
    return eqx.apply_updates(a, b)


@eqx.filter_jit
def subtract_pytrees(a: PyTree, b: PyTree) -> PyTree:
    def func(x, y):
        if x is None or y is None:
            return None
        return x - y

    return jt.map(func, a, b)


@eqx.filter_jit
def multiply_pytree_by_scalar(pytree: PyTree, scalar: PyTree) -> PyTree:
    return jt.map(lambda x: x * scalar if x is not None else None, pytree)


@eqx.filter_jit
def dot_pytrees(a: PyTree, b: PyTree) -> PyTree:
    def func(x, y):
        if x is None or y is None:
            return 0.0
        return jnp.sum(x * y, axis=None)

    return jt.reduce(lambda x, y: x + y, jt.map(func, a, b), 0.0)


@eqx.filter_jit
def pytree_max(a: PyTree) -> PyTree:
    def func(x):
        if x is None:
            return -100.0
        return jnp.max(x, axis=None)

    return jt.reduce(lambda x, y: jnp.maximum(x, y), jt.map(func, a), 0.0)


def index_pytree(structure: PyTree, index: int) -> PyTree:
    def f(x):
        if x is None:
            return None
        return x[index]

    return jt.map(f, structure)


def pytree_has_nan(tree: PyTree) -> Array:
    def f(t: PyTree) -> bool:
        if t is None:
            return None
        return jnp.isnan(t).any(axis=None)

    return jt.reduce(lambda x, y: jnp.logical_or(x, y), jt.map(f, tree), False)


def pytree_has_inf(tree: PyTree) -> Array:
    def f(t: PyTree) -> bool:
        if t is None:
            return None
        return ~jnp.isfinite(t).any(axis=None)

    return jt.reduce(lambda x, y: jnp.logical_or(x, y), jt.map(f, tree), False)


def ensure_valid_pytree(tree: PyTree, tree_name: str) -> PyTree:
    """
    Ensures PyTree does not have any Inf values or NaN values using eqx.error_if

    Args:
        Tree: PyTree to check

    Returns:
        Tree: Original Pytree unmodified, must be used to ensure DCE isn't applied to this function.
    """
    tree = eqx.error_if(
        tree, pytree_has_inf(tree), "Tree '" + tree_name + "' has infinite values!"
    )
    tree = eqx.error_if(
        tree, pytree_has_nan(tree), "Tree '" + tree_name + "' has NaN values!"
    )
    return tree


def subset_classification_accuracy(
    model: eqx.Module, x: Array, y: Array, percent: float, key: PRNGKeyArray
) -> Array:
    num_samples = x.shape[0]
    num_samples = int(num_samples * percent)
    idxs = jr.permutation(key, jnp.arange(x.shape[0]))[:num_samples]
    return classification_accuracy(model, x[idxs], y[idxs])


@eqx.filter_jit
def classification_accuracy(model: eqx.Module, x: Array, y: Array) -> Array:
    pred_y = jax.vmap(model)(x).squeeze()
    pred_y_int = jnp.argmax(pred_y, axis=1)
    y_int = jnp.argmax(y, axis=1)
    return jnp.mean(pred_y_int == y_int) * 100


def recursive_list_to_jnp_array(obj: dict | list | str | int):
    if isinstance(obj, dict):
        return {k: recursive_list_to_jnp_array(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return jnp.array(obj)
    else:
        return obj


def str_to_jnp_array(s: str, sep: str = ", ", with_brackets: bool = True) -> Array:
    if with_brackets:
        s = s[1:-1]
    return jnp.asarray(np.fromstring(s, dtype=float, sep=sep))


def determine_optimal_num_devices(
    devices, num_training_runs, printing=True
) -> tuple[NamedSharding, int]:
    """
    Maximize # of devices s.t. |runs| % |devices| = 0
    (otherwise JAX will throw an error when trying to distribute runs)
    """
    max_num_devices = min(len(devices), num_training_runs)
    trimmed_devices_ = devices[:1]
    for num_devices in range(1, max_num_devices + 1):
        if num_training_runs % num_devices == 0:
            trimmed_devices_ = devices[:num_devices]
    if printing:
        print("Using devices: ", trimmed_devices_)
    mesh = Mesh(trimmed_devices_, ("i",))
    return NamedSharding(mesh, P("i")), len(trimmed_devices_)
