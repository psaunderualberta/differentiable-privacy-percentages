import ast
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
    """Poisson-subsample a batch using approximate top-k on uniform random values.

    Args:
        x: Full dataset features of shape (N, *).
        y: Full dataset labels of shape (N, *).
        idxs: Dummy array whose length determines the batch size.
        key: PRNG key for sampling.

    Returns:
        Tuple of (batch_x, batch_y) with batch_size rows each.
    """
    assert len(x.shape) > 0 and len(idxs.shape) > 0 and len(y.shape) > 0
    # get random subset of idxs for training
    key, _key = jr.split(key)
    probs = jr.uniform(_key, (x.shape[0],))

    # https://arxiv.org/abs/2206.14286 for implementation of approx_max_k
    # jitted_approx_max_k = jax.lax.approx_max_k, static_argnums=(1,))
    _, subsample_idxs = jax.lax.approx_max_k(probs, idxs.shape[0])

    return x[subsample_idxs], y[subsample_idxs]  # type: ignore


@eqx.filter_jit
def clip_grads_abadi(grads: eqx.Module, C: Array) -> eqx.Module:
    """Clip per-example gradients using the Abadi smooth global-norm clipping rule.

    Applies the differentially-private clipping scheme from
    https://proceedings.neurips.cc/paper_files/paper/2023/file/8249b30d877c91611fd8c7aa6ac2b5fe-Paper-Conference.pdf,
    computing the global L2 norm over all layers and returning mean clipped gradients.

    Args:
        grads: Per-example gradient pytree with a leading batch dimension.
        C: Clipping threshold.

    Returns:
        Mean clipped gradient pytree (batch dimension removed).
    """
    # https://github.com/google-deepmind/optax/blob/main/optax/contrib/_privacy.py#L35#L87
    grads_flat, grads_treedef = jax.tree.flatten(grads)

    # computes "sum of the clipped per-example grads,"
    # per https://optax.readthedocs.io/en/latest/api/transformations.html#optax.per_example_global_norm_clip
    # NOTE: This computes the global norm over all layers, not layer-wise
    # sum_clipped, _ = per_example_global_norm_clip(grads_flat, C)

    # DP optimization as described in https://proceedings.neurips.cc/paper_files/paper/2023/file/8249b30d877c91611fd8c7aa6ac2b5fe-Paper-Conference.pdf

    def get_multiplier(grad: eqx.Module):
        l22_norm = jnp.sqrt(
            1e-8 + sum(jnp.sum(jnp.abs(x) ** 2) for x in jax.tree.leaves(grad))
        )
        return C / (l22_norm + 1 / (l22_norm + 1))

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
    """Subsample a batch uniformly at random using approximate top-k.

    Args:
        x: Full dataset features of shape (N, *).
        y: Full dataset labels of shape (N, *).
        key: PRNG key for sampling.
        idxs: Dummy array whose length determines the batch size.

    Returns:
        Tuple of (batch_x, batch_y) with batch_size rows each.
    """
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
    """Compute the DP MSE loss on a Poisson-subsampled minibatch.

    Args:
        model: The model to evaluate.
        x: Full dataset features of shape (N, *).
        y: Full dataset labels of shape (N, *).
        key: PRNG key for minibatch subsampling.
        idxs: Dummy array whose length determines the batch size.
        C: Gradient clipping threshold (passed to dp_mse_loss).

    Returns:
        Scalar MSE loss on the subsampled batch.
    """
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
    """Return a pytree of PRNG keys with the same structure as `model`.

    Args:
        model: Reference pytree whose structure is mirrored.
        key: Base PRNG key split into one key per leaf.
    """
    treedef = jt.structure(model)
    keys = jr.split(key, treedef.num_leaves)
    return jt.unflatten(treedef, keys)


@eqx.filter_jit
def is_none(x: Array | None) -> bool:
    """Return True if `x` is None."""
    return x is None


@eqx.filter_jit
def reinit_model(model: eqx.Module, key: PRNGKeyArray) -> eqx.Module:
    """Re-randomize a model's weights by calling its `reinitialize` method.

    Args:
        model: Model to reinitialize (must implement `reinitialize(key)`).
        key: PRNG key for new weight draws.
    """
    return model.reinitialize(key)


@eqx.filter_jit
def get_spherical_noise(
    grads: eqx.Module, action: float | Array, clip: float | Array, key: PRNGKeyArray
) -> eqx.Module:
    """Generate isotropic Gaussian noise scaled by `action` / batch_size for DP-SGD.

    Args:
        grads: Gradient pytree used to determine shapes and dtypes.
        action: Per-step noise level σ (standard deviation multiplier).
        clip: Unused in the noise computation; kept for API symmetry.
        key: PRNG key used to seed noise generation.

    Returns:
        Pytree of noise arrays matching the structure and shapes of `grads`.
    """
    batch_size = SingletonConfig.get_environment_config_instance().batch_size

    def f(g: eqx.Module | None, k: PRNGKeyArray):
        if g is None:
            return g
        return action * jax.random.normal(k, g.shape, g.dtype) / batch_size

    return jt.map(f, grads, pytree_keys(grads, key))


@eqx.filter_jit
def add_pytrees(a: PyTree, b: PyTree) -> PyTree:
    """Element-wise add two pytrees with the same structure."""
    return eqx.apply_updates(a, b)


@eqx.filter_jit
def subtract_pytrees(a: PyTree, b: PyTree) -> PyTree:
    """Element-wise subtract pytree `b` from `a`; returns None for None leaves."""
    def func(x, y):
        if x is None or y is None:
            return None
        return x - y

    return jt.map(func, a, b)


@eqx.filter_jit
def multiply_pytree_by_scalar(pytree: PyTree, scalar: PyTree) -> PyTree:
    """Multiply every leaf of `pytree` by `scalar`; None leaves are passed through."""
    return jt.map(lambda x: x * scalar if x is not None else None, pytree)


@eqx.filter_jit
def dot_pytrees(a: PyTree, b: PyTree) -> PyTree:
    """Compute the scalar dot product between two pytrees (sum of leaf-wise inner products).

    None leaves contribute 0.
    """
    def func(x, y):
        if x is None or y is None:
            return 0.0
        return jnp.sum(x * y, axis=None)

    return jt.reduce(lambda x, y: x + y, jt.map(func, a, b), 0.0)


@eqx.filter_jit
def pytree_max(a: PyTree) -> PyTree:
    """Return the global maximum scalar value across all leaves of `a`.

    None leaves are treated as -100.0 so they never win.
    """
    def func(x):
        if x is None:
            return -100.0
        return jnp.max(x, axis=None)

    return jt.reduce(lambda x, y: jnp.maximum(x, y), jt.map(func, a), 0.0)


def index_pytree(structure: PyTree, index: int) -> PyTree:
    """Index every array leaf of `structure` along its first axis at position `index`.

    Args:
        structure: Pytree whose array leaves have a leading dimension to index into.
        index: Position along the first axis to select.

    Returns:
        Pytree with the same structure but each array leaf replaced by `leaf[index]`.
    """
    def f(x):
        if x is None:
            return None
        return x[index]

    return jt.map(f, structure)


def pytree_has_nan(tree: PyTree) -> Array:
    """Return a scalar boolean array that is True if any leaf of `tree` contains NaN."""
    _false = jnp.asarray(False)

    def f(t: PyTree) -> Array:
        if t is None:
            return _false
        return jnp.isnan(t).any(axis=None)

    return jt.reduce(lambda x, y: jnp.logical_or(x, y), jt.map(f, tree), _false)


def pytree_has_inf(tree: PyTree) -> Array:
    """Return a scalar boolean array that is True if any leaf of `tree` contains ±Inf."""
    _false = jnp.asarray(False)

    def f(t: PyTree) -> Array:
        if t is None:
            return _false
        return jnp.isinf(t).any(axis=None)

    return jt.reduce(lambda x, y: jnp.logical_or(x, y), jt.map(f, tree), _false)


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
    """Compute classification accuracy on a random subset of the dataset.

    Args:
        model: Classifier to evaluate.
        x: Full dataset features of shape (N, *).
        y: One-hot labels of shape (N, nclasses).
        percent: Fraction of examples to sample (e.g. 0.01 for 1 %).
        key: PRNG key for subset selection.

    Returns:
        Scalar accuracy in [0, 100].
    """
    num_samples = x.shape[0]
    num_samples = int(num_samples * percent)
    idxs = jr.permutation(key, jnp.arange(x.shape[0]))[:num_samples]
    return classification_accuracy(model, x[idxs], y[idxs])


@eqx.filter_jit
def classification_accuracy(model: eqx.Module, x: Array, y: Array) -> Array:
    """Compute classification accuracy as a percentage over the full provided split.

    Args:
        model: Classifier whose output is vmapped over examples.
        x: Features of shape (N, *).
        y: One-hot labels of shape (N, nclasses).

    Returns:
        Scalar accuracy in [0, 100].
    """
    pred_y = jax.vmap(model)(x).squeeze()
    pred_y_int = jnp.argmax(pred_y, axis=1)
    y_int = jnp.argmax(y, axis=1)
    return jnp.mean(pred_y_int == y_int) * 100


def recursive_list_to_jnp_array(obj: dict | list | str | int):
    """Recursively convert lists inside a nested dict/list structure to jnp arrays.

    Args:
        obj: Nested Python dict, list, or scalar value.

    Returns:
        The same structure with every `list` replaced by a `jnp.array`.
    """
    if isinstance(obj, dict):
        return {k: recursive_list_to_jnp_array(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return jnp.array(obj)
    else:
        return obj


def str_to_jnp_array(s: str, sep: str = ", ", with_brackets: bool = True) -> Array:
    """Parse a string representation of a Python literal into a JAX array.

    Args:
        s: String to parse (e.g. '[1, 2, 3]' or '1, 2, 3').
        sep: Separator string (currently unused; kept for API compatibility).
        with_brackets: If True, strip the leading and trailing characters before parsing.

    Returns:
        JAX array of the parsed values.
    """
    if with_brackets:
        s = s[1:-1]
    return jnp.asarray(ast.literal_eval(s))


def determine_optimal_num_devices(
    devices, num_training_runs, printing=True
) -> tuple[NamedSharding, int]:
    """Maximize # of devices s.t. |runs| % |devices| = 0
    (otherwise JAX will throw an error when trying to distribute runs).

    Args:
        devices: List of JAX devices available.
        num_training_runs: Total number of parallel training runs.
        printing: If True, print the selected device subset.

    Returns:
        Tuple of (NamedSharding for the chosen mesh, number of devices used).
    """
    max_num_devices = min(len(devices), num_training_runs)
    trimmed_devices_ = devices[:1]
    for num_devices in range(1, max_num_devices + 1):
        if num_training_runs % num_devices == 0:
            trimmed_devices_ = devices[:num_devices]
    if printing:
        print("Given devices: ", devices)
        print("Using devices: ", trimmed_devices_)
    mesh = Mesh(trimmed_devices_, ("i",))
    return NamedSharding(mesh, P("i")), len(trimmed_devices_)


def get_optimal_mesh(devices_, num_training_runs, printing=True):
    """Build a JAX Mesh with the optimal number of devices for `num_training_runs`.

    Args:
        devices_: List of JAX devices available.
        num_training_runs: Total number of parallel training runs.
        printing: If True, print the selected device subset.

    Returns:
        A `jax.sharding.Mesh` over the chosen devices with axis name 'x'.
    """
    _, num_gpus = determine_optimal_num_devices(
        devices_, num_training_runs, printing=printing
    )
    return Mesh(devices_[:num_gpus], "x")
