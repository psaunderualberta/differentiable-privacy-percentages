from functools import partial
from typing import Any, Callable, Optional, Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import tree as jt
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from optax import (global_norm, per_example_global_norm_clip,
                   softmax_cross_entropy)


@eqx.filter_jit
@eqx.filter_value_and_grad
def mse_loss(model: Callable[[chex.Array], jnp.ndarray], x: chex.Array, y: chex.Array):
    pred_y = jax.vmap(model)(x).squeeze()

    return jnp.mean((pred_y - y) ** 2)


@eqx.filter_jit
@eqx.filter_value_and_grad
def cce_loss(model: Callable[[chex.Array], jnp.ndarray], x: chex.Array, y: chex.Array):
    pred_y = jax.vmap(model)(x).squeeze()

    return softmax_cross_entropy(pred_y, y).mean()


@eqx.filter_jit
@eqx.filter_value_and_grad
def non_vmap_mse_loss(model: Callable[[chex.Array], jnp.ndarray], x: chex.Array, y: chex.Array):
    pred_y = model(x).squeeze()

    return jnp.mean((pred_y - y) ** 2)


@eqx.filter_jit
@eqx.filter_value_and_grad
def non_vmap_cce_loss(model: Callable[[chex.Array], jnp.ndarray], x: chex.Array, y: chex.Array):
    pred_y = model(x).squeeze()

    return softmax_cross_entropy(pred_y, y).mean()


@eqx.filter_jit
@eqx.filter_value_and_grad
def mse_loss_poisson(
    model: eqx.Module, x: chex.Array, y: chex.Array, key: chex.PRNGKey, idxs: chex.Array
):
    # get random subset of idxs for training
    key, _key = jr.split(key)
    probs = jr.uniform(_key, (x.shape[0],))

    # https://arxiv.org/abs/2206.14286 for implementation of approx_max_k
    jitted_approx_max_k = jax.jit(jax.lax.approx_max_k, static_argnums=(1,))
    _, subsample_idxs = jitted_approx_max_k(probs, idxs.shape[0])

    return jnp.mean(
        (jax.vmap(model)(x[subsample_idxs]).squeeze() - y[subsample_idxs]) ** 2
    )


@eqx.filter_jit
def cce_loss_poisson(
    model: eqx.Module, x: chex.Array, y: chex.Array, key: chex.PRNGKey, idxs: chex.Array
):
    # get random subset of idxs for training
    key, _key = jr.split(key)
    probs = jr.uniform(_key, (x.shape[0],))

    # https://arxiv.org/abs/2206.14286 for implementation of approx_max_k
    jitted_approx_max_k = jax.jit(jax.lax.approx_max_k, static_argnums=(1,))
    _, subsample_idxs = jitted_approx_max_k(probs, idxs.shape[0])

    return cce_loss(model, x[subsample_idxs], y[subsample_idxs])


@eqx.filter_jit
def dp_mse_loss(model: eqx.Module, x: chex.Array, y: chex.Array, C: float):
    losses, grads = jax.vmap(non_vmap_mse_loss, in_axes=(None, 0, 0))(model, x, y)
    # https://github.com/google-deepmind/optax/blob/main/optax/contrib/_privacy.py#L35#L87
    grads_flat, grads_treedef = jax.tree.flatten(grads)
    batch_size = grads_flat[0].shape[0]

    # computes "sum of the clipped per-example grads,"
    # per https://optax.readthedocs.io/en/latest/api/transformations.html#optax.per_example_global_norm_clip
    # NOTE: This computes the global norm over all layers, not layer-wise
    # sum_clipped, _ = per_example_global_norm_clip(grads_flat, C)

    # DP optimization as described in
    # https://proceedings.neurips.cc/paper_files/paper/2023/file/8249b30d877c91611fd8c7aa6ac2b5fe-Paper-Conference.pdf
    global_grad_norms = jax.vmap(global_norm)(grads)
    gamma = 0.01
    multipliers = j1 / (global_grad_norms + gamma)
    sum_clipped = jax.tree.map(
        lambda g: jnp.tensordot(multipliers, g, axes=1), grads_flat
    )

    return losses.mean(), jax.tree.unflatten(grads_treedef, sum_clipped)


@eqx.filter_jit
def dp_cce_loss(model: eqx.Module, x: chex.Array, y: chex.Array, C: float):
    losses, grads = jax.vmap(non_vmap_cce_loss, in_axes=(None, 0, 0))(model, x, y)

    # https://github.com/google-deepmind/optax/blob/main/optax/contrib/_privacy.py#L35#L87
    grads_flat, grads_treedef = jax.tree.flatten(grads)

    # computes "sum of the clipped per-example grads,"
    # per https://optax.readthedocs.io/en/latest/api/transformations.html#optax.per_example_global_norm_clip
    # NOTE: This computes the global norm over all layers, not layer-wise
    # sum_clipped, _ = per_example_global_norm_clip(grads_flat, C)

    # DP optimization as described in https://proceedings.neurips.cc/paper_files/paper/2023/file/8249b30d877c91611fd8c7aa6ac2b5fe-Paper-Conference.pdf
    global_grad_norms = jax.vmap(global_norm)(grads)
    multipliers = jnp.minimum(1 / global_grad_norms, 1.0)
    sum_clipped = jax.tree.map(
        lambda g: jnp.tensordot(multipliers, g, axes=1), grads_flat
    )

    return (
        losses.mean(),
        jax.tree.unflatten(grads_treedef, sum_clipped)
    )


@eqx.filter_jit
def dp_cce_loss_poisson(
    model: eqx.Module,
    x: chex.Array,
    y: chex.Array,
    key: chex.PRNGKey,
    idxs: chex.Array,
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

    return dp_cce_loss(model, _x, _y, C)


@eqx.filter_jit()
def hidden_node_gradients(
    model: eqx.Module,
    x: chex.Array,
    y: chex.Array,
    key: chex.PRNGKey,
    idxs: chex.Array,
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

    @partial(jax.jit, static_argnums=(1,))  # Might be problematic w/ larger networks
    @partial(jax.value_and_grad, has_aux=True)
    def f(block_in, block_pos):
        block_out = jax.vmap(model.forward_through_block, in_axes=(0, None))(
            block_in, block_pos
        )
        if block_pos == len(model.layers) - 1:
            aux = ()
            loss, grad = jax.value_and_grad(
                lambda inp: softmax_cross_entropy(inp, _y).mean()
            )(block_out)
        else:
            ((loss, aux), grad) = f(block_out, block_pos + 1)

        # normalizing grad
        norm = jnp.linalg.norm(grad)
        norm = jnp.maximum(1.0, norm / C)
        grad = grad / norm
        grad = jnp.mean(grad, axis=0)

        aux = (grad, *aux)
        return loss, aux

    return f(_x, 0)[0]  # don't care abt grad w.r.t. X


@eqx.filter_jit
def dp_mse_loss_poisson(
    model: eqx.Module,
    x: chex.Array,
    y: chex.Array,
    key: chex.PRNGKey,
    idxs: chex.Array,
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
def pytree_keys(model, key):
    treedef = jt.structure(model)
    keys = jr.split(key, treedef.num_leaves)
    return jt.unflatten(treedef, keys)


@eqx.filter_jit
def is_none(x):
    return x is None


@eqx.filter_jit
def reinit_model(model, key):
    return model.reinitialize(key)


@eqx.filter_jit
def add_spherical_noise(
    grads: jax.Array, action: chex.Array, key: chex.PRNGKey
):
    def add_the_noise(g, k):
        if g is None:
            return g
        normal_noise = jax.random.normal(k, g.shape, g.dtype)
        # TODO: NaNs are caused by values > 1 in normal_noise, as they result
        # in exponential growth in gradient magnitudes for the policy. 
        return g + action * normal_noise
    return jt.map(add_the_noise, grads, pytree_keys(grads, key))


# @jax.jit
def baseline_model(env, variance):
    num_iters = 0
    variance = jnp.array([variance])

    if variance == 0.0:
        rewards = [env.step(variance, private=False)[1] for _ in range(500)]
        num_iters = 500
    else:
        terminated = False
        rewards = []
        while not terminated:
            num_iters += 1
            _, reward, terminated, _, _ = env.step(variance)
            rewards.append(reward)

    return env, rewards, num_iters


def index_vmapped(structure, index):
    return jt.map(lambda x: x[index], structure)


def subset_classification_accuracy(model, x, y, percent, key):
    num_samples = x.shape[0]
    num_samples = int(num_samples * percent)
    idxs = jr.permutation(key, jnp.arange(x.shape[0]))[:num_samples]
    return classification_accuracy(model, x[idxs], y[idxs])


@eqx.filter_jit
def classification_accuracy(model, x, y):
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


def str_to_jnp_array(s: str, sep: str = ", ", with_brackets: bool = True) -> chex.Array:
    if with_brackets:
        s = s[1:-1]
    return jnp.asarray(np.fromstring(s, dtype=float, sep=sep))


def determine_optimal_num_devices(devices, num_training_runs, printing=True) -> Tuple[NamedSharding, int]:
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
