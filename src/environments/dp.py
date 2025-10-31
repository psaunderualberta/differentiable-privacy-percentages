"""JAX Compatible  version of MountainCarContinuous-v0 OpenAI gym environment.


Source:
github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree, PRNGKeyArray, Array
import jax.random as jr
import optax
from util.util import (
    reinit_model,
    clip_grads_abadi,
    sample_batch_uniform,
    get_spherical_noise,
    subset_classification_accuracy,
)
from environments.losses import vmapped_loss, loss

from environments.dp_params import DP_RL_Params


def training_step(
    model: PyTree,
    optimizer: optax.GradientTransformation,
    opt_state: PyTree,
    mb_key: PRNGKeyArray,
    noise_key: PRNGKeyArray,
    noise: Array,
    params: DP_RL_Params,
) -> tuple[
    PyTree,
    optax.OptState,
    PRNGKeyArray,
    PRNGKeyArray,  # Carry values
    Array,
    Array,  # Scan outputs
]:
    mb_key, _key = jr.split(mb_key)
    batch_x, batch_y = sample_batch_uniform(
        params.X, params.y, params.dummy_batch, _key
    )

    new_loss, grads = vmapped_loss(model, batch_x, batch_y)
    clipped_grads = clip_grads_abadi(grads, params.C)

    # # Add spherical noise to gradients
    noise_key, _key = jr.split(noise_key)
    noises = get_spherical_noise(clipped_grads, noise, _key)
    noised_grads = eqx.apply_updates(clipped_grads, noises)

    # Add noisy gradients, update model and optimizer
    updates, new_opt_state = optimizer.update(noised_grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    # Subsample each with probability p
    mb_key, _key = jr.split(mb_key)
    accuracy = subset_classification_accuracy(new_model, params.X, params.y, 0.01, _key)

    return (new_model, new_opt_state, mb_key, noise_key, new_loss, accuracy)


@eqx.filter_jit
def train_with_noise(
    noise_schedule: jnp.ndarray,
    params: DP_RL_Params,
    mb_key: PRNGKeyArray,
    init_key: PRNGKeyArray,
    noise_key: PRNGKeyArray,
) -> tuple[eqx.Module, Array, Array]:
    # Create network
    network = reinit_model(params.network, init_key)
    net_params, net_static = eqx.partition(network, eqx.is_array)
    net_params = eqx.filter_jit(jax.lax.pvary)(net_params, "x")

    optimizer = getattr(optax, params.optimizer)(params.lr)
    opt_state = optimizer.init(net_params)
    opt_state_params, opt_state_static = eqx.partition(opt_state, eqx.is_array)
    opt_state_params = eqx.filter_jit(jax.lax.pvary)(opt_state_params, "x")

    @jax.checkpoint  # type:ignore
    def scanned_training_step(
        carry: tuple[PyTree, PyTree, PRNGKeyArray, PRNGKeyArray], noise: Array
    ) -> tuple[
        tuple[PyTree, optax.OptState, PRNGKeyArray, PRNGKeyArray],  # Carry values
        tuple[Array, Array],  # Scan outputs
    ]:
        net_params, opt_state_params, mb_key, noise_key = carry
        model = eqx.combine(net_params, net_static)
        opt_state = eqx.combine(opt_state_params, opt_state_static)

        new_model, new_opt_state, mb_key, noise_key, new_loss, accuracy = training_step(
            model, optimizer, opt_state, mb_key, noise_key, noise, params
        )

        new_net_params, _ = eqx.partition(new_model, eqx.is_array)
        opt_state_params, _ = eqx.partition(new_opt_state, eqx.is_array)
        return (new_net_params, new_opt_state, mb_key, noise_key), (new_loss, accuracy)

    initial_carry = (net_params, opt_state_params, mb_key, noise_key)
    (net_params, _, mb_key, noise_key), (losses, accuracies) = jax.lax.scan(
        scanned_training_step,
        initial_carry,
        xs=noise_schedule,
    )

    mb_key, batch_key = jr.split(mb_key)
    network_final = eqx.combine(net_params, net_static)
    batch_x, batch_y = sample_batch_uniform(
        params.X, params.y, params.dummy_batch, batch_key
    )
    final_loss, _ = loss(network_final, batch_x, batch_y)

    losses = jnp.concat([losses, jnp.asarray([final_loss])])

    return network_final, losses, accuracies


def get_noise_lookahead_grads(
    noise_schedule: jnp.ndarray,
    params: DP_RL_Params,
    mb_key: PRNGKeyArray,
    init_key: PRNGKeyArray,
    noise_key: PRNGKeyArray,
) -> tuple[eqx.Module, Array, Array]:
    filtered_pvary = eqx.filter_jit(jax.lax.pvary)

    # Create network
    network = reinit_model(params.network, init_key)
    net_params, net_static = eqx.partition(network, eqx.is_array)
    net_params = filtered_pvary(net_params, "x")

    optimizer = getattr(optax, params.optimizer)(params.lr)
    opt_state = optimizer.init(net_params)
    opt_state_params, opt_state_static = eqx.partition(opt_state, eqx.is_array)
    opt_state_params = eqx.filter_jit(jax.lax.pvary)(opt_state_params, "x")

    def scanned_lookahead(
        carry: tuple[PyTree, PyTree, PRNGKeyArray, PRNGKeyArray], noise: Array
    ) -> tuple[
        tuple[PyTree, optax.OptState, PRNGKeyArray, PRNGKeyArray],  # Carry values
        tuple[Array, Array],  # Scan outputs

    return network, jnp.zeros(1), jnp.zeros(1)
