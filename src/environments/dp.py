"""JAX Compatible  version of MountainCarContinuous-v0 OpenAI gym environment.


Source:
github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py
"""

import chex
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


@eqx.filter_jit
def train_with_noise(
    noise_schedule: jnp.ndarray,
    params: DP_RL_Params,
    mb_key: chex.PRNGKey,
    init_key: chex.PRNGKey,
    noise_key: chex.PRNGKey,
) -> tuple[eqx.Module, chex.Array, chex.Array]:
    # Create network
    network = reinit_model(params.network, init_key)
    net_params, net_static = eqx.partition(network, eqx.is_array)
    net_params = eqx.filter_jit(jax.lax.pvary)(net_params, "x")

    optimizer = optax.sgd(params.lr)
    opt_state = optimizer.init(network)
    opt_state_params, opt_state_static = eqx.partition(opt_state, eqx.is_array)
    opt_state_params = eqx.filter_jit(jax.lax.pvary)(opt_state_params, "x")

    @jax.checkpoint  # type:ignore
    def training_step(
        carry: tuple[PyTree, PyTree, PRNGKeyArray, PRNGKeyArray], noise: Array
    ) -> tuple[
        tuple[PyTree, optax.OptState, chex.PRNGKey, chex.PRNGKey],  # Carry values
        tuple[chex.Array, chex.Array],  # Scan outputs
    ]:
        net_params, opt_state_params, mb_key, noise_key = carry
        model = eqx.combine(net_params, net_static)
        opt_state = eqx.combine(opt_state_params, opt_state_static)

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
        accuracy = subset_classification_accuracy(
            new_model, params.X, params.y, 0.01, _key
        )

        new_net_params, _ = eqx.partition(new_model, eqx.is_array)
        opt_state_params, _ = eqx.partition(new_opt_state, eqx.is_array)
        return (new_net_params, new_opt_state, mb_key, noise_key), (new_loss, accuracy)

    initial_carry = (net_params, opt_state_params, mb_key, noise_key)
    (net_params, _, mb_key, noise_key), (losses, accuracies) = jax.lax.scan(
        training_step,
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

    # ### Manual Computation of approximated gradients ###
    # T = noise_schedule.size

    # def compute_grad_loop(i, carry):
    #     i = T - i - 1
    #     derivatives, prod = carry
    #     noise_key = noise_keys[i, :]
    #     batch_key = batch_keys[i, :]

    #     w_i = index_pytree(networks, i)
    #     n_i = get_spherical_noise(w_i, noise_schedule[i], noise_key)
    #     batch_x, batch_y = sample_batch_uniform(params.X, params.y, params.dummy_batch, batch_key)

    #     derivatives = derivatives.at[i].set(
    #         dot_pytrees(prod, n_i)
    #     )

    #     prod = subtract_pytrees(
    #         prod,
    #         multiply_pytree_by_scalar(
    #             neural_net_gnhvp(w_i, batch_x, batch_y, prod),
    #             params.lr
    #         )
    #     )

    #     return (derivatives, prod)

    # derivatives = jnp.zeros_like(noise_schedule)
    # prod = multiply_pytree_by_scalar(final_grads, -params.lr)
    # derivatives, _ = jax.lax.fori_loop(
    #     0,
    #     T,
    #     compute_grad_loop,
    #     (derivatives, prod)
    # )

    return network_final, losses, accuracies
