"""JAX Compatible  version of MountainCarContinuous-v0 OpenAI gym environment.


Source:
github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py
"""

from typing import Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental import checkify
import jax.random as jr
import optax
from util.util import reinit_model, clip_grads_abadi, sample_batch_uniform, get_spherical_noise, subset_classification_accuracy
from environments.losses import vmapped_loss, loss

from environments.dp_params import DP_RL_Params


def train_with_noise(
    noise_schedule: jnp.ndarray,
    params: DP_RL_Params,
    mb_key: chex.PRNGKey,
    init_key: chex.PRNGKey,
    noise_key: chex.PRNGKey,
) -> Tuple[eqx.Module, chex.Array, chex.Array]:
    # Create network
    network = reinit_model(jax.lax.pvary(params.network, 'x'), init_key)
    optimizer = optax.sgd(params.lr)

    @jax.checkpoint  #type:ignore
    def training_step(
        carry,
        noise
    ) -> Tuple[
        Tuple[eqx.Module, optax.OptState, chex.PRNGKey, chex.PRNGKey],  # Carry values
        Tuple[chex.Array, chex.Array, eqx.Module, chex.PRNGKey, chex.PRNGKey]  # Scan outputs
    ]:
        model, opt_state, mb_key, noise_key = carry

        mb_key, _key = jr.split(mb_key)
        batch_x, batch_y = sample_batch_uniform(params.X, params.y, params.dummy_batch, _key)

        new_loss, grads = vmapped_loss(model, batch_x, batch_y)
        clipped_grads = clip_grads_abadi(grads, params.C)

        # Add spherical noise to gradients
        noise_key, _key = jr.split(noise_key)
        noises = get_spherical_noise(clipped_grads, noise, _key)
        noised_grads = eqx.apply_updates(clipped_grads, noises)

        # Add noisy gradients, update model and optimizer
        updates, new_opt_state = optimizer.update(
            noised_grads, opt_state, model
        )
        new_model = eqx.apply_updates(model, updates)

        # Subsample each with probability p
        mb_key, _key = jr.split(mb_key)
        accuracy = subset_classification_accuracy(
            new_model, params.X, params.y, 0.01, _key
        )

        return (new_model, new_opt_state, mb_key, noise_key), (new_loss, accuracy, new_model, noise_key, mb_key)

    initial_carry = (network, optimizer.init(network), mb_key, noise_key)
    (network, _, mb_key, noise_key), (losses, accuracies, networks, noise_keys, batch_keys) = jax.lax.scan(
        training_step,
        initial_carry,
        xs=noise_schedule,
    )

    mb_key, batch_key = jr.split(mb_key)
    batch_x, batch_y = sample_batch_uniform(params.X, params.y, params.dummy_batch, batch_key)
    final_loss, _ = loss(network, batch_x, batch_y)

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


    return network, losses, accuracies
    