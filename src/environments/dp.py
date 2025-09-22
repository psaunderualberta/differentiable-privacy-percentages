"""JAX Compatible  version of MountainCarContinuous-v0 OpenAI gym environment.


Source:
github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py
"""

from typing import Tuple

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from util.util import reinit_model, clip_grads_abadi, sample_batch_uniform, get_spherical_noise, subset_classification_accuracy
from environments.losses import vmapped_loss, loss

from environments.dp_params import DP_RL_Params


@jax.jit
def train_with_noise(
    noise_schedule: chex.Array,
    params: DP_RL_Params,
    key: chex.PRNGKey
) -> Tuple[eqx.Module, chex.Array, chex.Array]:
    # Create key
    key, _key = jr.split(key)

    # Create network
    network = reinit_model(params.network, _key)
    optimizer = optax.sgd(params.lr)

    @jax.checkpoint #type: ignore
    def training_step(
        carry,
        noise
    ) -> Tuple[
        Tuple[eqx.Module, optax.OptState, chex.PRNGKey],  # Carry values
        Tuple[chex.Array, chex.Array, eqx.Module, chex.PRNGKey, chex.PRNGKey]  # Scan outputs
    ]:
        model, opt_state, loop_key = carry

        loop_key, batch_key = jr.split(loop_key)
        batch_x, batch_y = sample_batch_uniform(params.X, params.y, params.dummy_batch, batch_key)
        new_loss, grads = vmapped_loss(model, batch_x, batch_y)
        clipped_grads = clip_grads_abadi(grads, params.C)

        # Add spherical noise to gradients
        loop_key, noise_key = jr.split(loop_key)
        noises = get_spherical_noise(clipped_grads, noise, noise_key)
        noised_grads = eqx.apply_updates(clipped_grads, noises)

        # Add noisy gradients, update model and optimizer
        updates, new_opt_state = optimizer.update(
            noised_grads, opt_state, model
        )
        new_model = eqx.apply_updates(model, updates)

        # Subsample each with probability p
        loop_key, _used_key = jr.split(loop_key)
        accuracy = subset_classification_accuracy(
            new_model, params.X, params.y, 0.01, _used_key
        )

        return (new_model, new_opt_state, loop_key), (new_loss, accuracy, new_model, noise_key, batch_key)

    initial_carry = (network, optimizer.init(network), key)
    (network, _, loop_key), (losses, accuracies, networks, noise_keys, batch_keys) = jax.lax.scan(
        training_step,
        initial_carry,
        xs=noise_schedule,
    )

    loop_key, batch_key = jr.split(loop_key)
    batch_x, batch_y = sample_batch_uniform(params.X, params.y, params.dummy_batch, batch_key)
    final_loss, _ = loss(network, batch_x, batch_y)

    losses = jnp.concat([losses, jnp.asarray([final_loss])])

    return network, losses, accuracies
    