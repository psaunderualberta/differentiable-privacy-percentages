"""JAX Compatible  version of MountainCarContinuous-v0 OpenAI gym environment.


Source:
github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py
"""

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, PRNGKeyArray, PyTree

from environments.dp_params import DP_RL_Params
from environments.losses import loss, vmapped_loss
from policy.schedules.abstract import AbstractNoiseAndClipSchedule
from policy.stateful_schedules.abstract import (
    AbstractScheduleState,
    AbstractStatefulNoiseAndClipSchedule,
)
from util.logger import Loggable, LoggingSchema
from util.util import (
    classification_accuracy,
    clip_grads_abadi,
    get_spherical_noise,
    reinit_model,
    sample_batch_uniform,
    subset_classification_accuracy,
)


def get_private_model_training_schemas() -> list[LoggingSchema]:
    return [
        LoggingSchema(
            table_name="train_loss",
            cols=["losses"],
        ),
        LoggingSchema(
            table_name="accuracy",
            cols=["accuracies"],
        ),
    ]


def training_step(
    model: PyTree,
    optimizer: optax.GradientTransformation,
    opt_state: PyTree,
    mb_key: PRNGKeyArray,
    noise_key: PRNGKeyArray,
    noise: Array,
    clip: Array,
    params: DP_RL_Params,
    private: bool = True,
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

    if private:
        new_loss, grads = vmapped_loss(model, batch_x, batch_y)
        clipped_grads = clip_grads_abadi(grads, clip)

        # Add spherical noise to gradients
        noise_key, _key = jr.split(noise_key)
        noises = get_spherical_noise(clipped_grads, noise, clip, _key)
        noised_grads = eqx.apply_updates(clipped_grads, noises)
    else:
        new_loss, grads = loss(model, batch_x, batch_y)
        noised_grads = grads

    # Add noisy gradients, update model and optimizer
    updates, new_opt_state = optimizer.update(noised_grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    # Subsample each with probability p
    mb_key, _key = jr.split(mb_key)
    accuracy = subset_classification_accuracy(new_model, params.X, params.y, 0.01, _key)

    return new_model, new_opt_state, mb_key, noise_key, new_loss, accuracy


@eqx.filter_jit
def train_with_noise(
    schedule: AbstractNoiseAndClipSchedule,
    params: DP_RL_Params,
    mb_key: PRNGKeyArray,
    init_key: PRNGKeyArray,
    noise_key: PRNGKeyArray,
) -> tuple[eqx.Module, Array, Array, Array, Array]:
    # Get noise and clip schedules
    noise_schedule = schedule.get_private_sigmas()
    clip_schedule = schedule.get_private_clips()

    # Create network
    network = reinit_model(params.network, init_key)
    net_params, net_static = eqx.partition(network, eqx.is_array)
    net_params = eqx.filter_jit(jax.lax.pvary)(net_params, "x")

    optimizer = getattr(optax, params.optimizer)(params.lr)
    opt_state = optimizer.init(net_params)
    opt_state_params, opt_state_static = eqx.partition(opt_state, eqx.is_array)
    opt_state_params = eqx.filter_jit(jax.lax.pvary)(opt_state_params, "x")

    @partial(
        jax.checkpoint, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    )
    def scanned_training_step(
        carry: tuple[PyTree, PyTree, PRNGKeyArray, PRNGKeyArray],
        noise_and_clip: tuple[Array, Array],
    ) -> tuple[
        tuple[PyTree, optax.OptState, PRNGKeyArray, PRNGKeyArray],  # Carry values
        tuple[Array, Array],  # Scan outputs
    ]:
        net_params, opt_state_params, mb_key, noise_key = carry
        model = eqx.combine(net_params, net_static)
        opt_state = eqx.combine(opt_state_params, opt_state_static)

        new_model, new_opt_state, mb_key, noise_key, new_loss, accuracy = training_step(
            model, optimizer, opt_state, mb_key, noise_key, *noise_and_clip, params
        )

        new_net_params, _ = eqx.partition(new_model, eqx.is_array)
        new_opt_state_params, _ = eqx.partition(new_opt_state, eqx.is_array)
        return (new_net_params, new_opt_state_params, mb_key, noise_key), (
            new_loss,
            accuracy,
        )

    initial_carry = (net_params, opt_state_params, mb_key, noise_key)
    (net_params, _, mb_key, noise_key), (losses, accuracies) = jax.lax.scan(
        scanned_training_step,
        initial_carry,
        xs=(noise_schedule, clip_schedule),
    )

    mb_key, batch_key = jr.split(mb_key)
    network_final = eqx.combine(net_params, net_static)
    batch_x, batch_y = sample_batch_uniform(
        params.X, params.y, params.dummy_batch, batch_key
    )
    final_loss, _ = loss(network_final, batch_x, batch_y)
    losses = jnp.concat([losses, jnp.asarray([final_loss])])

    val_loss, _ = loss(network_final, params.valX, params.valy)
    val_accuracy = classification_accuracy(network_final, params.valX, params.valy)

    return network_final, val_loss, losses, accuracies, val_accuracy


@eqx.filter_jit
def train_with_stateful_noise(
    schedule: AbstractStatefulNoiseAndClipSchedule,
    params: DP_RL_Params,
    mb_key: PRNGKeyArray,
    init_key: PRNGKeyArray,
    noise_key: PRNGKeyArray,
) -> tuple[eqx.Module, Array, Array, Array, Array]:
    # Get noise and clip schedules
    initial_schedule_state = schedule.get_initial_state()
    iters = schedule.get_iteration_array()

    # Create network
    network = reinit_model(params.network, init_key)
    net_params, net_static = eqx.partition(network, eqx.is_array)
    net_params = eqx.filter_jit(jax.lax.pvary)(net_params, "x")

    optimizer = getattr(optax, params.optimizer)(params.lr)
    opt_state = optimizer.init(net_params)
    opt_state_params, opt_state_static = eqx.partition(opt_state, eqx.is_array)
    opt_state_params = eqx.filter_jit(jax.lax.pvary)(opt_state_params, "x")

    @partial(
        jax.checkpoint, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable
    )
    def scanned_training_step(
        carry: tuple[PyTree, PyTree, AbstractScheduleState, PRNGKeyArray, PRNGKeyArray],
        iter: Array,
    ) -> tuple[
        tuple[
            PyTree, optax.OptState, AbstractScheduleState, PRNGKeyArray, PRNGKeyArray
        ],  # Carry values
        tuple[Array, Array],  # Scan outputs
    ]:
        net_params, opt_state_params, schedule_state, mb_key, noise_key = carry
        model = eqx.combine(net_params, net_static)
        opt_state = eqx.combine(opt_state_params, opt_state_static)

        mb_key, _key = jr.split(mb_key)
        batch_x, batch_y = sample_batch_uniform(
            params.X, params.y, params.dummy_batch, _key
        )
        new_loss, grads = vmapped_loss(model, batch_x, batch_y)

        # update state
        new_schedule_state = schedule.update_state(
            schedule_state, grads, iter, batch_x, batch_y
        )
        clip = new_schedule_state.get_clip()
        noise = new_schedule_state.get_noise()

        # Clip gradients
        clipped_grads = clip_grads_abadi(grads, clip)

        # Add spherical noise to gradients
        noise_key, _key = jr.split(noise_key)
        noises = get_spherical_noise(clipped_grads, noise, clip, _key)
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
        new_opt_state_params, _ = eqx.partition(new_opt_state, eqx.is_array)
        return (
            new_net_params,
            new_opt_state_params,
            new_schedule_state,
            mb_key,
            noise_key,
        ), (
            new_loss,
            accuracy,
        )

    initial_carry = (
        net_params,
        opt_state_params,
        initial_schedule_state,
        mb_key,
        noise_key,
    )
    (net_params, _, _, mb_key, noise_key), (losses, accuracies) = jax.lax.scan(
        scanned_training_step,
        initial_carry,
        xs=iters,
    )

    mb_key, batch_key = jr.split(mb_key)
    network_final = eqx.combine(net_params, net_static)
    batch_x, batch_y = sample_batch_uniform(
        params.X, params.y, params.dummy_batch, batch_key
    )
    final_loss, _ = loss(network_final, batch_x, batch_y)
    losses = jnp.concat([losses, jnp.asarray([final_loss])])

    val_loss, _ = loss(network_final, params.valX, params.valy)
    val_accuracy = classification_accuracy(network_final, params.valX, params.valy)

    return network_final, val_loss, losses, accuracies, val_accuracy
