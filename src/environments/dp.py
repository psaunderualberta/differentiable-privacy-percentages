"""DP-SGD inner training loop and associated utilities."""

from functools import partial
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jax import checkpoint as jax_checkpoint  # type: ignore[attr-defined]
from jaxtyping import Array, PRNGKeyArray, PyTree

from conf.singleton_conf import SingletonConfig
from environments.dp_params import DPTrainingParams
from environments.losses import loss, vmapped_loss
from policy.schedules.abstract import AbstractNoiseAndClipSchedule
from policy.stateful_schedules.abstract import (
    AbstractScheduleState,
    AbstractStatefulNoiseAndClipSchedule,
)
from util.logger import LoggingSchema
from util.util import (
    classification_accuracy,
    clip_grads_abadi,
    get_spherical_noise,
    reinit_model,
)


def get_private_model_training_schemas() -> list[LoggingSchema]:
    """Return logging schemas for the inner-loop train-loss and accuracy tables."""
    return [
        LoggingSchema(
            table_name="train_losses",
            cols=["losses"],
        ),
        LoggingSchema(
            table_name="accuracies",
            cols=["accuracies"],
        ),
    ]


def training_step(
    model: PyTree,
    optimizer: optax.GradientTransformation,
    opt_state: PyTree,
    noise_key: PRNGKeyArray,
    batch_x: Array,
    batch_y: Array,
    noise: Array,
    clip: Array,
    params: DPTrainingParams,
    private: bool = True,
) -> tuple[
    PyTree,
    optax.OptState,
    PRNGKeyArray,  # Carry values
    Array,
    Array,  # Scan outputs
]:
    """Perform a single DP-SGD (or non-private) training step.

    Receives a pre-fetched mini-batch (batch_x, batch_y) rather than sampling
    from the dataset internally.  Accuracy is computed on the current batch.

    Args:
        model: Current model parameters as an equinox pytree.
        optimizer: Optax gradient transformation.
        opt_state: Current optimizer state.
        noise_key: PRNG key for spherical Gaussian noise.
        batch_x: Pre-fetched batch features of shape (B, *sample_shape).
        batch_y: Pre-fetched batch labels of shape (B, *label_shape).
        noise: Per-step σ value for the Gaussian mechanism.
        clip: Per-step gradient clipping threshold.
        params: DP environment parameters (optimizer config etc.).
        private: If False, skip clipping and noise (non-private SGD).

    Returns:
        Tuple of (new_model, new_opt_state, noise_key, loss, accuracy).
    """
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

    # Apply noisy gradients, update model and optimizer
    updates, new_opt_state = optimizer.update(noised_grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    # Accuracy on the current training batch (avoids extra dataset access per step)
    accuracy = classification_accuracy(new_model, batch_x, batch_y)

    return new_model, new_opt_state, noise_key, new_loss, accuracy


@eqx.filter_jit
def train_with_noise(
    schedule: AbstractNoiseAndClipSchedule,
    params: DPTrainingParams,
    mb_key: PRNGKeyArray,
    init_key: PRNGKeyArray,
    noise_key: PRNGKeyArray,
) -> tuple[eqx.Module, Array, Array, Array, Array]:
    # Get noise and clip schedules
    noise_schedule = schedule.get_private_sigmas()
    clip_schedule = schedule.get_private_clips()

    T = params.num_training_steps
    K = params.scan_segments
    loader = params.loader
    N = loader.n_train
    batch_size = SingletonConfig.get_environment_config_instance().batch_size

    # --- Pre-compute all batch indices before the scan ---
    # Generate T+1 sets: T for scan steps, 1 for the post-scan final-loss batch.
    # Using approx_max_k on a uniform vector matches the existing Poisson sampling
    # semantics; vmap over T+1 keys avoids a (T, N) intermediate array.
    step_keys = jr.split(mb_key, T + 1)

    def _gen_indices(key: PRNGKeyArray) -> Array:
        probs = jr.uniform(key, (N,))
        _, idxs = jax.lax.approx_max_k(probs, batch_size)
        return idxs  # (batch_size,) int

    all_indices = jax.vmap(_gen_indices)(step_keys[:T])  # (T, batch_size)
    final_indices = _gen_indices(step_keys[T])  # (batch_size,)

    # --- Callback specs for pure_callback batch fetching ---
    seg_len = T // K
    # Per-segment specs: load an entire segment's batches in one host call.
    # This reduces host-device sync points from T → K and lets XLA fuse the
    # inner scan without interruption.  Memory cost is (seg_len, B, *shape).
    seg_x_spec = jax.ShapeDtypeStruct((seg_len, batch_size, *loader.sample_shape), jnp.float32)
    seg_y_spec = jax.ShapeDtypeStruct((seg_len, batch_size, *loader.label_shape), jnp.float32)
    # Single-batch spec used only for the post-scan final-loss batch.
    x_spec = jax.ShapeDtypeStruct((batch_size, *loader.sample_shape), jnp.float32)
    y_spec = jax.ShapeDtypeStruct((batch_size, *loader.label_shape), jnp.float32)

    def _fetch_segment(seg_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Load seg_len batches from disk in one shot; seg_indices shape: (seg_len, B)."""
        flat = seg_indices.reshape(-1)
        bx, by = loader.load_train_batch(np.asarray(flat))
        return bx.reshape(seg_len, batch_size, *loader.sample_shape), by.reshape(
            seg_len, batch_size, *loader.label_shape
        )

    def _fetch_batch(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return loader.load_train_batch(np.asarray(indices))

    # --- Create network ---
    network = reinit_model(cast(eqx.Module, params.network), init_key)
    net_params, net_static = eqx.partition(network, eqx.is_array)

    optimizer = getattr(optax, params.optimizer)(params.lr)
    opt_state = optimizer.init(net_params)
    opt_state_params, opt_state_static = eqx.partition(opt_state, eqx.is_array)

    # --- Segmented scan-of-scans ---
    # Reshape (T, ...) → (K, T//K, ...) for the outer scan over K segments.
    # With K=1 this is equivalent to the original single scan.
    noise_segs = noise_schedule.reshape(K, seg_len)
    clip_segs = clip_schedule.reshape(K, seg_len)
    idx_segs = all_indices.reshape(K, seg_len, batch_size)

    @partial(
        jax_checkpoint,
        policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
    )
    def inner_step(
        carry: tuple[PyTree, PyTree, PRNGKeyArray],
        xs: tuple[Array, Array, Array, Array],
    ) -> tuple[tuple[PyTree, PyTree, PRNGKeyArray], tuple[Array, Array]]:
        """Single checkpointed DP-SGD step; batch data arrives as scan input."""
        net_params, opt_state_params, noise_key = carry
        noise_t, clip_t, batch_x, batch_y = xs

        model = eqx.combine(net_params, net_static)
        opt_state = eqx.combine(opt_state_params, opt_state_static)
        new_model, new_opt_state, noise_key, new_loss, accuracy = training_step(
            model,
            optimizer,
            opt_state,
            noise_key,
            batch_x,
            batch_y,
            noise_t,
            clip_t,
            params,
        )
        new_net_params, _ = eqx.partition(new_model, eqx.is_array)
        new_opt_state_params, _ = eqx.partition(new_opt_state, eqx.is_array)
        return (new_net_params, new_opt_state_params, noise_key), (new_loss, accuracy)

    def outer_step(
        carry: tuple[PyTree, PyTree, PRNGKeyArray],
        seg_xs: tuple[Array, Array, Array],
    ) -> tuple[tuple[PyTree, PyTree, PRNGKeyArray], tuple[Array, Array]]:
        """One outer scan step: load segment batches once, then run inner scan."""
        noise_seg, clip_seg, idx_seg = seg_xs
        # One host-device sync per segment instead of one per step.
        seg_bx, seg_by = jax.pure_callback(_fetch_segment, (seg_x_spec, seg_y_spec), idx_seg)
        return jax.lax.scan(inner_step, carry, (noise_seg, clip_seg, seg_bx, seg_by))

    initial_carry = (net_params, opt_state_params, noise_key)
    (net_params, _, noise_key), (losses_seg, accs_seg) = jax.lax.scan(
        outer_step,
        initial_carry,
        (noise_segs, clip_segs, idx_segs),
    )
    # Flatten (K, T//K) → (T,)
    losses = losses_seg.reshape(T)
    accuracies = accs_seg.reshape(T)

    # --- Post-scan final training loss on a fresh batch ---
    network_final = eqx.combine(net_params, net_static)
    final_batch_x, final_batch_y = jax.pure_callback(_fetch_batch, (x_spec, y_spec), final_indices)
    final_loss, _ = loss(network_final, final_batch_x, final_batch_y)
    losses = jnp.concat([losses, jnp.asarray([final_loss])])

    # --- Chunked validation evaluation via pure_callback ---
    n_val_chunks = loader.n_val // loader.val_chunk_size
    val_chunk_idx = jnp.arange(loader.n_val, dtype=jnp.int32).reshape(
        n_val_chunks, loader.val_chunk_size
    )

    xv_spec = jax.ShapeDtypeStruct((loader.val_chunk_size, *loader.sample_shape), jnp.float32)
    yv_spec = jax.ShapeDtypeStruct((loader.val_chunk_size, *loader.label_shape), jnp.float32)

    def _fetch_val_chunk(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return loader.load_val_chunk(np.asarray(indices))

    def val_body(
        carry: tuple[Array, Array],
        chunk_indices: Array,
    ) -> tuple[tuple[Array, Array], None]:
        batch_x, batch_y = jax.pure_callback(_fetch_val_chunk, (xv_spec, yv_spec), chunk_indices)
        batch_loss, _ = loss(network_final, batch_x, batch_y)
        batch_acc = classification_accuracy(network_final, batch_x, batch_y)
        n = jnp.array(loader.val_chunk_size, jnp.float32)
        total_loss, total_acc = carry
        return (total_loss + batch_loss * n, total_acc + batch_acc * n), None

    (total_loss, total_acc), _ = jax.lax.scan(val_body, (0.0, 0.0), val_chunk_idx)
    n_val = jnp.array(loader.n_val, jnp.float32)
    val_loss = total_loss / n_val
    val_accuracy = total_acc / n_val

    return network_final, val_loss, losses, accuracies, val_accuracy


@eqx.filter_jit
def train_with_stateful_noise(
    schedule: AbstractStatefulNoiseAndClipSchedule,
    params: DPTrainingParams,
    mb_key: PRNGKeyArray,
    init_key: PRNGKeyArray,
    noise_key: PRNGKeyArray,
) -> tuple[eqx.Module, Array, Array, Array, Array]:
    initial_schedule_state = schedule.get_initial_state()
    iters = schedule.get_iteration_array()

    T = params.num_training_steps
    K = params.scan_segments
    loader = params.loader
    N = loader.n_train
    batch_size = SingletonConfig.get_environment_config_instance().batch_size

    # --- Pre-compute all batch indices before the scan ---
    step_keys = jr.split(mb_key, T + 1)

    def _gen_indices(key: PRNGKeyArray) -> Array:
        probs = jr.uniform(key, (N,))
        _, idxs = jax.lax.approx_max_k(probs, batch_size)
        return idxs

    all_indices = jax.vmap(_gen_indices)(step_keys[:T])  # (T, batch_size)
    final_indices = _gen_indices(step_keys[T])  # (batch_size,)

    # --- Callback specs ---
    seg_len = T // K
    seg_x_spec = jax.ShapeDtypeStruct((seg_len, batch_size, *loader.sample_shape), jnp.float32)
    seg_y_spec = jax.ShapeDtypeStruct((seg_len, batch_size, *loader.label_shape), jnp.float32)
    x_spec = jax.ShapeDtypeStruct((batch_size, *loader.sample_shape), jnp.float32)
    y_spec = jax.ShapeDtypeStruct((batch_size, *loader.label_shape), jnp.float32)

    def _fetch_segment(seg_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flat = seg_indices.reshape(-1)
        bx, by = loader.load_train_batch(np.asarray(flat))
        return bx.reshape(seg_len, batch_size, *loader.sample_shape), by.reshape(
            seg_len, batch_size, *loader.label_shape
        )

    def _fetch_batch(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return loader.load_train_batch(np.asarray(indices))

    # --- Create network ---
    network = reinit_model(cast(eqx.Module, params.network), init_key)
    net_params, net_static = eqx.partition(network, eqx.is_array)

    optimizer = getattr(optax, params.optimizer)(params.lr)
    opt_state = optimizer.init(net_params)
    opt_state_params, opt_state_static = eqx.partition(opt_state, eqx.is_array)

    # --- Segmented scan-of-scans ---
    # iters is a 1-D array of length T; zip with idx_segs for the inner scan.
    iters_segs = iters.reshape(K, seg_len)
    idx_segs = all_indices.reshape(K, seg_len, batch_size)

    @partial(
        jax_checkpoint,
        policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
    )
    def inner_step(
        carry: tuple[PyTree, PyTree, AbstractScheduleState, PRNGKeyArray, PRNGKeyArray],
        xs: tuple[Array, Array, Array, Array],
    ) -> tuple[
        tuple[PyTree, PyTree, AbstractScheduleState, PRNGKeyArray, PRNGKeyArray],
        tuple[Array, Array],
    ]:
        """Single scan body: update stateful schedule, run DP-SGD step."""
        net_params, opt_state_params, schedule_state, mb_key, noise_key = carry
        iter_t, batch_x, batch_y = xs

        model = eqx.combine(net_params, net_static)
        opt_state = eqx.combine(opt_state_params, opt_state_static)

        new_loss, grads = vmapped_loss(model, batch_x, batch_y)

        # Update stateful schedule
        new_schedule_state = schedule.update_state(
            schedule_state,
            grads,
            iter_t,
            batch_x,
            batch_y,
        )
        clip = new_schedule_state.get_clip()
        noise = new_schedule_state.get_noise()

        # Clip gradients
        clipped_grads = clip_grads_abadi(grads, clip)

        # Add spherical noise to gradients
        noise_key, _key = jr.split(noise_key)
        noises = get_spherical_noise(clipped_grads, noise, clip, _key)
        noised_grads = eqx.apply_updates(clipped_grads, noises)

        # Update model
        updates, new_opt_state = optimizer.update(noised_grads, opt_state, model)
        new_model = eqx.apply_updates(model, updates)

        # Batch accuracy
        accuracy = classification_accuracy(new_model, batch_x, batch_y)

        new_net_params, _ = eqx.partition(new_model, eqx.is_array)
        new_opt_state_params, _ = eqx.partition(new_opt_state, eqx.is_array)
        return (
            new_net_params,
            new_opt_state_params,
            new_schedule_state,
            mb_key,
            noise_key,
        ), (new_loss, accuracy)

    def outer_step(
        carry: tuple[PyTree, PyTree, AbstractScheduleState, PRNGKeyArray, PRNGKeyArray],
        seg_xs: tuple[Array, Array],
    ) -> tuple[
        tuple[PyTree, PyTree, AbstractScheduleState, PRNGKeyArray, PRNGKeyArray],
        tuple[Array, Array],
    ]:
        iters_seg, idx_seg = seg_xs
        seg_bx, seg_by = jax.pure_callback(_fetch_segment, (seg_x_spec, seg_y_spec), idx_seg)
        return jax.lax.scan(inner_step, carry, (iters_seg, seg_bx, seg_by))

    initial_carry = (
        net_params,
        opt_state_params,
        initial_schedule_state,
        mb_key,
        noise_key,
    )
    (net_params, _, _, mb_key, noise_key), (losses_seg, accs_seg) = jax.lax.scan(
        outer_step,
        initial_carry,
        (iters_segs, idx_segs),
    )
    losses = losses_seg.reshape(T)
    accuracies = accs_seg.reshape(T)

    # --- Post-scan final training loss ---
    network_final = eqx.combine(net_params, net_static)
    final_batch_x, final_batch_y = jax.pure_callback(_fetch_batch, (x_spec, y_spec), final_indices)
    final_loss, _ = loss(network_final, final_batch_x, final_batch_y)
    losses = jnp.concat([losses, jnp.asarray([final_loss])])

    # --- Chunked validation evaluation ---
    n_val_chunks = loader.n_val // loader.val_chunk_size
    val_chunk_idx = jnp.arange(loader.n_val, dtype=jnp.int32).reshape(
        n_val_chunks, loader.val_chunk_size
    )

    xv_spec = jax.ShapeDtypeStruct((loader.val_chunk_size, *loader.sample_shape), jnp.float32)
    yv_spec = jax.ShapeDtypeStruct((loader.val_chunk_size, *loader.label_shape), jnp.float32)

    def _fetch_val_chunk(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return loader.load_val_chunk(np.asarray(indices))

    def val_body(
        carry: tuple[Array, Array],
        chunk_indices: Array,
    ) -> tuple[tuple[Array, Array], None]:
        batch_x, batch_y = jax.pure_callback(_fetch_val_chunk, (xv_spec, yv_spec), chunk_indices)
        batch_loss, _ = loss(network_final, batch_x, batch_y)
        batch_acc = classification_accuracy(network_final, batch_x, batch_y)
        n = jnp.array(loader.val_chunk_size, jnp.float32)
        total_loss, total_acc = carry
        return (total_loss + batch_loss * n, total_acc + batch_acc * n), None

    (total_loss, total_acc), _ = jax.lax.scan(val_body, (0.0, 0.0), val_chunk_idx)
    n_val = jnp.array(loader.n_val, jnp.float32)
    val_loss = total_loss / n_val
    val_accuracy = total_acc / n_val

    return network_final, val_loss, losses, accuracies, val_accuracy
