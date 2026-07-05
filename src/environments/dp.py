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
from environments.losses import loss, per_example_loss_and_grads
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
    poisson_buffer_indices,
    reinit_model,
    sum_clipped_per_example_grads,
)


class TrainingStatistics(eqx.Module):
    val_loss: Array
    val_accuracy: Array
    test_loss: Array
    test_accuracy: Array
    losses: Array
    accuracies: Array


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
    valid: Array | None = None,
    private: bool = True,
) -> tuple[
    PyTree,
    optax.OptState,
    PRNGKeyArray,  # Carry values
    Array,
    Array,  # Scan outputs
]:
    """Perform a single DP-SGD (or non-private) training step.

    Receives a pre-fetched truncated-Poisson buffer (batch_x, batch_y) of ``B``
    slots plus a ``valid`` mask (ADR 0009) marking which slots hold genuinely
    Poisson-included records; invalid slots are masked out of the summed gradient
    and the reported loss. Accuracy is computed on the buffer.

    The physical buffer count is ``B = params.buffer_size``; the gradient-mean and
    noise divisor stay the expected batch ``L = batch_size`` (never ``B`` nor the
    realized count ``m``), which is what keeps the Gaussian sensitivity and the
    entire privacy calibration unchanged.

    Args:
        model: Current model parameters as an equinox pytree.
        optimizer: Optax gradient transformation.
        opt_state: Current optimizer state.
        noise_key: PRNG key for spherical Gaussian noise.
        batch_x: Pre-fetched buffer features of shape (B, *sample_shape).
        batch_y: Pre-fetched buffer labels of shape (B, *label_shape).
        noise: Per-step σ value for the Gaussian mechanism.
        clip: Per-step gradient clipping threshold.
        params: DP environment parameters (optimizer config, buffer_size etc.).
        valid: Bool mask of shape (B,); True for genuinely-included buffer slots.
        private: If False, skip clipping and noise (non-private SGD).

    Returns:
        Tuple of (new_model, new_opt_state, noise_key, loss, accuracy).
    """
    if private:
        env_conf = SingletonConfig.get_environment_config_instance()
        batch_size = env_conf.batch_size  # L = pN: gradient-mean / noise divisor
        B = params.buffer_size  # physical buffer count
        raw_micro = env_conf.microbatch_size
        micro = B if (raw_micro <= 0 or raw_micro >= B) else raw_micro

        # m = number of genuinely-included records; used only for the (cosmetic)
        # train-loss report, guarded against the degenerate empty-buffer draw.
        m = jnp.maximum(valid.sum(), 1.0)

        if micro >= B:
            # Whole-buffer per-sample gradients (no microbatching).
            per_losses, grads = per_example_loss_and_grads(model, batch_x, batch_y)
            summed = sum_clipped_per_example_grads(grads, clip, valid)
            loss_sum = jnp.sum(per_losses * valid)
        else:
            # Accumulate the sum of clipped per-sample gradients one microbatch
            # at a time. The scan body is checkpointed so the backward pass
            # rematerialises (and frees) one microbatch's per-sample gradients at
            # a time, capping the live working set at micro x params.
            num_micro = B // micro
            mb_x = batch_x.reshape(num_micro, micro, *batch_x.shape[1:])
            mb_y = batch_y.reshape(num_micro, micro, *batch_y.shape[1:])
            mb_valid = valid.reshape(num_micro, micro)

            def _micro_sum(bx: Array, by: Array, bv: Array) -> tuple[Array, PyTree]:
                # Per-example losses so masked (invalid) rows drop out of both the
                # loss sum and the clipped-gradient sum.
                per_losses, per_sample_grads = per_example_loss_and_grads(model, bx, by)
                return (
                    jnp.sum(per_losses * bv),
                    sum_clipped_per_example_grads(per_sample_grads, clip, bv),
                )

            init = jax.tree.map(
                lambda s: jnp.zeros(s.shape, s.dtype),
                jax.eval_shape(_micro_sum, mb_x[0], mb_y[0], mb_valid[0]),
            )

            @partial(
                jax_checkpoint,
                policy=jax.checkpoint_policies.nothing_saveable,
            )
            def _accumulate(
                carry: tuple[Array, PyTree],
                xs: tuple[Array, Array, Array],
            ) -> tuple[tuple[Array, PyTree], None]:
                loss_acc, sum_acc = carry
                bx, by, bv = xs
                mb_loss, mb_sum = _micro_sum(bx, by, bv)
                sum_acc = jax.tree.map(lambda a, b: a + b, sum_acc, mb_sum)
                return (loss_acc + mb_loss, sum_acc), None

            (loss_sum, summed), _ = jax.lax.scan(_accumulate, init, (mb_x, mb_y, mb_valid))

        # Mean clipped gradient + a single spherical-noise draw. Both divide by the
        # public expected-batch constant L (batch_size) — never B nor m — so the
        # sensitivity C and noise scale, hence the whole privacy calibration, hold.
        mean_clipped = jax.tree.map(lambda g: g / batch_size, summed)
        new_loss = loss_sum / m

        noise_key, _key = jr.split(noise_key)
        noises = get_spherical_noise(mean_clipped, noise, clip, _key)
        noised_grads = eqx.apply_updates(mean_clipped, noises)
    else:
        new_loss, grads = loss(model, batch_x, batch_y)
        noised_grads = grads

    # Apply noisy gradients, update model and optimizer
    updates, new_opt_state = optimizer.update(noised_grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    # Accuracy on the current training batch (avoids extra dataset access per step)
    accuracy = classification_accuracy(new_model, batch_x, batch_y)

    return new_model, new_opt_state, noise_key, new_loss, accuracy


def get_validation_statistics(network, loader) -> tuple[Array, Array]:
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
        batch_loss, _ = loss(network, batch_x, batch_y)
        batch_acc = classification_accuracy(network, batch_x, batch_y)
        n = jnp.array(loader.val_chunk_size, jnp.float32)
        total_loss, total_acc = carry
        return (total_loss + batch_loss * n, total_acc + batch_acc * n), None

    (total_loss, total_acc), _ = jax.lax.scan(val_body, (0.0, 0.0), val_chunk_idx)
    n_val = jnp.array(loader.n_val, jnp.float32)
    val_loss = total_loss / n_val
    val_accuracy = total_acc / n_val

    return val_loss, val_accuracy


def get_test_statistics(network, loader) -> tuple[Array, Array]:
    # --- Chunked validation evaluation via pure_callback ---
    n_test_chunks = loader.n_test // loader.val_chunk_size
    test_chunk_idx = jnp.arange(loader.n_test, dtype=jnp.int32).reshape(
        n_test_chunks, loader.val_chunk_size
    )

    xv_spec = jax.ShapeDtypeStruct((loader.val_chunk_size, *loader.sample_shape), jnp.float32)
    yv_spec = jax.ShapeDtypeStruct((loader.val_chunk_size, *loader.label_shape), jnp.float32)

    def _fetch_test_chunk(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return loader.load_test_chunk(np.asarray(indices))

    def test_body(
        carry: tuple[Array, Array],
        chunk_indices: Array,
    ) -> tuple[tuple[Array, Array], None]:
        batch_x, batch_y = jax.pure_callback(_fetch_test_chunk, (xv_spec, yv_spec), chunk_indices)
        batch_loss, _ = loss(network, batch_x, batch_y)
        batch_acc = classification_accuracy(network, batch_x, batch_y)
        n = jnp.array(loader.val_chunk_size, jnp.float32)
        total_loss, total_acc = carry
        return (total_loss + batch_loss * n, total_acc + batch_acc * n), None

    (total_loss, total_acc), _ = jax.lax.scan(test_body, (0.0, 0.0), test_chunk_idx)
    n_test = jnp.array(loader.n_test, jnp.float32)
    test_loss = total_loss / n_test
    test_accuracy = total_acc / n_test

    return test_loss, test_accuracy


@eqx.filter_jit
def train_with_noise(
    schedule: AbstractNoiseAndClipSchedule,
    params: DPTrainingParams,
    mb_key: PRNGKeyArray,
    init_key: PRNGKeyArray,
    noise_key: PRNGKeyArray,
) -> tuple[eqx.Module, TrainingStatistics]:
    # Get noise and clip schedules
    noise_schedule = schedule.get_private_noise_scales()
    clip_schedule = schedule.get_private_clips()

    T = params.num_training_steps
    K = params.scan_segments
    loader = params.loader
    N = loader.n_train
    batch_size = SingletonConfig.get_environment_config_instance().batch_size
    B = params.buffer_size  # truncated-Poisson buffer size (ADR 0009); B > L
    p = batch_size / N  # Poisson inclusion probability

    # --- Pre-compute all buffer indices + valid masks before the scan ---
    # Generate T+1 buffers: T for scan steps, 1 for the post-scan final-loss batch.
    # True Poisson subsampling via a fixed-size buffer of B slots (exact top_k on the
    # Bernoulli inclusion mask); `valid` marks which slots hold included records.
    step_keys = jr.split(mb_key, T + 1)

    def _gen_indices(key: PRNGKeyArray) -> tuple[Array, Array]:
        return poisson_buffer_indices(key, N, p, B)  # (B,) int, (B,) bool

    all_indices, all_valid = jax.vmap(_gen_indices)(step_keys[:T])  # (T, B), (T, B)
    final_indices, final_valid = _gen_indices(step_keys[T])  # (B,), (B,)

    # --- Callback specs for pure_callback batch fetching ---
    seg_len = T // K
    # Per-segment specs: load an entire segment's buffers in one host call.
    # This reduces host-device sync points from T → K and lets XLA fuse the
    # inner scan without interruption.  Memory cost is (seg_len, B, *shape).
    seg_x_spec = jax.ShapeDtypeStruct((seg_len, B, *loader.sample_shape), jnp.float32)
    seg_y_spec = jax.ShapeDtypeStruct((seg_len, B, *loader.label_shape), jnp.float32)
    # Single-buffer spec used only for the post-scan final-loss batch.
    x_spec = jax.ShapeDtypeStruct((B, *loader.sample_shape), jnp.float32)
    y_spec = jax.ShapeDtypeStruct((B, *loader.label_shape), jnp.float32)

    def _fetch_segment(seg_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Load seg_len buffers from disk in one shot; seg_indices shape: (seg_len, B)."""
        flat = seg_indices.reshape(-1)
        bx, by = loader.load_train_batch(np.asarray(flat))
        return bx.reshape(seg_len, B, *loader.sample_shape), by.reshape(
            seg_len, B, *loader.label_shape
        )

    def _fetch_batch(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return loader.load_train_batch(np.asarray(indices))

    # --- Create network ---
    network = reinit_model(cast(eqx.Module, params.network), init_key)
    net_params, net_static = eqx.partition(network, eqx.is_array)

    optimizer = params.optimizer
    opt_state = optimizer.init(net_params)
    opt_state_params, opt_state_static = eqx.partition(opt_state, eqx.is_array)

    # --- Segmented scan-of-scans ---
    # Reshape (T, ...) → (K, T//K, ...) for the outer scan over K segments.
    # With K=1 this is equivalent to the original single scan.
    noise_segs = noise_schedule.reshape(K, seg_len)
    clip_segs = clip_schedule.reshape(K, seg_len)
    idx_segs = all_indices.reshape(K, seg_len, B)
    valid_segs = all_valid.reshape(K, seg_len, B)

    @partial(
        jax_checkpoint,
        policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
    )
    def inner_step(
        carry: tuple[PyTree, PyTree, PRNGKeyArray],
        xs: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[tuple[PyTree, PyTree, PRNGKeyArray], tuple[Array, Array]]:
        """Single DP-SGD step; buffer data + valid mask arrive as scan input."""
        net_params, opt_state_params, noise_key = carry
        noise_t, clip_t, batch_x, batch_y, valid_t = xs

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
            valid_t,
        )
        new_net_params, _ = eqx.partition(new_model, eqx.is_array)
        new_opt_state_params, _ = eqx.partition(new_opt_state, eqx.is_array)
        return (new_net_params, new_opt_state_params, noise_key), (new_loss, accuracy)

    def outer_step(
        carry: tuple[PyTree, PyTree, PRNGKeyArray],
        seg_xs: tuple[Array, Array, Array, Array],
    ) -> tuple[tuple[PyTree, PyTree, PRNGKeyArray], tuple[Array, Array]]:
        """One outer scan step: load segment buffers once, then run inner scan."""
        noise_seg, clip_seg, idx_seg, valid_seg = seg_xs
        # One host-device sync per segment instead of one per step.
        seg_bx, seg_by = jax.pure_callback(_fetch_segment, (seg_x_spec, seg_y_spec), idx_seg)
        return jax.lax.scan(inner_step, carry, (noise_seg, clip_seg, seg_bx, seg_by, valid_seg))

    initial_carry = (net_params, opt_state_params, noise_key)
    (net_params, _, noise_key), (losses_seg, accs_seg) = jax.lax.scan(
        outer_step,
        initial_carry,
        (noise_segs, clip_segs, idx_segs, valid_segs),
    )
    # Flatten (K, T//K) → (T,)
    losses = losses_seg.reshape(T)
    accuracies = accs_seg.reshape(T)

    # --- Post-scan final training loss on a fresh buffer (masked to valid rows) ---
    network_final = eqx.combine(net_params, net_static)
    final_batch_x, final_batch_y = jax.pure_callback(_fetch_batch, (x_spec, y_spec), final_indices)
    final_per_losses, _ = per_example_loss_and_grads(network_final, final_batch_x, final_batch_y)
    final_loss = jnp.sum(final_per_losses * final_valid) / jnp.maximum(final_valid.sum(), 1.0)
    losses = jnp.concat([losses, jnp.asarray([final_loss])])

    val_loss, val_accuracy = get_validation_statistics(network_final, loader)
    test_loss, test_accuracy = get_test_statistics(network_final, loader)

    return network_final, TrainingStatistics(
        val_loss=val_loss,
        val_accuracy=val_accuracy,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        losses=losses,
        accuracies=accuracies,
    )


@eqx.filter_jit
def train_with_stateful_noise(
    schedule: AbstractStatefulNoiseAndClipSchedule,
    params: DPTrainingParams,
    mb_key: PRNGKeyArray,
    init_key: PRNGKeyArray,
    noise_key: PRNGKeyArray,
) -> tuple[eqx.Module, TrainingStatistics]:
    initial_schedule_state = schedule.get_initial_state()
    iters = schedule.get_iteration_array()

    T = params.num_training_steps
    K = params.scan_segments
    loader = params.loader
    N = loader.n_train
    batch_size = SingletonConfig.get_environment_config_instance().batch_size
    B = params.buffer_size  # truncated-Poisson buffer size (ADR 0009); B > L
    p = batch_size / N  # Poisson inclusion probability

    # --- Pre-compute all buffer indices + valid masks before the scan ---
    step_keys = jr.split(mb_key, T + 1)

    def _gen_indices(key: PRNGKeyArray) -> tuple[Array, Array]:
        return poisson_buffer_indices(key, N, p, B)

    all_indices, all_valid = jax.vmap(_gen_indices)(step_keys[:T])  # (T, B), (T, B)
    final_indices, final_valid = _gen_indices(step_keys[T])  # (B,), (B,)

    # --- Callback specs ---
    seg_len = T // K
    seg_x_spec = jax.ShapeDtypeStruct((seg_len, B, *loader.sample_shape), jnp.float32)
    seg_y_spec = jax.ShapeDtypeStruct((seg_len, B, *loader.label_shape), jnp.float32)
    x_spec = jax.ShapeDtypeStruct((B, *loader.sample_shape), jnp.float32)
    y_spec = jax.ShapeDtypeStruct((B, *loader.label_shape), jnp.float32)

    def _fetch_segment(seg_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        flat = seg_indices.reshape(-1)
        bx, by = loader.load_train_batch(np.asarray(flat))
        return bx.reshape(seg_len, B, *loader.sample_shape), by.reshape(
            seg_len, B, *loader.label_shape
        )

    def _fetch_batch(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return loader.load_train_batch(np.asarray(indices))

    # --- Create network ---
    network = reinit_model(cast(eqx.Module, params.network), init_key)
    net_params, net_static = eqx.partition(network, eqx.is_array)

    optimizer = params.optimizer
    opt_state = optimizer.init(net_params)
    opt_state_params, opt_state_static = eqx.partition(opt_state, eqx.is_array)

    # --- Segmented scan-of-scans ---
    # iters is a 1-D array of length T; zip with idx_segs for the inner scan.
    iters_segs = iters.reshape(K, seg_len)
    idx_segs = all_indices.reshape(K, seg_len, B)
    valid_segs = all_valid.reshape(K, seg_len, B)

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
        iter_t, batch_x, batch_y, valid_t = xs

        model = eqx.combine(net_params, net_static)
        opt_state = eqx.combine(opt_state_params, opt_state_static)

        per_losses, grads = per_example_loss_and_grads(model, batch_x, batch_y)
        # Reported loss over the m genuinely-included buffer rows only.
        new_loss = jnp.sum(per_losses * valid_t) / jnp.maximum(valid_t.sum(), 1.0)

        # Update stateful schedule — the within-clip fraction (median statistic)
        # is taken over the valid rows only.
        new_schedule_state = schedule.update_state(
            schedule_state,
            grads,
            iter_t,
            batch_x,
            batch_y,
            valid_t,
        )
        clip = new_schedule_state.get_clip()
        noise = new_schedule_state.get_noise()

        # Clip gradients (masking invalid buffer rows; divisor stays L)
        clipped_grads = clip_grads_abadi(grads, clip, valid_t)

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
        seg_xs: tuple[Array, Array, Array],
    ) -> tuple[
        tuple[PyTree, PyTree, AbstractScheduleState, PRNGKeyArray, PRNGKeyArray],
        tuple[Array, Array],
    ]:
        iters_seg, idx_seg, valid_seg = seg_xs
        seg_bx, seg_by = jax.pure_callback(_fetch_segment, (seg_x_spec, seg_y_spec), idx_seg)
        return jax.lax.scan(inner_step, carry, (iters_seg, seg_bx, seg_by, valid_seg))

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
        (iters_segs, idx_segs, valid_segs),
    )
    losses = losses_seg.reshape(T)
    accuracies = accs_seg.reshape(T)

    # --- Post-scan final training loss (masked to valid rows) ---
    network_final = eqx.combine(net_params, net_static)
    final_batch_x, final_batch_y = jax.pure_callback(_fetch_batch, (x_spec, y_spec), final_indices)
    final_per_losses, _ = per_example_loss_and_grads(network_final, final_batch_x, final_batch_y)
    final_loss = jnp.sum(final_per_losses * final_valid) / jnp.maximum(final_valid.sum(), 1.0)
    losses = jnp.concat([losses, jnp.asarray([final_loss])])

    val_loss, val_accuracy = get_validation_statistics(network_final, loader)
    test_loss, test_accuracy = get_test_statistics(network_final, loader)

    return network_final, TrainingStatistics(
        val_loss=val_loss,
        val_accuracy=val_accuracy,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        losses=losses,
        accuracies=accuracies,
    )
