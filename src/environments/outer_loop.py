"""Factory for the outer-loop training-loss function.

The returned callable is JIT-compiled and vmapped over ``schedule_batch_size``
independent DP-SGD training runs.  GPU parallelism is handled at the call site
by passing a sharded ``noise_keys`` array (NamedSharding over the device mesh),
letting XLA distribute the vmap across devices automatically.
"""

from functools import partial

import equinox as eqx
from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from environments.dp import DPTrainingParams, train_with_noise
from policy.schedules.abstract import AbstractNoiseAndClipSchedule


def make_training_loss_fn(env_params: DPTrainingParams):
    """Return the JIT-compiled training loss function.

    The returned function has signature::

        get_training_loss(schedule, mb_key, init_key, noise_keys)
            -> ((loss, (losses, accuracies, val_accs)), grads)

    It is decorated with ``eqx.filter_jit`` and ``eqx.filter_value_and_grad``
    so calling it returns both the loss/aux and the schedule gradients.

    GPU parallelism is achieved by passing ``noise_keys`` as a sharded
    ``jax.Array`` (NamedSharding over the device mesh) at the call site;
    XLA distributes the vmap across devices without any explicit shard_map here.

    Parameters
    ----------
    env_params:
        Frozen DP-SGD training parameters (dataset, optimizer, privacy).
    """
    vmapped_train_with_noise = eqx.filter_vmap(
        train_with_noise,
        in_axes=(None, None, None, None, 0),
    )

    @eqx.filter_jit
    @partial(eqx.filter_value_and_grad, has_aux=True)
    def get_training_loss(
        schedule: AbstractNoiseAndClipSchedule,
        mb_key: PRNGKeyArray,
        init_key: PRNGKeyArray,
        noise_keys: PRNGKeyArray,
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        """Calculate the training loss averaged over all vmapped networks."""
        _, to_diff, losses, accuracies, val_acc = vmapped_train_with_noise(
            schedule,
            env_params,
            mb_key,
            init_key,
            noise_keys,
        )
        to_diff = jnp.mean(to_diff).squeeze()
        return to_diff, (losses, accuracies, val_acc)

    return get_training_loss
