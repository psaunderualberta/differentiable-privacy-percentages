"""Factory for the outer-loop training-loss function.

The returned callable is JIT-compiled and shard-mapped over the available
GPU mesh so that ``schedule_batch_size`` networks are trained in parallel.
Wrapping construction in a factory keeps ``main.py`` free of the decorator
stack and JAX sharding details.
"""

from functools import partial

import equinox as eqx
from jax import lax as jlax
from jax import numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PRNGKeyArray

from environments.dp import DPTrainingParams, train_with_noise
from policy.schedules.abstract import AbstractNoiseAndClipSchedule


def make_training_loss_fn(mesh: Mesh, env_params: DPTrainingParams):
    """Return the JIT-compiled, shard-mapped training loss function.

    The returned function has signature::

        get_training_loss(schedule, mb_key, init_key, noise_keys)
            -> ((loss, (losses, accuracies, val_accs)), grads)

    It is decorated with ``eqx.filter_jit`` and ``eqx.filter_value_and_grad``
    so calling it returns both the loss/aux and the schedule gradients.

    Parameters
    ----------
    mesh:
        JAX device mesh produced by ``get_optimal_mesh``.
    env_params:
        Frozen DP-SGD training parameters (dataset, optimizer, privacy).
    """
    vmapped_train_with_noise = eqx.filter_vmap(
        train_with_noise,
        in_axes=(None, None, None, None, 0),
    )

    @eqx.filter_jit
    @partial(eqx.filter_value_and_grad, has_aux=True)
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P(), P(), P("x")),
        out_specs=(P(), (P("x"), P("x"), P("x"))),
        check_rep=False,
    )
    def get_training_loss(
        schedule: AbstractNoiseAndClipSchedule,
        mb_key: PRNGKeyArray,
        init_key: PRNGKeyArray,
        noise_keys: PRNGKeyArray,
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        """Calculate the training loss averaged over all shard-mapped networks."""
        _, to_diff, losses, accuracies, val_acc = vmapped_train_with_noise(
            schedule,
            env_params,
            mb_key,
            init_key,
            noise_keys,
        )
        to_diff = jnp.mean(to_diff)
        to_diff = jlax.pmean(to_diff, "x").squeeze()
        return to_diff, (losses, accuracies, val_acc)

    return get_training_loss
