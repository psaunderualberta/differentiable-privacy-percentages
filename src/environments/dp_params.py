import chex
from networks.util import Network
import equinox as eqx
from conf.config import EnvConfig
import jax.numpy as jnp


class DP_RL_Params(eqx.Module):
    X: chex.Array = jnp.zeros((1, 1))  # Dataset features
    y: chex.Array = jnp.zeros((1, 1))  # Dataset labels
    lr: float = 0.01  # Learning rate for the optimizer
    network: Network = Network()  # Network architecture for the environment
    dummy_batch: chex.Array = jnp.asarray(1)# Batch size for training
    C: float = 1.0
    action: chex.Array = jnp.asarray(1.0)
    max_steps_in_episode: int = 500  # Maximum number of steps in an episode

    @classmethod
    def create(
        cls, conf: EnvConfig, network_arch: Network, X: chex.Array, y: chex.Array
    ) -> "DP_RL_Params":
        # Set dataset, w/ default values if using default params
        # derived args
        dummy_batch = jnp.arange(conf.batch_size)
        network = network_arch
        lr = conf.lr.min
        action = conf.action

        # create privacy accountant
        assert len(X.shape) >= 2, "X must be 2D"

        return DP_RL_Params(
            X=X,
            y=y,
            lr=lr,
            network=network,
            dummy_batch=dummy_batch,
            C=conf.C,
            action=jnp.asarray(action),
            max_steps_in_episode=conf.max_steps_in_episode,
        )

    def __hash__(self):
        return 0
