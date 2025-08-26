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
    max_steps_in_episode: int = 500  # Maximum number of steps in an episode

    @classmethod
    def create(
        cls, conf: EnvConfig, network_arch: Network, X: chex.Array, y: chex.Array
    ) -> "DP_RL_Params":
        return DP_RL_Params(
            X=X,
            y=y,
            lr=conf.lr.sample(),
            network=network_arch,
            dummy_batch=jnp.arange(conf.batch_size),
            C=conf.C,
            max_steps_in_episode=conf.max_steps_in_episode,
        )

    def __hash__(self):
        return 0
