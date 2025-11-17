from networks.util import Network
from jaxtyping import Array
import equinox as eqx
from conf.config import EnvConfig
import jax.numpy as jnp


class DP_RL_Params(eqx.Module):
    X: Array  # Dataset features
    y: Array  # Dataset labels
    valX: Array  # Validation dataset features
    valy: Array  # Validation dataset labels
    dummy_batch: Array  # Batch size for training
    optimizer: str = "sgd"
    lr: float = 0.01  # Learning rate for the optimizer
    network: Network = Network()  # Network architecture for the environment
    C: float = 1.0
    max_steps_in_episode: int = 500  # Maximum number of steps in an episode

    @classmethod
    def create(
        cls, conf: EnvConfig, network_arch: Network, X: Array, y: Array, valX: Array, valy: Array
    ) -> "DP_RL_Params":
        return DP_RL_Params(
            X=X,
            y=y,
            valX=valX,
            valy=valy,
            lr=conf.lr.sample(),
            optimizer=conf.optimizer,
            network=network_arch,
            dummy_batch=jnp.arange(conf.batch_size),
            C=conf.C,
            max_steps_in_episode=conf.max_steps_in_episode,
        )

    def __hash__(self):
        return 0
