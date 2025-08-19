import chex
from networks.util import Network
from privacy.privacy import PrivacyAccountant
import equinox as eqx
from conf.config import EnvConfig
import jax.numpy as jnp


class DP_RL_Params(eqx.Module):
    X: chex.Array = jnp.zeros((1, 1))  # Dataset features
    y: chex.Array = jnp.zeros((1, 1))  # Dataset labels
    lr: float = 0.01  # Learning rate for the optimizer
    network: Network = Network()  # Network architecture for the environment
    dummy_batch: chex.Array = jnp.asarray(1)# Batch size for training
    privacy_accountant: PrivacyAccountant = PrivacyAccountant(jnp.asarray([5]), 1e-5, 0.5, 0.01)  # Privacy accountant for the environment
    var_low: float = -1
    var_high: float = 20
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
        var_low = conf.var_low
        var_high = conf.var_high
        action = conf.action

        # create privacy accountant
        assert len(X.shape) >= 2, "X must be 2D"
        sample_prob = jnp.asarray(conf.batch_size / X.shape[0])
        moments = jnp.asarray(conf.moments)
        privacy_accountant = PrivacyAccountant(
            moments, conf.delta, conf.eps, sample_prob
        )

        starting_state = privacy_accountant.reset_state()
        lowest_possible_var = privacy_accountant.get_correct_noise(
            starting_state, 1e-10, return_new_state=False
        )
        var_low = lowest_possible_var

        return DP_RL_Params(
            X=X,
            y=y,
            lr=lr,
            network=network,
            dummy_batch=dummy_batch,
            privacy_accountant=privacy_accountant,
            var_low=var_low,
            var_high=var_high,
            C=conf.C,
            action=jnp.asarray(action),
            max_steps_in_episode=conf.max_steps_in_episode,
        )

    def __hash__(self):
        return 0
