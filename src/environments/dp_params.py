import chex
from gymnax.environments import environment
from networks.util import Network
from privacy.privacy import PrivacyAccountant
import equinox as eqx
from conf.config import EnvConfig


class DP_RL_Params(eqx.Module, environment.EnvParams):
    X: chex.Array = None
    y: chex.Array = None
    lr: chex.Array = None
    privacy_accountant: PrivacyAccountant = None
    network: Network = None
    batch_size: chex.Array = None
    var_low: float = -1
    var_high: float = 20
    max_steps_in_episode: int = 30
    reward_fn: eqx.Module = None
    C: float = 1.0
    action: float = 1.0

    @classmethod
    def create(
        cls, conf: EnvConfig, network_arch: Network, X=None, y=None
    ) -> "DP_RL_Params":
        # Set dataset, w/ default values if using default params
        defaults = DP_RL_Params()

        # derived args
        batch_size = conf["batch_size"]
        network = network_arch
        lr = conf["lr"]
        var_low = conf["var_low"]
        var_high = conf["var_high"]
        action = conf.get("action", defaults.action)

        # create privacy accountant
        sample_prob = jnp.array(batch_size.shape[0] / X.shape[0])
        privacy_accountant = PrivacyAccountant(
            conf["moments"], conf["delta"], conf["eps"], sample_prob
        )

        starting_state = privacy_accountant.reset_state()
        lowest_possible_var = privacy_accountant.get_correct_noise(
            starting_state, 1e-10, return_new_state=False
        )
        var_low = lowest_possible_var

        C = conf.get("C", defaults.C)

        return DP_RL_Params(
            X=X,
            y=y,
            lr=lr,
            network=network,
            privacy_accountant=privacy_accountant,
            batch_size=batch_size,
            var_low=var_low,
            var_high=var_high,
            reward_fn=REWARDS["loss-difference"](),
            C=C,
            action=action,
        )

    def __hash__(self):
        return 0
