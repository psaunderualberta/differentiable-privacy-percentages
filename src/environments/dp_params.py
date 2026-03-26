import equinox as eqx

from conf.config import EnvConfig
from conf.singleton_conf import SingletonConfig
from networks.net_factory import net_factory_from_config
from networks.util import Network
from util.dataloaders import DatasetLoader, get_dataset_loader


class DP_RL_Params(eqx.Module):
    loader: DatasetLoader  # static non-JAX-array field; equinox treats as aux structure
    optimizer: str = "sgd"
    lr: float = 0.01
    network: Network = Network()
    max_steps_in_episode: int = 500
    scan_segments: int = 1

    @classmethod
    def create(
        cls,
        conf: EnvConfig,
        network_arch: Network,
        loader: DatasetLoader,
    ) -> "DP_RL_Params":
        return DP_RL_Params(
            loader=loader,
            lr=conf.lr.sample(),
            optimizer=conf.optimizer,
            network=network_arch,
            max_steps_in_episode=conf.max_steps_in_episode,
            scan_segments=conf.scan_segments,
        )

    @classmethod
    def create_direct_from_config(cls) -> "DP_RL_Params":
        env_conf = SingletonConfig.get_environment_config_instance()
        private_network_arch = net_factory_from_config()
        loader = get_dataset_loader()
        return cls.create(env_conf, network_arch=private_network_arch, loader=loader)

    def __hash__(self):
        return 0
