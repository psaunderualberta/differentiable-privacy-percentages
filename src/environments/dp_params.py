import equinox as eqx

from conf.config import EnvConfig
from conf.singleton_conf import SingletonConfig
from networks.net_factory import net_factory_from_config
from networks.util import Network
from util.dataloaders import DatasetLoader, get_dataset_loader


class DPTrainingParams(eqx.Module):
    loader: DatasetLoader  # static non-JAX-array field; equinox treats as aux structure
    optimizer: str = "sgd"
    lr: float = 0.01
    network: Network = Network()
    num_training_steps: int = 500
    scan_segments: int = 1

    @classmethod
    def create(
        cls,
        conf: EnvConfig,
        network_arch: Network,
        loader: DatasetLoader,
    ) -> "DPTrainingParams":

        assert conf.num_training_steps % conf.scan_segments_derived == 0, (
            "Scan Segments does not cleanly divide training steps"
        )
        return DPTrainingParams(
            loader=loader,
            lr=conf.lr.sample(),
            optimizer=conf.optimizer,
            network=network_arch,
            num_training_steps=conf.num_training_steps,
            scan_segments=conf.scan_segments_derived,
        )

    @classmethod
    def create_direct_from_config(cls) -> "DPTrainingParams":
        env_conf = SingletonConfig.get_environment_config_instance()
        private_network_arch = net_factory_from_config()
        loader = get_dataset_loader()
        return cls.create(env_conf, network_arch=private_network_arch, loader=loader)

    def __hash__(self):
        return 0
