from conf.singleton_conf import SingletonConfig
from networks._registry import build as _network_build
from networks.cnn.CNN import CNN  # noqa: F401 — triggers @register(CNNConfig)
from networks.mlp.MLP import MLP  # noqa: F401 — triggers @register(MLPConfig)
from util.dataloaders import get_datasets


def net_factory(conf, input_shape: tuple[int, ...], output_shape: tuple[int, ...], key: int = 0):
    assert len(input_shape) >= 2, "Input shape must have at least 2 dimensions"
    assert len(output_shape) >= 2, "Output shape must have at least 2 dimensions"
    return _network_build(conf, input_shape, output_shape, key)


def net_factory_from_config():
    X, y, _, _ = get_datasets()
    network_conf = SingletonConfig.get_environment_config_instance().network
    return net_factory(network_conf, input_shape=X.shape, output_shape=y.shape)
