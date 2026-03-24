from conf.singleton_conf import SingletonConfig
from networks._registry import build as _network_build
from networks.auto.config import AutoNetworkConfig
from networks.cnn.CNN import CNN  # noqa: F401 — triggers @register(CNNConfig)
from networks.cnn.config import CNNConfig
from networks.mlp.config import MLPConfig
from networks.mlp.MLP import MLP  # noqa: F401 — triggers @register(MLPConfig)
from util.dataloaders import get_datasets

DATASET_NETWORK_DEFAULTS = {
    "mnist": CNNConfig(
        channels=(16, 32),
        kernel_sizes=(8, 4),
        paddings=(2, 0),
        strides=(2, 2),
        pool_kernel_size=2,
        mlp=MLPConfig(hidden_sizes=(32,)),
    ),
    "fashion-mnist": CNNConfig(
        channels=(16, 32),
        kernel_sizes=(8, 4),
        paddings=(2, 0),
        strides=(2, 2),
        pool_kernel_size=2,
        mlp=MLPConfig(hidden_sizes=(32,)),
    ),
    "cifar-10": CNNConfig(
        channels=(32, 64),
        kernel_sizes=(3, 3),
        paddings=(1, 1),
        strides=(1, 1),
        pool_kernel_size=2,
        mlp=MLPConfig(hidden_sizes=(256,)),
    ),
    "california": MLPConfig(hidden_sizes=(64, 32)),
}


def resolve_network_config(conf, dataset: str):
    """Resolve AutoNetworkConfig to a dataset-specific config; pass others through."""
    if isinstance(conf, AutoNetworkConfig):
        return DATASET_NETWORK_DEFAULTS[dataset]
    return conf


def net_factory(conf, input_shape: tuple[int, ...], output_shape: tuple[int, ...], key: int = 0):
    """Build a network from a config object and dataset shapes.

    Args:
        conf: Network config (e.g. MLPConfig or CNNConfig).
        input_shape: Full input batch shape (N, *features).
        output_shape: Full output batch shape (N, nclasses).
        key: Integer seed for weight initialization.
    """
    assert len(input_shape) >= 2, "Input shape must have at least 2 dimensions"
    assert len(output_shape) >= 2, "Output shape must have at least 2 dimensions"
    return _network_build(conf, input_shape, output_shape, key)


def net_factory_from_config():
    """Build a network from the global singleton config and the configured dataset shapes."""
    X, y, _, _ = get_datasets()
    dataset = SingletonConfig.get_sweep_config_instance().dataset
    network_conf = SingletonConfig.get_environment_config_instance().network
    network_conf = resolve_network_config(network_conf, dataset)
    return net_factory(network_conf, input_shape=X.shape, output_shape=y.shape)
