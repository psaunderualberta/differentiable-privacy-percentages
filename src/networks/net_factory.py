import jax.numpy as jnp

from conf.singleton_conf import SingletonConfig
from networks.cnn.CNN import CNN
from networks.cnn.config import CNNConfig
from networks.mlp.MLP import MLP
from networks.mlp.config import MLPConfig
from util.dataloaders import get_datasets


def net_factory(
    conf: MLPConfig | CNNConfig,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    key: int = 0,
):
    assert len(input_shape) >= 2, "Input shape must have at least 2 dimensions"
    assert len(output_shape) >= 2, "Output shape must have at least 2 dimensions"

    nclasses = output_shape[1]

    if isinstance(conf, MLPConfig):
        din = int(jnp.prod(jnp.asarray(input_shape[1:])))
        return MLP.from_config(conf, din=din, nclasses=nclasses, key=key)

    if isinstance(conf, CNNConfig):
        nchannels = input_shape[1]
        dummy_data = jnp.zeros(input_shape[1:])
        return CNN.from_config(
            conf, nchannels=nchannels, dummy_data=dummy_data, nclasses=nclasses, key=key
        )

    raise ValueError(f"Network config of type '{type(conf).__name__}' is not recognized!")


def net_factory_from_config():
    X, y, _, _ = get_datasets()
    network_conf = SingletonConfig.get_environment_config_instance().network
    return net_factory(network_conf, input_shape=X.shape, output_shape=y.shape)
