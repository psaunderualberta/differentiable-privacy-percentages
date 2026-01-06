from dataclasses import replace

from jax import numpy as jnp

from conf.singleton_conf import SingletonConfig
from networks.cnn.CNN import CNN
from networks.cnn.config import CNNConfig
from networks.mlp.config import MLPConfig
from networks.mlp.MLP import MLP
from util.dataloaders import get_datasets


def net_factory(
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    conf: MLPConfig | CNNConfig,
):
    assert len(input_shape) >= 2, "Input shape must have at least 2 dimensions"
    assert len(output_shape) >= 2, "Output shape must have at least 2 dimensions"

    if isinstance(conf, MLPConfig):
        din = jnp.prod(jnp.asarray(input_shape[1:])).item()
        conf = replace(conf, din=din, nclasses=output_shape[1])
        return MLP.from_config(conf)
    elif isinstance(conf, CNNConfig):
        conf = replace(
            conf,
            nchannels=input_shape[1],
            dummy_data=jnp.zeros(input_shape[1:]),
            mlp=replace(conf.mlp, nclasses=output_shape[1]),
        )

        return CNN.from_config(conf)

    raise ValueError(f"Network config of type '{type(conf)}' is not recognized!")


def net_factory_from_config():
    X, y, _, _ = get_datasets()
    network_conf = SingletonConfig.get_environment_config_instance().network

    return net_factory(
        input_shape=X.shape,
        output_shape=y.shape,
        conf=network_conf,
    )
