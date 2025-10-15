from dataclasses import replace
from typing import Tuple
from jax import numpy as jnp

from conf.config import CNNConfig, MLPConfig
from networks.MLP import MLP
from networks.CNN import CNN


def net_factory(
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, ...],
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
