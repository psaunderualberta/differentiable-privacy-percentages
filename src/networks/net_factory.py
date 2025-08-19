from dataclasses import replace
from typing import Tuple

from conf.config import CNNConfig, MLPConfig
from networks.nets import MLP


def net_factory(
    input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], conf: MLPConfig | CNNConfig
):
    assert len(input_shape) >= 2, "Input shape must have at least 2 dimensions"
    assert len(output_shape) >= 2, "Output shape must have at least 2 dimensions"

    if isinstance(conf, MLPConfig):
        conf = replace(conf, din=input_shape[1], nclasses=output_shape[1])
        return MLP.from_config(conf)
    # if isinstance(conf, CNNConfig):
    #     conf = replace(conf, nchannels=input_shape[1], nclasses=output_shape[1])
    #     return CNN.from_config(conf)

    raise ValueError(f"Network config of type '{type(conf)}' is not recognized!")
