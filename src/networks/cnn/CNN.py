from typing import Any, List

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from networks.cnn.config import CNNConfig
from networks.mlp.MLP import MLP
from networks.util import Network


class CNN(eqx.Module, Network):
    layers: list

    def __init__(self, layers: List[Any]):
        self.layers = layers

    @classmethod
    def from_config(
        cls,
        conf: CNNConfig,
        nchannels: int,
        dummy_data: jnp.ndarray,
        nclasses: int,
        key: int = 0,
    ) -> "CNN":
        rng = jr.PRNGKey(key)
        in_channels = nchannels
        blocks = []
        assert (
            len(conf.channels)
            == len(conf.kernel_sizes)
            == len(conf.paddings)
            == len(conf.strides)
        ), "channels and kernel_sizes must have the same length!"

        for out_channels, kernel_size, padding, stride in zip(
            conf.channels, conf.kernel_sizes, conf.paddings, conf.strides
        ):
            rng, _key = jr.split(rng)
            new_layer = [
                eqx.nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    padding=padding,
                    stride=stride,
                    key=_key,
                ),
                jax.nn.tanh,
                eqx.nn.MaxPool2d(kernel_size=conf.pool_kernel_size),
            ]
            in_channels = out_channels
            blocks.append(new_layer)

        blocks.append([jnp.ravel])

        # Run a forward pass to determine the MLP input dimension.
        cnn_trunk = CNN(blocks)
        mlp_din = cnn_trunk(dummy_data).size

        rng, _key = jr.split(rng)
        mlp = MLP.from_config(conf.mlp, din=mlp_din, nclasses=nclasses, key=_key.sum().item())

        cnn = CNN(blocks + [[mlp]])
        cnn.reinitialize(rng)
        return cnn

    def reinitialize(self, key: chex.PRNGKey) -> "CNN":
        new_blocks = []
        for block in self.layers:
            new_block = []
            for layer in block:
                key, _key = jr.split(key)
                if isinstance(layer, eqx.nn.Conv2d):
                    new_block.append(
                        eqx.nn.Conv2d(
                            in_channels=layer.in_channels,
                            out_channels=layer.out_channels,
                            kernel_size=layer.kernel_size,
                            key=_key,
                            padding=layer.padding,
                            stride=layer.stride,
                        )
                    )
                elif isinstance(layer, MLP):
                    new_block.append(layer.reinitialize(key))
                else:
                    new_block.append(layer)
            new_blocks.append(new_block)

        return CNN(new_blocks)

    def __call__(self, x: jnp.ndarray):
        for block in self.layers:
            for layer in block:
                x = layer(x)
        return x

    def forward_through_block(self, x: jnp.ndarray, block_idx: int):
        for layer in self.layers[block_idx]:
            x = layer(x)
        return x

    def hidden_outputs(self, x: jnp.ndarray):
        outputs = jnp.zeros(1)
        for layer in self.layers:
            x = layer(x)
            outputs = jnp.concatenate([outputs, x], axis=0)
        return outputs[1:]

    def get_num_hidden_units(self):
        def get_out_dim(arr):
            if arr is None:
                return 0
            return jax.lax.select(len(arr.shape) > 1, 0, arr.shape[0])

        model_arrays, _ = eqx.partition(self.layers, eqx.is_array)
        hidden_dims = jax.tree.map(
            get_out_dim, model_arrays, is_leaf=lambda x: x is None
        )
        return jax.tree.reduce(lambda x, y: x + y, hidden_dims).item()
