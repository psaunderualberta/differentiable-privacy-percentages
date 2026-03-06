from typing import Any

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from networks._registry import register
from networks.cnn.config import CNNConfig
from networks.mlp.MLP import MLP
from networks.util import Network


@register(CNNConfig)
class CNN(eqx.Module, Network):
    layers: list

    def __init__(self, layers: list[Any]):
        """Store the pre-built list of layer blocks.

        Args:
            layers: Nested list of callable layers/activations grouped into blocks.
        """
        self.layers = layers

    @classmethod
    def build(
        cls,
        conf: CNNConfig,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        key: int = 0,
    ) -> "CNN":
        """Construct a CNN from a config and dataset shapes.

        Args:
            conf: CNNConfig specifying convolutional and MLP sub-network parameters.
            input_shape: Shape of the input batch (N, C, H, W).
            output_shape: Shape of the output batch (N, nclasses).
            key: Integer seed for weight initialization.
        """
        nchannels = input_shape[1]
        dummy_data = jnp.zeros(input_shape[1:])
        nclasses = output_shape[1]
        return cls.from_config(
            conf,
            nchannels=nchannels,
            dummy_data=dummy_data,
            nclasses=nclasses,
            key=key,
        )

    @classmethod
    def from_config(
        cls,
        conf: CNNConfig,
        nchannels: int,
        dummy_data: jnp.ndarray,
        nclasses: int,
        key: int = 0,
    ) -> "CNN":
        """Build the convolutional trunk and MLP head from explicit dimensions.

        Runs a dummy forward pass to determine the flattened dimension fed into the MLP.

        Args:
            conf: CNNConfig specifying channels, kernel sizes, paddings, strides, and MLP config.
            nchannels: Number of input channels.
            dummy_data: Zero-valued array with shape (C, H, W) used to infer the MLP input size.
            nclasses: Number of output classes.
            key: Integer seed for weight initialization.
        """
        rng = jr.PRNGKey(key)
        in_channels = nchannels
        blocks = []
        assert (
            len(conf.channels) == len(conf.kernel_sizes) == len(conf.paddings) == len(conf.strides)
        ), "channels and kernel_sizes must have the same length!"

        for out_channels, kernel_size, padding, stride in zip(
            conf.channels,
            conf.kernel_sizes,
            conf.paddings,
            conf.strides,
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

        cnn = CNN([*blocks, [mlp]])
        cnn.reinitialize(rng)
        return cnn

    def reinitialize(self, key: chex.PRNGKey) -> "CNN":
        """Return a new CNN with freshly randomized Conv2d and MLP weights; other layers are unchanged.

        Args:
            key: PRNG key used to draw new weights.
        """
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
                        ),
                    )
                elif isinstance(layer, MLP):
                    new_block.append(layer.reinitialize(key))
                else:
                    new_block.append(layer)
            new_blocks.append(new_block)

        return CNN(new_blocks)

    def __call__(self, x: jnp.ndarray):
        """Run a forward pass through all layer blocks and return the output."""
        for block in self.layers:
            for layer in block:
                x = layer(x)
        return x

    def forward_through_block(self, x: jnp.ndarray, block_idx: int):
        """Run a forward pass through a single block identified by `block_idx`."""
        for layer in self.layers[block_idx]:
            x = layer(x)
        return x

    def hidden_outputs(self, x: jnp.ndarray):
        """Return the concatenated outputs of every block (excluding the input)."""
        outputs = jnp.zeros(1)
        for layer in self.layers:
            x = layer(x)
            outputs = jnp.concatenate([outputs, x], axis=0)
        return outputs[1:]

    def get_num_hidden_units(self):
        """Return the total number of scalar hidden units across all weight arrays."""

        def get_out_dim(arr):
            if arr is None:
                return 0
            return jax.lax.select(len(arr.shape) > 1, 0, arr.shape[0])

        model_arrays, _ = eqx.partition(self.layers, eqx.is_array)
        hidden_dims = jax.tree.map(
            get_out_dim,
            model_arrays,
            is_leaf=lambda x: x is None,
        )
        return jax.tree.reduce(lambda x, y: x + y, hidden_dims).item()
