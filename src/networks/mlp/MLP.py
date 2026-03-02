from typing import Any, List

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from networks._registry import register
from networks.mlp.config import MLPConfig
from networks.util import Linear, Network


@register(MLPConfig)
class MLP(eqx.Module, Network):
    layers: list

    def __init__(self, layers: List[Any]):
        self.layers = layers

    @classmethod
    def build(
        cls,
        conf: MLPConfig,
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
        key: int = 0,
    ) -> "MLP":
        import jax.numpy as jnp

        din = int(jnp.prod(jnp.asarray(input_shape[1:])))
        nclasses = output_shape[1]
        return cls.from_config(conf, din=din, nclasses=nclasses, key=key)

    @classmethod
    def from_config(
        cls, conf: MLPConfig, din: int, nclasses: int, key: int = 0
    ) -> "MLP":
        rng = jr.PRNGKey(key)
        rng, _key = jr.split(rng)
        has_hidden = len(conf.hidden_sizes) > 0
        layer_out = conf.hidden_sizes[0] if has_hidden else nclasses
        layers = [
            [
                Linear(din, layer_out, key=_key, initialization=conf.initialization),
                eqx.nn.LayerNorm(layer_out),
            ]
        ]

        layer_in = layer_out
        if has_hidden:
            for layer_out in conf.hidden_sizes[1:]:
                rng, _key = jr.split(rng)
                layers.append(
                    [
                        jax.nn.tanh,
                        Linear(
                            layer_in,
                            layer_out,
                            key=_key,
                            initialization=conf.initialization,
                        ),
                        eqx.nn.LayerNorm(layer_out),
                    ]
                )
                layer_in = layer_out

            rng, _key = jr.split(rng)
            layers.append(
                [
                    jax.nn.tanh,
                    Linear(
                        layer_out,
                        nclasses,
                        key=_key,
                        initialization=conf.initialization,
                    ),
                ]
            )

        return MLP(layers)

    def reinitialize(self, key: chex.PRNGKey) -> "MLP":

        new_blocks = []
        for block in self.layers:
            new_block = []
            for layer in block:
                key, _key = jr.split(key)
                if isinstance(layer, Linear):
                    new_block.append(
                        Linear(
                            din=layer.weight.shape[1],
                            dout=layer.weight.shape[0],
                            initialization=layer.initialization,
                            key=key,
                        )
                    )
                else:
                    new_block.append(layer)
            new_blocks.append(new_block)

        return MLP(new_blocks)

    def __call__(self, x: chex.Array):
        x = x.reshape(-1, 1).squeeze()
        for block in self.layers:
            for layer in block:
                x = layer(x)
        return x

    def forward_through_block(self, x: chex.Array, block_idx: int):
        for layer in self.layers[block_idx]:
            x = layer(x)
        return x

    def hidden_outputs(self, x: chex.Array):
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
