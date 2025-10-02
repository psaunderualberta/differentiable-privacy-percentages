from typing import Any, List

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from conf.config import CNNConfig
from networks.util import Network, ReLU
from dataclasses import replace
from networks.MLP import MLP



class CNN(eqx.Module, Network):
    layers: list

    class __ravel(eqx.Module):
        def __call__(self, x):
            return jnp.ravel(x)

    def __init__(self, layers: List[Any]):
        self.layers = layers

    @classmethod
    def from_config(cls, conf: CNNConfig) -> "CNN":
        key = jr.PRNGKey(conf.key)
        in_channels = conf.nchannels
        blocks = []
        for _ in range(conf.nhidden_conv):
            key, _key = jr.split(key)
            new_layer = [
                eqx.nn.Conv2d(in_channels, conf.hidden_channels, conf.kernel_size, key=_key),
                ReLU(),
                eqx.nn.MaxPool2d(
                    kernel_size=conf.kernel_size,
                )
            ]

            blocks.append(new_layer)

        blocks.append([CNN.__ravel()])

        in_channels = conf.hidden_channels

        cnn_wo_mlp = CNN(blocks)
        assert conf.dummy_data is not None, "CNN Configuration's dummy data must be filled!"
        dummy_out = cnn_wo_mlp(conf.dummy_data)

        # Create final MLP
        conf = replace(conf, mlp=replace(conf.mlp, din=dummy_out.size))
        mlp = MLP.from_config(conf.mlp)

        cnn = CNN(blocks + [[mlp]])
        cnn.reinitialize(key)
        return cnn

    def reinitialize(self, key: chex.PRNGKey) -> "CNN":
        new_blocks = []
        for block in self.layers:
            new_block = []
            for layer in block:
                key, _key = jr.split(key)
                if isinstance(layer, eqx.nn.Conv2d): 
                    new_block.append(eqx.nn.Conv2d(
                        in_channels=layer.in_channels,
                        out_channels=layer.out_channels,
                        kernel_size=layer.kernel_size,
                        key=_key,
                    ))
                elif isinstance(layer, MLP):
                    new_block.append(
                        layer.reinitialize(key)
                    )
                else:
                    # relu, MaxPool2d, ravel
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

            # only look at bias matrices, as o/w
            # we'll double count weight & bias matricess
            return jax.lax.select(len(arr.shape) > 1, 0, arr.shape[0])

        model_arrays, _ = eqx.partition(self.layers, eqx.is_array)
        hidden_dims = jax.tree.map(get_out_dim, model_arrays, is_leaf=lambda x: x is None)
        return jax.tree.reduce(lambda x, y: x + y, hidden_dims).item()


if __name__ == "__main__":
    LR = 1e-3
    EPOCHS = 100
    x = jnp.ones((10, 5))
    y = jnp.ones((10, 1))

    from conf.singleton_conf import SingletonConfig
    model_conf = SingletonConfig.get_environment_config_instance().mlp

    model = MLP.from_config(model_conf)
    optim = optax.sgd(LR)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def loss_fn(model, x, y):
        pred_y = jax.vmap(model)(x)
        return jnp.mean((pred_y - y) ** 2)

    def train_step(model, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)

        return model, opt_state, loss

    losses = []
    for _ in range(EPOCHS):
        model, opt_state, loss = train_step(model, opt_state, x, y)
        losses.append(loss)

    print(losses)
