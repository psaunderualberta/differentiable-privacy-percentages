from typing import Any, List

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from conf.config import MLPConfig
from networks.util import Network, ReLU


class MLP(eqx.Module, Network):
    layers: list

    def __init__(self, layers: List[Any]):
        self.layers = layers

    @classmethod
    def from_config(cls, conf: MLPConfig) -> "MLP":
        key = jr.PRNGKey(conf.key)
        key, _key = jr.split(key)
        layer_out = conf.dhidden if conf.nhidden > 0 else conf.nclasses
        layers: list[list[ReLU | eqx.nn.Linear]] = [
            [
                eqx.nn.Linear(conf.din, layer_out, key=_key),
            ]
        ]

        if conf.nhidden > 0:
            for _ in range(conf.nhidden - 1):
                key, _key = jr.split(key)
                layers.append(
                    [
                        ReLU(),
                        eqx.nn.Linear(conf.dhidden, conf.dhidden, key=_key),
                    ]
                )

            key, _key = jr.split(key)
            layers.append(
                [
                    ReLU(),
                    eqx.nn.Linear(conf.dhidden, conf.nclasses, key=_key),
                ]
            )

        return MLP(layers)

    def reinitialize(self, key: chex.PRNGKey) -> "MLP":
        net_flat, net_treedef = jax.tree.flatten(self.layers)

        rngs = jax.random.split(key, len(net_flat))

        # Using an existing network means that the linear layer sizes are statically known
        new_net_flat = [
            (
                eqx.nn.Linear(layer.weight.shape[0], layer.weight.shape[1], key=rng)
                if hasattr(layer, "weight")
                else layer
            )
            for layer, rng in zip(net_flat, rngs)
        ]

        new_net_layers = jax.tree.unflatten(net_treedef, new_net_flat)
        return MLP(new_net_layers)

    def __call__(self, x: chex.Array):
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
