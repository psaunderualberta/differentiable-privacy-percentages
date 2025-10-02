from networks.MLP import MLP
from networks.CNN import CNN
from conf.config import MLPConfig, CNNConfig

import jax.numpy as jnp
import jax.random as jr
import chex


def test_mlp_reinitialize():
    """Test Reinitialized MLP has same shape"""
    mlp_conf = MLPConfig(din=28, nclasses=10) # MNIST shape
    network = MLP.from_config(mlp_conf)
    reinit_network = network.reinitialize(jr.PRNGKey(0))

    chex.assert_trees_all_equal_shapes_and_dtypes(network, reinit_network)


def test_cnn_reinitialize():
    """Test Reinitialized CNN has same shape"""
    mlp_conf = CNNConfig(
        mlp=MLPConfig(nclasses=10),
        nchannels=1,
        dummy_data=jnp.zeros((1, 32, 32))
    )
    network = CNN.from_config(mlp_conf)
    reinit_network = network.reinitialize(jr.PRNGKey(0))

    # MaxPool has an 'init-max' attribute without a shape, cannot assert it's shape
    chex.assert_trees_all_equal_structs(network, reinit_network)

    