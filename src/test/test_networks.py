"""Tests for network architecture implementations.

Covers:
- Linear layer: construction, initialization modes, forward pass
- MLP: from_config variants, forward pass, reinitialize, forward_through_block,
        get_num_hidden_units
- CNN: from_config, forward pass, reinitialize, forward_through_block,
        mismatched-config assertion
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from networks.cnn.CNN import CNN
from networks.cnn.config import CNNConfig
from networks.mlp.config import MLPConfig
from networks.mlp.MLP import MLP
from networks.util import Linear

# ---------------------------------------------------------------------------
# Shared geometry constants
# ---------------------------------------------------------------------------

DIN = 28
NCLASSES = 10
HIDDEN = (32,)
HIDDEN2 = (32, 64)

# CNN image geometry (C, H, W)
NCHANNELS = 1
IMG_H, IMG_W = 32, 32
IMG_SHAPE = (NCHANNELS, IMG_H, IMG_W)
DUMMY_DATA = jnp.zeros(IMG_SHAPE)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def eqx_partition_array(x):
    return eqx.partition(x, eqx.is_array)[0]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mlp_single() -> MLP:
    return MLP.from_config(MLPConfig(hidden_sizes=HIDDEN), din=DIN, nclasses=NCLASSES)


@pytest.fixture
def mlp_deep() -> MLP:
    return MLP.from_config(MLPConfig(hidden_sizes=HIDDEN2), din=DIN, nclasses=NCLASSES)


@pytest.fixture
def mlp_no_hidden() -> MLP:
    return MLP.from_config(MLPConfig(hidden_sizes=()), din=DIN, nclasses=NCLASSES)


@pytest.fixture
def cnn() -> CNN:
    return CNN.from_config(
        CNNConfig(), nchannels=NCHANNELS, dummy_data=DUMMY_DATA, nclasses=NCLASSES
    )


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------


class TestLinear:
    def test_glorot_weight_shape(self):
        layer = Linear(DIN, NCLASSES, key=jr.PRNGKey(0), initialization="glorot")
        assert layer.weight.shape == (NCLASSES, DIN)
        assert layer.bias.shape == (NCLASSES,)

    def test_glorot_weights_nonzero(self):
        layer = Linear(DIN, NCLASSES, key=jr.PRNGKey(0), initialization="glorot")
        assert not jnp.all(layer.weight == 0)

    def test_zeros_init_all_zero(self):
        layer = Linear(DIN, NCLASSES, key=jr.PRNGKey(0), initialization="zeros")
        assert jnp.all(layer.weight == 0)
        assert jnp.all(layer.bias == 0)

    def test_forward_output_shape(self):
        layer = Linear(DIN, NCLASSES, key=jr.PRNGKey(0))
        x = jnp.ones(DIN)
        assert layer(x).shape == (NCLASSES,)

    def test_zeros_init_forward_is_zero(self):
        layer = Linear(DIN, NCLASSES, key=jr.PRNGKey(0), initialization="zeros")
        assert jnp.all(layer(jnp.ones(DIN)) == 0)

    def test_invalid_initialization_raises(self):
        with pytest.raises(ValueError, match="not known"):
            Linear(DIN, NCLASSES, key=jr.PRNGKey(0), initialization="xavier")

    def test_different_keys_give_different_weights(self):
        l1 = Linear(DIN, NCLASSES, key=jr.PRNGKey(0))
        l2 = Linear(DIN, NCLASSES, key=jr.PRNGKey(1))
        assert not jnp.allclose(l1.weight, l2.weight)


# ---------------------------------------------------------------------------
# MLP — construction
# ---------------------------------------------------------------------------


class TestMLPFromConfig:
    def test_single_hidden_layer_block_count(self, mlp_single):
        # hidden_sizes=(32,) → 2 blocks: [Linear+LN] + [tanh+Linear]
        assert len(mlp_single.layers) == 2

    def test_two_hidden_layers_block_count(self, mlp_deep):
        # hidden_sizes=(32, 64) → 3 blocks
        assert len(mlp_deep.layers) == 3

    def test_no_hidden_layers_block_count(self, mlp_no_hidden):
        # hidden_sizes=() → 1 block: [Linear(din→nclasses)+LN]
        assert len(mlp_no_hidden.layers) == 1

    def test_glorot_weights_nonzero(self, mlp_single):
        arrays = eqx_partition_array(mlp_single.layers)
        flat = jax.tree.leaves(arrays)
        assert any(not jnp.all(a == 0) for a in flat)

    def test_zeros_init_weights_all_zero(self):
        mlp = MLP.from_config(
            MLPConfig(hidden_sizes=HIDDEN, initialization="zeros"),
            din=DIN,
            nclasses=NCLASSES,
        )
        arrays = eqx_partition_array(mlp.layers)
        flat = jax.tree.leaves(arrays)
        # Linear weights and biases are zero; LayerNorm is initialised by eqx.
        linear_layers = [
            a
            for a in flat
            if a.ndim == 2 or (a.ndim == 1 and a.shape[0] in (DIN, *HIDDEN, NCLASSES))
        ]
        # Just verify that 2D weight matrices are zero.
        weight_matrices = [a for a in flat if a.ndim == 2]
        assert all(jnp.all(w == 0) for w in weight_matrices)

    def test_different_keys_give_different_networks(self):
        mlp0 = MLP.from_config(
            MLPConfig(hidden_sizes=HIDDEN), din=DIN, nclasses=NCLASSES, key=0
        )
        mlp1 = MLP.from_config(
            MLPConfig(hidden_sizes=HIDDEN), din=DIN, nclasses=NCLASSES, key=1
        )
        w0 = eqx_partition_array(mlp0.layers)
        w1 = eqx_partition_array(mlp1.layers)
        flat0 = jax.tree.leaves(w0)
        flat1 = jax.tree.leaves(w1)
        assert any(not jnp.allclose(a, b) for a, b in zip(flat0, flat1))

    def test_same_key_gives_identical_networks(self):
        mlp0 = MLP.from_config(
            MLPConfig(hidden_sizes=HIDDEN), din=DIN, nclasses=NCLASSES, key=42
        )
        mlp1 = MLP.from_config(
            MLPConfig(hidden_sizes=HIDDEN), din=DIN, nclasses=NCLASSES, key=42
        )
        w0 = eqx_partition_array(mlp0.layers)
        w1 = eqx_partition_array(mlp1.layers)
        assert all(
            jnp.allclose(a, b) for a, b in zip(jax.tree.leaves(w0), jax.tree.leaves(w1))
        )


# ---------------------------------------------------------------------------
# MLP — forward pass
# ---------------------------------------------------------------------------


class TestMLPCall:
    def test_output_shape_1d_input(self, mlp_single):
        x = jnp.ones(DIN)
        assert mlp_single(x).shape == (NCLASSES,)

    def test_output_shape_deep(self, mlp_deep):
        x = jnp.ones(DIN)
        assert mlp_deep(x).shape == (NCLASSES,)

    def test_output_shape_no_hidden(self, mlp_no_hidden):
        x = jnp.ones(DIN)
        assert mlp_no_hidden(x).shape == (NCLASSES,)

    def test_deterministic(self, mlp_single):
        x = jnp.ones(DIN)
        assert jnp.allclose(mlp_single(x), mlp_single(x))

    def test_vmap_output_shape(self, mlp_single):
        batch = jnp.ones((8, DIN))
        out = jax.vmap(mlp_single)(batch)
        assert out.shape == (8, NCLASSES)

    def test_zeros_init_output_zero_for_any_input(self):
        mlp = MLP.from_config(
            MLPConfig(hidden_sizes=HIDDEN, initialization="zeros"),
            din=DIN,
            nclasses=NCLASSES,
        )
        x = jnp.ones(DIN) * 42.0
        assert jnp.allclose(mlp(x), jnp.zeros(NCLASSES))


# ---------------------------------------------------------------------------
# MLP — reinitialize
# ---------------------------------------------------------------------------


class TestMLPReinitialize:
    def test_same_shapes_and_dtypes(self, mlp_single):
        reinit = mlp_single.reinitialize(jr.PRNGKey(0))
        mlp_s_w = eqx_partition_array(mlp_single.layers)
        reinit_w = eqx_partition_array(reinit.layers)
        chex.assert_trees_all_equal_shapes_and_dtypes(mlp_s_w, reinit_w)

    def test_weights_differ_after_reinit(self, mlp_single):
        reinit = mlp_single.reinitialize(jr.PRNGKey(99))
        w_orig = eqx_partition_array(mlp_single.layers)
        w_new = eqx_partition_array(reinit.layers)
        flat_orig = [a for a in jax.tree.leaves(w_orig)]
        flat_new = [a for a in jax.tree.leaves(w_new)]
        chex.assert_trees_all_equal_shapes_and_dtypes(flat_orig, flat_new)
        assert not any(
            jnp.allclose(a, b) for a, b in zip(flat_orig, flat_new) if a.ndim >= 2
        )

    def test_different_keys_give_different_reinits(self, mlp_single):
        r1 = mlp_single.reinitialize(jr.PRNGKey(1))
        r2 = mlp_single.reinitialize(jr.PRNGKey(2))
        w1 = eqx_partition_array(r1.layers)
        w2 = eqx_partition_array(r2.layers)
        flat1 = [a for a in jax.tree.leaves(w1)]
        flat2 = [a for a in jax.tree.leaves(w2)]
        assert any(not jnp.allclose(a, b, atol=1e-3) for a, b in zip(flat1, flat2))

    def test_same_key_gives_identical_reinits(self, mlp_single):
        r1 = mlp_single.reinitialize(jr.PRNGKey(0))
        r2 = mlp_single.reinitialize(jr.PRNGKey(0))
        w1 = eqx_partition_array(r1.layers)
        w2 = eqx_partition_array(r2.layers)
        chex.assert_trees_all_close(w1, w2)

    def test_deep_mlp_reinitializes_correctly(self, mlp_deep):
        reinit = mlp_deep.reinitialize(jr.PRNGKey(0))
        w1 = eqx_partition_array(mlp_deep.layers)
        w2 = eqx_partition_array(reinit.layers)
        chex.assert_trees_all_equal_shapes_and_dtypes(w1, w2)


# ---------------------------------------------------------------------------
# MLP — forward_through_block
# ---------------------------------------------------------------------------


class TestMLPForwardThroughBlock:
    def test_block0_output_shape(self, mlp_single):
        x = jnp.ones(DIN)
        out = mlp_single.forward_through_block(x, 0)
        assert out.shape == (HIDDEN[0],)

    def test_block0_matches_partial_call(self, mlp_single):
        # forward_through_block(0) should equal the first step of __call__.
        x = jnp.ones(DIN)
        block0_out = mlp_single.forward_through_block(x, 0)
        # Manually run block 0
        manual = x.reshape(-1, 1).squeeze()
        for layer in mlp_single.layers[0]:
            manual = layer(manual)
        assert jnp.allclose(block0_out, manual)

    def test_deep_mlp_intermediate_shape(self, mlp_deep):
        x = jnp.ones(DIN)
        out = mlp_deep.forward_through_block(x, 0)
        assert out.shape == (HIDDEN2[0],)


# ---------------------------------------------------------------------------
# MLP — get_num_hidden_units
# ---------------------------------------------------------------------------


class TestMLPGetNumHiddenUnits:
    def test_positive_for_single_hidden(self, mlp_single):
        assert mlp_single.get_num_hidden_units() > 0

    def test_positive_for_no_hidden(self, mlp_no_hidden):
        # Even with no hidden layers, biases and LayerNorm params count.
        assert mlp_no_hidden.get_num_hidden_units() > 0

    def test_more_hidden_units_for_larger_arch(self, mlp_single, mlp_deep):
        assert mlp_deep.get_num_hidden_units() > mlp_single.get_num_hidden_units()

    def test_known_single_hidden_count(self):
        # hidden_sizes=(32,), nclasses=10, din=28:
        # Block 0: Linear.bias(32) + LN.weight(32) + LN.bias(32) = 96
        # Block 1: Linear.bias(10)                                = 10
        # Total = 106
        mlp = MLP.from_config(MLPConfig(hidden_sizes=(32,)), din=28, nclasses=10)
        assert mlp.get_num_hidden_units() == 106

    def test_known_no_hidden_count(self):
        # hidden_sizes=(), nclasses=10, din=28:
        # Block 0: Linear.bias(10) + LN.weight(10) + LN.bias(10) = 30
        mlp = MLP.from_config(MLPConfig(hidden_sizes=()), din=28, nclasses=10)
        assert mlp.get_num_hidden_units() == 30


# ---------------------------------------------------------------------------
# CNN — construction
# ---------------------------------------------------------------------------


class TestCNNFromConfig:
    def test_correct_num_blocks(self, cnn):
        # Default CNNConfig has 2 conv stages + ravel block + MLP block = 4 blocks.
        assert len(cnn.layers) == 4

    def test_mismatched_config_lengths_raises(self):
        with pytest.raises(AssertionError):
            CNN.from_config(
                CNNConfig(channels=(16,), kernel_sizes=(8, 4)),  # length mismatch
                nchannels=NCHANNELS,
                dummy_data=DUMMY_DATA,
                nclasses=NCLASSES,
            )

    def test_single_conv_stage(self):
        conf = CNNConfig(
            channels=(8,),
            kernel_sizes=(3,),
            paddings=(1,),
            strides=(1,),
            pool_kernel_size=2,
        )
        net = CNN.from_config(
            conf, nchannels=NCHANNELS, dummy_data=DUMMY_DATA, nclasses=NCLASSES
        )
        # 1 conv stage + ravel + mlp = 3 blocks
        assert len(net.layers) == 3

    def test_different_keys_give_different_networks(self):
        n0 = CNN.from_config(
            CNNConfig(),
            nchannels=NCHANNELS,
            dummy_data=DUMMY_DATA,
            nclasses=NCLASSES,
            key=0,
        )
        n1 = CNN.from_config(
            CNNConfig(),
            nchannels=NCHANNELS,
            dummy_data=DUMMY_DATA,
            nclasses=NCLASSES,
            key=1,
        )
        w0 = eqx_partition_array(n0.layers)
        w1 = eqx_partition_array(n1.layers)
        flat0 = [a for a in jax.tree.leaves(w0) if a.ndim >= 2]
        flat1 = [a for a in jax.tree.leaves(w1) if a.ndim >= 2]
        assert any(not jnp.allclose(a, b) for a, b in zip(flat0, flat1))


# ---------------------------------------------------------------------------
# CNN — forward pass
# ---------------------------------------------------------------------------


class TestCNNCall:
    def test_output_shape(self, cnn):
        x = jnp.ones(IMG_SHAPE)
        assert cnn(x).shape == (NCLASSES,)

    def test_deterministic(self, cnn):
        x = jnp.ones(IMG_SHAPE)
        assert jnp.allclose(cnn(x), cnn(x))

    def test_vmap_output_shape(self, cnn):
        batch = jnp.ones((4, *IMG_SHAPE))
        out = jax.vmap(cnn)(batch)
        assert out.shape == (4, NCLASSES)

    def test_different_input_different_output(self, cnn):
        x1 = jnp.ones(IMG_SHAPE)
        x2 = jnp.zeros(IMG_SHAPE)
        # With glorot init the outputs should differ for different inputs.
        assert not jnp.allclose(cnn(x1), cnn(x2))


# ---------------------------------------------------------------------------
# CNN — reinitialize
# ---------------------------------------------------------------------------


class TestCNNReinitialize:
    def test_same_tree_structure(self, cnn):
        reinit = cnn.reinitialize(jr.PRNGKey(0))
        # MaxPool has non-array leaves; compare structure only.
        chex.assert_trees_all_equal_structs(cnn, reinit)

    def test_conv_weights_differ_after_reinit(self, cnn):
        reinit = cnn.reinitialize(jr.PRNGKey(42))
        w_orig = eqx_partition_array(cnn.layers)
        w_new = eqx_partition_array(reinit.layers)
        # Compare 4D conv kernels.
        flat_orig = [a for a in jax.tree.leaves(w_orig) if a.ndim == 4]
        flat_new = [a for a in jax.tree.leaves(w_new) if a.ndim == 4]
        assert any(not jnp.allclose(a, b) for a, b in zip(flat_orig, flat_new))

    def test_different_keys_give_different_reinits(self, cnn):
        r1 = cnn.reinitialize(jr.PRNGKey(1))
        r2 = cnn.reinitialize(jr.PRNGKey(2))
        w1 = eqx_partition_array(r1.layers)
        w2 = eqx_partition_array(r2.layers)
        flat1 = [a for a in jax.tree.leaves(w1) if a.ndim == 4]
        flat2 = [a for a in jax.tree.leaves(w2) if a.ndim == 4]
        assert any(not jnp.allclose(a, b) for a, b in zip(flat1, flat2))

    def test_same_key_gives_identical_reinits(self, cnn):
        r1 = cnn.reinitialize(jr.PRNGKey(0))
        r2 = cnn.reinitialize(jr.PRNGKey(0))
        w1 = eqx_partition_array(r1.layers)
        w2 = eqx_partition_array(r2.layers)
        flat1 = jax.tree.leaves(w1)
        flat2 = jax.tree.leaves(w2)
        assert all(jnp.allclose(a, b) for a, b in zip(flat1, flat2))


# ---------------------------------------------------------------------------
# CNN — forward_through_block
# ---------------------------------------------------------------------------


class TestCNNForwardThroughBlock:
    def test_first_conv_block_output_3d(self, cnn):
        x = jnp.ones(IMG_SHAPE)
        out = cnn.forward_through_block(x, 0)
        # Conv2d + tanh + MaxPool → (out_channels, H', W')
        assert out.ndim == 3
        assert out.shape[0] == CNNConfig().channels[0]

    def test_second_conv_block_output_3d(self, cnn):
        x = jnp.ones(IMG_SHAPE)
        # Run first block manually, then second through forward_through_block.
        x = cnn.forward_through_block(x, 0)
        out = cnn.forward_through_block(x, 1)
        assert out.ndim == 3
        assert out.shape[0] == CNNConfig().channels[1]

    def test_ravel_block_output_1d(self, cnn):
        x = jnp.ones(IMG_SHAPE)
        x = cnn.forward_through_block(x, 0)
        x = cnn.forward_through_block(x, 1)
        out = cnn.forward_through_block(x, 2)
        assert out.ndim == 1

    def test_mlp_block_output_shape(self, cnn):
        x = jnp.ones(IMG_SHAPE)
        x = cnn.forward_through_block(x, 0)
        x = cnn.forward_through_block(x, 1)
        x = cnn.forward_through_block(x, 2)
        out = cnn.forward_through_block(x, 3)
        assert out.shape == (NCLASSES,)


# ---------------------------------------------------------------------------
# CNN — get_num_hidden_units
# ---------------------------------------------------------------------------


class TestCNNGetNumHiddenUnits:
    def test_positive(self, cnn):
        assert cnn.get_num_hidden_units() > 0

    def test_larger_mlp_more_units(self):
        small = CNN.from_config(
            CNNConfig(mlp=MLPConfig(hidden_sizes=(16,))),
            nchannels=NCHANNELS,
            dummy_data=DUMMY_DATA,
            nclasses=NCLASSES,
        )
        large = CNN.from_config(
            CNNConfig(mlp=MLPConfig(hidden_sizes=(128,))),
            nchannels=NCHANNELS,
            dummy_data=DUMMY_DATA,
            nclasses=NCLASSES,
        )
        assert large.get_num_hidden_units() > small.get_num_hidden_units()
