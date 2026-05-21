"""Property-based tests for networks/mlp/.

Covers: MLP reinitialise determinism and vmap shape contract.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from networks.mlp.config import MLPConfig
from networks.mlp.MLP import MLP

from ._shared import _jax_settings

_DIN = 28
_NCLASSES = 10
_HIDDEN = (32,)

_MLP = MLP.from_config(MLPConfig(hidden_sizes=_HIDDEN), din=_DIN, nclasses=_NCLASSES)


class TestMLPProperties:
    """Universal invariants of the MLP module."""

    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    @_jax_settings
    def test_reinitialize_deterministic_for_any_key(self, seed):
        """reinitialize(key) is a pure function: same key always → same weights."""
        key = jr.PRNGKey(seed)
        r1 = _MLP.reinitialize(key)
        r2 = _MLP.reinitialize(key)
        leaves1 = jax.tree.leaves(eqx.partition(r1.layers, eqx.is_array)[0])
        leaves2 = jax.tree.leaves(eqx.partition(r2.layers, eqx.is_array)[0])
        assert all(jnp.allclose(a, b) for a, b in zip(leaves1, leaves2))

    @given(
        seed1=st.integers(min_value=0, max_value=2**31 - 1),
        seed2=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @_jax_settings
    def test_different_keys_give_different_weights(self, seed1, seed2):
        """Distinct PRNG keys produce distinct network initialisations."""
        assume(seed1 != seed2)
        r1 = _MLP.reinitialize(jr.PRNGKey(seed1))
        r2 = _MLP.reinitialize(jr.PRNGKey(seed2))
        w1 = jax.tree.leaves(eqx.partition(r1.layers, eqx.is_array)[0])
        w2 = jax.tree.leaves(eqx.partition(r2.layers, eqx.is_array)[0])
        assert any(not jnp.allclose(a, b) for a, b in zip(w1, w2) if a.ndim >= 2)

    @given(batch_size=st.integers(min_value=1, max_value=16))
    @settings(max_examples=5, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_vmap_output_shape_for_any_batch_size(self, batch_size):
        """jax.vmap over an MLP always yields (batch_size, nclasses) output."""
        batch = jnp.ones((batch_size, _DIN))
        out = jax.vmap(_MLP)(batch)
        assert out.shape == (batch_size, _NCLASSES)
