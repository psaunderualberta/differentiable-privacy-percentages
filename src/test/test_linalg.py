from util.linalg import vhp
import jax.numpy as jnp
import jax.random as jr
import chex
from jax import hessian
import equinox as eqx
import jax.tree as jt


def __cube_sum(x):
    return jnp.sum(x**3)


def __shift_sum(x):
    return jnp.sum(x[:-2] * x[1:-1] * x[2:])



""" Basic Tests """
def test_vhp_1():
    x = jnp.array([1.0, 2.0, 3.0])
    v = jnp.array([0.1, 0.2, 0.3])
    vhp_result = vhp(__cube_sum, (x,), v)
    expected_result = v @ hessian(__cube_sum)(x)
    assert jnp.allclose(vhp_result, expected_result), f"Expected {expected_result}, but got {vhp_result}"


def test_vhp_2():
    x = jnp.array([0.0, 0.0, 0.0])
    v = jnp.array([1.0, 1.0, 1.0])
    vhp_result = vhp(__cube_sum, (x,), v)
    expected_result = v @ hessian(__cube_sum)(x)
    assert jnp.allclose(vhp_result, expected_result), f"Expected {expected_result}, but got {vhp_result}"


def test_vhp_3():
    x = jnp.array([-1.0, -2.0, -3.0])
    v = jnp.array([0.5, 0.5, 0.5])
    vhp_result = vhp(__shift_sum, (x,), v)
    actual_hessian = hessian(__shift_sum)(x)
    assert jnp.nonzero(actual_hessian), "Hessian should have non-zero entries at every element"
    expected_result = v @ hessian(__shift_sum)(x)
    assert jnp.allclose(vhp_result, expected_result), f"Expected {expected_result}, but got {vhp_result}"


