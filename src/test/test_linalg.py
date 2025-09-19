import jax.numpy as jnp
from jax import hessian, jacobian

from util.linalg import gnhvp, hvp, vhp


def __cube_sum(x):
    return jnp.sum(x**3)


def __shift_sum(x):
    return jnp.sum(x[:-2] * x[1:-1] * x[2:])


""" Basic Tests for vhp"""


def test_vhp_1():
    x = jnp.array([1.0, 2.0, 3.0])
    v = jnp.array([0.1, 0.2, 0.3])
    vhp_result = vhp(__cube_sum, (x,), (v,))
    expected_result = v @ hessian(__cube_sum)(x)
    assert jnp.allclose(
        vhp_result, expected_result
    ), f"Expected {expected_result}, but got {vhp_result}"


def test_vhp_2():
    x = jnp.array([0.0, 0.0, 0.0])
    v = jnp.array([1.0, 1.0, 1.0])
    vhp_result = vhp(__cube_sum, (x,), (v,))
    expected_result = v @ hessian(__cube_sum)(x)
    assert jnp.allclose(
        vhp_result, expected_result
    ), f"Expected {expected_result}, but got {vhp_result}"


def test_vhp_3():
    x = jnp.array([-1.0, -2.0, -3.0])
    v = jnp.array([0.5, 0.5, 0.5])
    vhp_result = vhp(__shift_sum, (x,), (v,))
    actual_hessian = hessian(__shift_sum)(x)
    assert jnp.nonzero(
        actual_hessian
    ), "Hessian should have non-zero entries at every element"
    expected_result = v @ hessian(__shift_sum)(x)
    assert jnp.allclose(
        vhp_result, expected_result
    ), f"Expected {expected_result}, but got {vhp_result}"


""" Basic Tests for hvp"""


def test_hvp_1():
    x = jnp.array([1.0, 2.0, 3.0])
    v = jnp.array([0.1, 0.2, 0.3])
    hvp_result = hvp(__cube_sum, (x,), (v,))
    expected_result = hessian(__cube_sum)(x) @ v
    assert jnp.allclose(
        hvp_result, expected_result
    ), f"Expected {expected_result}, but got {hvp_result}"


def test_hvp_2():
    x = jnp.array([0.0, 0.0, 0.0])
    v = jnp.array([1.0, 1.0, 1.0])
    hvp_result = hvp(__cube_sum, (x,), (v,))
    expected_result = hessian(__cube_sum)(x) @ v
    assert jnp.allclose(
        hvp_result, expected_result
    ), f"Expected {expected_result}, but got {hvp_result}"


def test_hvp_3():
    x = jnp.array([-1.0, -2.0, -3.0])
    v = jnp.array([0.5, 0.5, 0.5])
    hvp_result = hvp(__shift_sum, (x,), (v,))
    actual_hessian = hessian(__shift_sum)(x)
    assert jnp.nonzero(
        actual_hessian
    ), "Hessian should have non-zero entries at every element"
    expected_result = hessian(__shift_sum)(x) @ v
    assert jnp.allclose(
        hvp_result, expected_result
    ), f"Expected {expected_result}, but got {hvp_result}"


""" Basic Tests for gnhvp"""


def test_gnhvp_1():
    def f(x):
        return x**2

    def L(z):
        return jnp.sum(z**2)

    x = jnp.array([1.0, 2.0])
    v = jnp.array([0.1, 0.2])
    gnhvp_result = gnhvp(f, L, (x,), (v,))
    z = f(x)
    Jz = jacobian(f)(x)
    H = hessian(L)(z)  # Hessian of L
    expected_result = Jz.T @ H @ Jz @ v
    assert jnp.allclose(
        gnhvp_result, expected_result
    ), f"Expected {expected_result}, but got {gnhvp_result}"


def test_gnhvp_2():
    def f(x):
        # Reshaping to ensure Jacobian is 2D array
        return (x.reshape(2, -1) @ x.reshape(-1, 2)).reshape(-1)

    def L(z):
        return jnp.sum(z**2)

    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    v = jnp.array([0.1, 0.2, 0.3, 0.4])
    gnhvp_result = gnhvp(f, L, (x,), (v,))
    z = f(x)
    Jz = jacobian(f)(x)
    H = hessian(L)(z)  # Hessian of L
    expected_result = Jz.T @ H @ Jz @ v
    assert jnp.allclose(
        gnhvp_result, expected_result
    ), f"Expected {str(expected_result)}, but got {str(gnhvp_result)}"

    # sanity check that gnhvp is symmetric
    expected_result = v @ Jz.T @ H @ Jz
    assert jnp.allclose(
        gnhvp_result, expected_result
    ), f"Expected {str(expected_result)}, but got {str(gnhvp_result)}"
