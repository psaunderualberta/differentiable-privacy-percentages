from functools import partial
from typing import Any, Callable

import chex
import jax.numpy as jnp
from jax import grad, jit, jvp, vjp

"""
Much of this code is adapted from: https://www.cs.toronto.edu/~rgrosse/courses/csc2541_2022/readings/L02_Taylor_approximations.pdf
"""


@partial(jit, static_argnames=("J",))
def hvp(J, w, v):
    """
    Hessian-vector product of J at w in the direction of v.

    Args:
        J: A function which computes the Jacobian of a scalar-valued function.
        w: The point where the Hessian is evaluated.
        v: The vector to be multiplied with the Hessian.
    Returns:
        The Hessian-vector product H(w) @ v
    """
    return jvp(grad(J), w, v)[1]


@partial(jit, static_argnames=("f",))
def vhp(
    f: Callable[..., chex.Array], x: tuple[chex.Array], v: tuple[chex.Array]
) -> Callable[[chex.Array], chex.Array]:
    """
    Vector-Hessian product of f at x in the direction of v.
    Args:
        f: A function which computes the Jacobian of a scalar-valued function.
        x: The point where the Hessian is evaluated.
        v: The vector to be multiplied with the Hessian.
    Returns:
        A function that takes a vector and returns the vector-Hessian product v^T H(x)


    Note: This implementation assumes a twice-differentiable 'f'. For non-smooth functions, consider using
    gnhvp linked below.

    """
    _, vjp_fun = vjp(grad(f), *x)
    return vjp_fun(*v)[0]


@partial(jit, static_argnames=("f", "L"))
def gnhvp(
    f: Callable[..., chex.Array],
    L: Callable[..., chex.Array],
    x: tuple[Any],
    v: chex.Array,
) -> Callable[[chex.Array], chex.Array]:
    """
    Approximation of the Vector-Hessian product of f at x in the direction of v,
    using the generalized Gauss-Newton approximation with loss function L.
     The generalized Gauss-Newton matrix is defined as J^T H J, where J is the Jacobian of f
     and H is the Hessian of L with respect to f.
    Args:
        f: A function which computes the Jacobian of a vector-valued function.
        L: A twice-differentiable scalar-value function (i.e. loss function).
        x: The point where the Hessian is evaluated.
        v: The vector to be multiplied with the Hessian.
    Returns:
        A function that takes a vector and returns the vector-Hessian product v^T J^T H(x) J

    """
    z, f_vjp = vjp(f, *x)
    _, f_vjp_vjp = vjp(f_vjp, jnp.zeros_like(z))
    R_z = f_vjp_vjp(v)[0]  # J @ v
    R_gz = hvp(L, (z,), (R_z,))  # H @ J_f @ v
    return f_vjp(R_gz)[0]  # J^T @ H @ J @ v
