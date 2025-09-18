from jax import vjp, grad, jit
import chex
from typing import Callable, Any
from functools import partial


@partial(jit, static_argnames=("f",))
def vhp(f: Callable[..., chex.Array], x: tuple[Any], v: chex.Array) -> Callable[[chex.Array], chex.Array]:
    """
    Vector-Hessian product of f at x in the direction of v.
    Args:
        f: A function which computes the Jacobian of a scalar-valued function.
        x: The point where the Hessian is evaluated.
        v: The vector to be multiplied with the Hessian.
    Returns:
        A function that takes a vector and returns the vector-Hessian product v^T H(x)
    
    """
    _, vjp_fun = vjp(grad(f), *x)
    return vjp_fun(v)[0]

