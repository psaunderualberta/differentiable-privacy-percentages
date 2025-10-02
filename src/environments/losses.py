import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental import checkify
import chex
from typing import Callable, Tuple
from functools import partial
from conf.singleton_conf import SingletonConfig
from util.linalg import hvp
from optax import softmax_cross_entropy
from util.util import pytree_has_inf


def __py_y_loss(pred_y: jnp.ndarray, y: jnp.ndarray) -> chex.Array:
    """
    Actually compute the losses between predicted 'y' values, and 'y' values

    Args:
        pred_y: Predicted Y values
        y: Actual y values
    
    Returns:
        The loss value
    """
    loss_type = SingletonConfig.get_environment_config_instance().loss_type
    if loss_type == "mse":
        loss_value = ((pred_y - y) ** 2).mean()
    elif loss_type == "cce":
        loss_value = softmax_cross_entropy(pred_y, y).mean()
    else:
        raise ValueError(f"Unknown loss type in 'py_y_loss: {loss_type}")

    return loss_value


@eqx.filter_value_and_grad
@eqx.filter_jit
def __loss_helper(model, x, y, to_vmap: bool = True) -> Tuple[chex.Array, chex.Array]:
    """
    Compute the loss and gradients for a given model and data.
    Args:
        model: The model to compute the loss and gradients for.
        x: Input data.
        y: Target data.
        to_vmap: Whether to vmap the model across the batch dimension. Default is True
    Returns:
        A tuple containing the loss and the gradients.
    """
    if to_vmap:
        model = jax.vmap(model)
    pred_y = model(x)

    return __py_y_loss(pred_y, y)  #type: ignore


# See page 16 of https://www.cs.toronto.edu/~rgrosse/courses/csc2541_2022/readings/L02_Taylor_approximations.pdf
def neural_net_gnhvp(
    model: Callable[..., chex.Array],
    x: jnp.ndarray,
    y: jnp.ndarray,
    v: eqx.Module,
):
    pred_y, model_vjp = jax.vjp(lambda model: jax.vmap(model)(x), model)
    _, model_vjp_vjp = jax.vjp(model_vjp, jnp.zeros_like(pred_y))
    J_v = model_vjp_vjp((v,))[0]
    H_J_v = hvp(lambda py: __py_y_loss(py, y), (pred_y,), (J_v,))
    JT_H_J_v = model_vjp(H_J_v)[0]  # J^T @ H @ J @ v
    return JT_H_J_v


@eqx.filter_jit
def loss(model: Callable[[chex.Array], jnp.ndarray], x: chex.Array, y: chex.Array):
    """
    Compute the loss and gradients for a given model and data.
    Args:
        model: The model to compute the loss and gradients for.
        x: Input data.
        y: Target data.
    Returns:
        A tuple containing the loss and the gradients.
    """
    return __loss_helper(model, x, y)


@eqx.filter_jit
def vmapped_loss(model: Callable[[chex.Array], jnp.ndarray], x: chex.Array, y: chex.Array):
    """
    Compute the loss and gradients using vmap across the model, producing per-example gradients.

    Args:
        model: The model to compute the loss and gradients for.
        x: Input data.
        y: Target data.
    Returns:
        A tuple containing the mean loss and the per-example gradients.

    """
    # model = eqx.error_if(model, jnp.logical_not(pytree_has_nan(model)), "Model has NaN values")
    losses, grads = jax.vmap(__loss_helper, in_axes=(None, 0, 0, None))(model, x, y, False)
    return losses.mean(), grads


