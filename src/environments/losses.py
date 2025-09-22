import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
import chex
from typing import Callable, Tuple
from functools import partial
from conf.singleton_conf import SingletonConfig


@eqx.filter_value_and_grad
@partial(jax.jit, static_argnames=("to_vmap",))
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
    loss_type = SingletonConfig.get_environment_config_instance().loss_type
    if to_vmap:
        model = jax.vmap(model)
    pred_y = model(x)
    if loss_type == "mse":
        return ((pred_y - y) ** 2).mean()
    elif loss_type == "cce":
        # Directly from optax softmax_cross_entropy
        log_probs = jax.nn.log_softmax(pred_y, -1, None)
        softmax_cross_entropy = -(y * log_probs).sum(-1, where=None)
        return softmax_cross_entropy.mean()
    
    raise ValueError(f"Unknown loss type in 'loss_helper: {loss_type}")

def __loss_hessian(cls, pred_y: chex.Array, y: chex.Array) -> chex.Array:
    """
    Compute the Hessian of the loss with respect to the model parameters.
    Source: https://cnguyen10.github.io/posts/Gauss-Newton-matrix/
    """
    loss_type = SingletonConfig.get_environment_config_instance().loss_type
    
    if loss_type == "mse":
        """
        d/dx(d/dx 1/2 * (x - y)**2)
        = d/dx(x - y)
        = 1.0
        """
        return jnp.ones(1)

    elif loss_type == "cce":
        softmax = jax.nn.log_softmax(pred_y, -1, None)
        print(softmax)
        exit()

   
    raise ValueError(f"Unknown loss type in 'loss_hessian': {loss_type}")


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
    losses, grads = jax.vmap(__loss_helper, in_axes=(None, 0, 0, None))(model, x, y, False)
    return losses.mean(), grads


