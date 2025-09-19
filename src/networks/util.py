import chex
import equinox as eqx
import jax.lax as jlax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, vjp
from util.linalg import gnhvp
from util.util import subtract_pytrees, add_pytrees, dot_pytrees, multiply_pytree_by_scalar


class Network:
    pass


class ReLU(eqx.Module):
    def __call__(self, x: chex.Array):
        return jnp.maximum(x, 0.0)


class Flatten(eqx.Module):
    def __call__(self, x: chex.Array):
        return jnp.ravel(x)


class Linear(eqx.Module):
    weight: chex.Array
    bias: chex.Array

    def __init__(self, din, dout, key: chex.PRNGKey, initialization: str = "glorot"):
        if initialization == "glorot":
            layer = eqx.nn.Linear(din, dout, key=key)
            assert layer.bias is not None
            self.weight = layer.weight
            self.bias = layer.bias
        elif initialization == "zeros":
            self.weight = jnp.zeros((dout, din))
            self.bias = jnp.zeros((dout,))
        else:
            raise ValueError(
                f"Initialization for Linear Layer '{initialization}' not known"
            )

    def __call__(self, x):
        return self.weight @ x + self.bias


class MaxPool2d(eqx.Module):
    kernel_size: int
    stride: int

    def __init__(self, kernel_size: int, stride: int):
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, x: chex.Array):
        def per_channel_pool(channel: chex.Array):
            m, n = channel.shape  # type: ignore

            mk = m // self.kernel_size
            nk = n // self.kernel_size

            return jlax.reshape(channel, (m // 2, 2, n // 2, 2)).max(axis=(1, 3))

        return vmap(per_channel_pool)(x)


def augment_image(
    image: chex.Array, key: chex.PRNGKey, patch_size: int = 24
) -> chex.Array:
    """
    Augment an image by randomly picking a 24 × 24 patch from the image,
    randomly flipping the image along the left-right direction,
    and randomly distorting the brightness and the contrast of the image.

    Args:
        image (chex.Array): The input image to augment.

    Returns:
        chex.Array: The augmented image.
    """

    # generate keys
    xkey, ykey, brightness_key, contrast_key = jr.split(key, 4)

    # Randomly pick a 24 × 24 patch from the image
    c, h, w = image.shape  # type: ignore
    x = jr.randint(xkey, (1,), 0, h - patch_size)
    y = jr.randint(ykey, (1,), 0, w - patch_size)
    image = jlax.dynamic_slice(image, (0, x[0], y[0]), (c, patch_size, patch_size))  # type: ignore

    # Randomly flip the image along the left-right direction
    image = jlax.select(jr.uniform(key) > 0.5, image, jnp.flip(image, axis=1))

    # Randomly distort the brightness and the contrast of the image
    brightness = jr.uniform(brightness_key, (1,), minval=-0.2, maxval=0.2)
    contrast = jr.uniform(contrast_key, (1,), minval=-0.2, maxval=0.2)
    image = image * (1 + contrast) + brightness

    # Clip the image to be in the range [0, 1]
    image = jnp.clip(image, 0, 1)

    return image


def compute_dwI_dsj(f, loss, alpha, ws, ns):
    """
    alpha: learning rate (scalar)
    f: Network function
      loss: loss function
    ws: vector of weights, where ws[i] = w_i
    ns: vector of noises, where ns[j] = n_j:
          Note: We could also keep a vector of the noise *keys*, since JAX
      is deterministic wrt keys
    """
    assert len(ws) - 1 == len(ns)
    derivatives = jnp.zeros((ns.shape[0],))
    prod = multiply_pytree_by_scalar(-alpha, vjp(f, ws[-1]))  # alpha * Jac(l(w_I))

    # Implement via jax fori_loop or scan over ns[::-1]
    for i in range(ns.shape[0] - 1, -1, -1):
        derivatives[i] = dot_pytrees(prod, ns[i])  #
        derivatives[i] = prod @ ns[i]

        # prod = prod * (I - \alpha * H(w[i]))
        prod = subtract_pytrees(prod, multiply_pytree_by_scalar(-alpha, gnhvp(f, loss, ws[i], prod)))

    return derivatives
