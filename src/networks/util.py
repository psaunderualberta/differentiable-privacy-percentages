import chex
import equinox as eqx
import jax.lax as jlax
import jax.numpy as jnp
import jax.random as jr
from typing import Sequence
from jaxtyping import Array, PRNGKeyArray
from jax.experimental import checkify

class Network:
    pass


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
        weight = self.weight @ x
        return weight + self.bias


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
