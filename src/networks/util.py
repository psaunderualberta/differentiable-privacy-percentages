import chex
import equinox as eqx
import jax.image as jimage
import jax.lax as jlax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap


class Network:
    pass


class ReLU(eqx.Module):
    def __call__(self, x: chex.Array):
        return jnp.maximum(x, 0.0)


class Flatten(eqx.Module):
    def __call__(self, x: chex.Array):
        return jnp.ravel(x)


class MaxPool2d(eqx.Module):
    kernel_size: int
    stride: int

    def __init__(self, kernel_size: int, stride: int):
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, x: chex.Array):
        def per_channel_pool(channel: chex.Array):
            m, n = channel.shape # type: ignore

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
    c, h, w = image.shape # type: ignore
    x = jr.randint(xkey, (1,), 0, h - patch_size)
    y = jr.randint(ykey, (1,), 0, w - patch_size)
    image = jlax.dynamic_slice(image, (0, x[0], y[0]), (c, patch_size, patch_size)) # type: ignore

    # Randomly flip the image along the left-right direction
    image = jlax.select(jr.uniform(key) > 0.5, image, jnp.flip(image, axis=1))

    # Randomly distort the brightness and the contrast of the image
    brightness = jr.uniform(brightness_key, (1,), minval=-0.2, maxval=0.2)
    contrast = jr.uniform(contrast_key, (1,), minval=-0.2, maxval=0.2)
    image = image * (1 + contrast) + brightness

    # Clip the image to be in the range [0, 1]
    image = jnp.clip(image, 0, 1)

    return image
