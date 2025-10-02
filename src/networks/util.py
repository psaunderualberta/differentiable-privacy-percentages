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
        weight = self.weight @ x
        return weight + self.bias


# https://github.com/patrick-kidger/equinox/blob/27f8d7d7a09fa37e2186b62090ba4945fede50f5/equinox/nn/_pool.py#L299-L344
# Accessed Oct 2nd, 2025
class MaxPool2d(eqx.nn.Pool):
    """
    Two-dimensional downsample using the maximum over a sliding window.
    NB: Almost Identical to eqx.nn.MaxPool2d, but using a Jax-able operation
    """

    kernel_size: int
    stride: int
    dummy_kernel: Array

    class __max(eqx.Module):
        def __call__(self, x, y):
            return jlax.max(x, y)

    def __init__(
        self,
        dummy_kernel: Array,
        kernel_size: int | Sequence[int] = 2,
        stride: int | Sequence[int] = 1,
        padding: int | Sequence[int] | Sequence[tuple[int, int]] = 0,
        use_ceil: bool = False,
    ):
        """**Arguments:**

        - `kernel_size`: The size of the convolutional kernel.
        - `stride`: The stride of the convolution.
        - `padding`: The amount of padding to apply before and after each
            spatial dimension.
        - `use_ceil`: If `True`, then `ceil` is used to compute the final output
            shape instead of `floor`. For `ceil`, if required, extra padding is added.
            Defaults to `False`.
        """
        super().__init__(
            init=-jnp.inf,
            operation=MaxPool2d.__max(),
            num_spatial_dims=2,
            kernel_size=dummy_kernel.shape,
            stride=stride,
            padding=padding,
            use_ceil=use_ceil,
        )

        self.dummy_kernel = dummy_kernel

    def __call__(self, x: Array, *, key: PRNGKeyArray | None = None) -> Array:
            """**Arguments:**

            - `x`: The input. Should be a JAX array of shape `(channels, dim_1, dim_2)`.
            - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
                (Keyword only argument.)

            **Returns:**

            A JAX array of shape `(channels, new_dim_1, new_dim_2)`.
            """

            return super().__call__(x)
    
    def _check_is_padding_valid(self, padding):
        for (left_padding, right_padding), kernel_size in zip(
            padding, self.kernel_size
        ):
            checkify.check(
                jlax.max(left_padding, right_padding) <= kernel_size,
                "Paddings should be less than the size of the kernel. "
                "Padding ({}, {}) received for kernel size "
                "{}.",
                jnp.asarray(left_padding),
                jnp.asarray(right_padding),
                jnp.asarray(kernel_size)
            )


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

