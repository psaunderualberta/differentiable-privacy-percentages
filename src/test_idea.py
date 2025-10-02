import jax.numpy as jnp
import jax.lax as jlax


def main():
    padding = 0
    x_side_len = 28
    x = jnp.arange(x_side_len * x_side_len).reshape(x_side_len, x_side_len)
    x = jnp.pad(x, ((padding, padding), (padding, padding)), "constant")
    ksize = 3
    k = jnp.zeros((ksize, ksize))

    arange = jnp.arange(x.size).reshape(*x.shape)
    slice_lengths = (x.shape[0]-ksize+1, x.shape[1]-ksize+1)
    def f(carry, idxs):
        return carry, jlax.dynamic_slice(arange, idxs, slice_lengths).reshape(-1)

    _, idxs = jlax.scan(
        f,
        None,
        xs=jnp.indices(k.shape).reshape(2, -1).T
    )


    flat_image = x.reshape(-1)
    new_side_len = x_side_len + 2 * padding - k.shape[0] + 1
    pooled = jnp.max(flat_image[idxs], axis=0, keepdims=True).reshape(new_side_len, new_side_len)



if __name__ == "__main__":
    main()