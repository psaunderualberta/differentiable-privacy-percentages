import chex
import equinox as eqx
import optax


class DP_RL_State(eqx.Module):
    grads: chex.Array
    average_grads: chex.Array
    model: eqx.Module
    opt_state: optax.OptState
    loss: chex.Array
    initial_accuracy: chex.Array
    accuracy: chex.Array
    time: int
    action: chex.Array