import chex
import equinox as eqx
import optax
from privacy.privacy import PrivacyAccountantState


class DP_RL_State(eqx.Module):
    grads: chex.Array
    average_grads: chex.Array
    model: eqx.Module
    opt_state: optax.OptState
    loss: chex.Array
    initial_accuracy: chex.Array
    accuracy: chex.Array
    privacy_accountant_state: PrivacyAccountantState
    time: int
    action: chex.Array