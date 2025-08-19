import chex
from gymnax.environments import environment
import equinox as eqx
from typing import Union
import optax
from privacy.privacy import PrivacyAccountantState


class DP_RL_State(eqx.Module, environment.EnvState):
    grads: chex.Array
    average_grads: chex.Array
    model: eqx.Module
    opt_state: optax.OptState
    reward: Union[float, chex.Array]
    loss: Union[float, chex.Array]
    initial_accuracy: Union[float, chex.Array]
    accuracy: Union[float, chex.Array]
    privacy_accountant_state: PrivacyAccountantState
    time: int
    action: chex.Array