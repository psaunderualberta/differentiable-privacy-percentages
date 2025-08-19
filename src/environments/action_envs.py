from typing import Optional

import equinox as eqx
from jax import numpy as jnp, nn as jnn, debug as debug
import chex
from environments.dp_params import DP_RL_Params
from environments.dp_state import DP_RL_State
from gymnax.environments import spaces


class ActionTaker(eqx.Module):
    def __init__(self, *args, **kwargs):
        pass

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        raise NotImplementedError()

    def action_space(self, params: Optional[DP_RL_Params] = None) -> spaces.Box:
        """Creates a continuous action space of the environment."""
        raise NotImplementedError()

    def __call__(self, state: DP_RL_State, action: chex.Array, key: chex.PRNGKey, params: DP_RL_Params) -> chex.Array:
        return action

class DiscreteActionTaker(ActionTaker):
    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def action_space(self, params: Optional[DP_RL_Params] = None) -> spaces.Discrete:
        """Creates a discrete action space of the environment."""
        return spaces.Discrete(jnp.floor(params.var_high - params.var_low)) # type: ignore


class ContinuousActionTaker(ActionTaker):
    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 1

    def action_space(self, params: DP_RL_Params) -> spaces.Box:
        """Creates a continuous action space of the environment."""

        return spaces.Box(
            low=params.var_low,
            high=params.var_high,
            shape=(1,),
            dtype=jnp.float32,
        )


class SquashedActionTaker(ContinuousActionTaker):
    def __call__(self, _: DP_RL_State, action: chex.Array, key: chex.PRNGKey, params: DP_RL_Params) -> chex.Array:
        """Map action (-inf, inf) -> (var_low, inf)"""
        
        # numerical stability
        softplus_action = jnp.where(20 <= action, action, jnn.softplus(action))
        return softplus_action + params.var_low
    

class ChangeActionTaker(ContinuousActionTaker):
    def __call__(self, state: DP_RL_State, action: chex.Array, key: chex.PRNGKey, params: DP_RL_Params) -> chex.Array:
        """Map action (-inf, inf) -> (var_low, inf)"""
        
        # numerical stability
        moved_action = state.action + action
        smoothed_action = jnp.where(20 <= moved_action, moved_action, jnn.softplus(moved_action)) 
        return smoothed_action + params.var_low
    

class PrivacyPercentageAction(ActionTaker):
    dummy_action: chex.Array
    def __init__(self, params: DP_RL_Params):
        self.dummy_action = jnp.zeros((params.max_steps_in_episode,), dtype=jnp.float32)
    
    @property
    def num_actions(self) -> int:
        return self.dummy_action.shape[0] # type: ignore

    def action_space(self, params: Optional[DP_RL_Params] = None) -> spaces.Box:
        return spaces.Box(low=0, high=1, shape=self.dummy_action.shape, dtype=jnp.float32)

    def __call__(self, state: DP_RL_State, action: chex.Array, key: chex.PRNGKey, params: DP_RL_Params) -> chex.Array:
        # convert action to cumulative sum of epsilons
        return jnp.where(action >= 20, action, jnn.softplus(action))


ActionTakers = {
    "discrete": DiscreteActionTaker,
    "continuous": ContinuousActionTaker,
    "squashed": SquashedActionTaker,
    "change": ChangeActionTaker,
    "privacy-percentage": PrivacyPercentageAction,
}
