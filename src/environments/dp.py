"""JAX Compatible  version of MountainCarContinuous-v0 OpenAI gym environment.


Source:
github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py
"""

from typing import Any, Dict, Optional, Tuple, Union

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from gymnax.environments import environment, spaces

from environments.action_envs import ActionTaker, ActionTakers
from environments.obs_envs import ObservationMaker, ObservationMakers
from environments.step_envs import StepTaker, StepTakers
from environments.dp_params import DP_RL_Params
from environments.dp_state import DP_RL_State


class DP_RL(environment.Environment[DP_RL_State, DP_RL_Params]):
    action_obj: ActionTaker
    obs_obj: ObservationMaker
    step_obj: StepTaker

    @property
    def default_params(self) -> None:
        # Default environment parameters
        return None

    def __init__(
        self, step_taker: str, action_taker: str, obs_maker: str, params: DP_RL_Params
    ):
        super().__init__()
        self.step_obj = StepTakers[step_taker](params=params)
        self.action_obj = ActionTakers[action_taker](params=params)
        self.obs_obj = ObservationMakers[obs_maker](params=params)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: DP_RL_State,
        action: chex.Array,
        params: DP_RL_Params,
        private: Optional[bool] = True,
    ) -> Tuple[chex.Array, DP_RL_State, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        key, _key = jax.random.split(key)
        action = self.action_obj(state, action, key=key, params=params)
        state, reward, done, info = self.step_obj.step_env(
            _key, state, action, params, private
        )
        return self.obs_obj.get_obs(state, params, key), state, reward, done, info

    def reset_env(
        self, key: chex.PRNGKey, params: DP_RL_Params
    ) -> Tuple[chex.Array, DP_RL_State]:
        key, _key = jax.random.split(key)
        state = self.step_obj.reset_env(_key, params)
        return self.obs_obj.get_obs(state, params, key), state

    def is_terminal(
        self,
        state: DP_RL_State,
        params: DP_RL_Params,
        action: Union[float, chex.Array] = jnp.inf,
    ) -> jnp.ndarray:
        """Check whether state is terminal."""
        return self.step_obj.is_terminal(state, params, action)

    @property
    def name(self) -> str:
        """Environment name."""
        return "DP_RL-v0"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.action_obj.num_actions

    def action_space(self, params: Optional[DP_RL_Params] = None) -> spaces.Box:
        """Action space of the environment."""
        return self.action_obj.action_space(params)

    def get_obs(self, state: DP_RL_State, params: DP_RL_Params, key: chex.PRNGKey) -> chex.Array:
        """Return observation from raw state class."""
        return self.obs_obj.get_obs(state, params, key)

    def observation_space(self, params: DP_RL_Params) -> spaces.Box | spaces.Discrete:
        """Observation space of the environment."""
        return self.obs_obj.observation_space(params)
