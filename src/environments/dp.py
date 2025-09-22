"""JAX Compatible  version of MountainCarContinuous-v0 OpenAI gym environment.


Source:
github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py
"""

from typing import Any, Dict, Optional, Tuple, Union
from functools import partial

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from gymnax.environments import spaces
import optax
from util.util import reinit_model, dp_cce_loss_poisson, get_spherical_noise, subset_classification_accuracy

from environments.action_envs import ActionTaker, ActionTakers
from environments.obs_envs import ObservationMaker, ObservationMakers
from environments.step_envs import StepTaker, StepTakers
from environments.dp_params import DP_RL_Params
from environments.dp_state import DP_RL_State


class DP_RL(eqx.Module):
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

    @partial(jax.jit, static_argnames=("return_action"))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: DP_RL_State,
        action: chex.Array,
        params: DP_RL_Params,
        private: Optional[bool] = True,
        return_action: bool = True,
    ) -> Tuple[chex.Array, DP_RL_State, jnp.ndarray, Dict[Any, Any]] \
        | Tuple[chex.Array, DP_RL_State, jnp.ndarray, Dict[Any, Any], chex.Array]:
        key, _key = jax.random.split(key)
        action = self.action_obj(state, action, key=key, params=params)
        state, done, info = self.step_obj.step_env(
            _key, state, action, params, private
        )
        if not return_action:
            return self.obs_obj.get_obs(state, params, key), state, done, info
        
        return self.obs_obj.get_obs(state, params, key), state, done, info, action

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
        action: chex.Array,
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


@jax.jit
def train_with_noise(noise_schedule: chex.Array, params: DP_RL_Params, key: chex.PRNGKey) -> Tuple[
    eqx.Module, eqx.Module, chex.Array, chex.Array
]:
    # Create key
    key, _key = jr.split(key)

    # Create network
    network = reinit_model(params.network, _key)
    optimizer = optax.sgd(params.lr)

    @jax.checkpoint
    def training_step(carry, noise) -> Tuple[
        Tuple[eqx.Module, optax.OptState, chex.PRNGKey],
    Tuple[chex.Array, chex.Array]]:
        model, opt_state, loop_key = carry

        loop_key, _key = jr.split(loop_key)
        new_loss, grads = dp_cce_loss_poisson(
            model, params.X, params.y, _key, params.dummy_batch, params.C
        )

        # Add spherical noise to gradients
        loop_key, _used_key = jr.split(loop_key)
        noises = get_spherical_noise(grads, noise, _used_key)
        noised_grads = eqx.apply_updates(grads, noises)

        # Add noisy gradients, update model and optimizer
        updates, new_opt_state = optimizer.update(
            noised_grads, opt_state, model
        )
        new_model = eqx.apply_updates(model, updates)

        # Subsample each with probability p
        loop_key, _used_key = jr.split(loop_key)
        accuracy = subset_classification_accuracy(
            new_model, params.X, params.y, 0.01, _used_key
        )

        return (new_model, new_opt_state, loop_key), (new_loss, accuracy)

    initial_carry = (network, optimizer.init(network), key)
    (network, _, loop_key), (losses, accuracies) = jax.lax.scan(
        training_step,
        initial_carry,
        xs=noise_schedule,
    )

    final_loss, _ = dp_cce_loss_poisson(
        network, params.X, params.y, loop_key, params.dummy_batch, params.C
    )

    losses = jnp.concat([losses, jnp.asarray([final_loss])])

    return network, losses, accuracies
    