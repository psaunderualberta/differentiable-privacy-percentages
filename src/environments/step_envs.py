from typing import Any, Dict, Optional, Tuple, Union

import chex
import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
import jax.debug as debug
import optax
from environments.dp_params import DP_RL_Params
from environments.dp_state import DP_RL_State
from jax import vmap

from privacy.privacy import PrivacyAccountantState
from util.util import (add_spherical_noise, dp_cce_loss_poisson, reinit_model,
                       subset_classification_accuracy)


class StepTaker(eqx.Module):
    def __init__(self, *args, **kwargs):
        pass

    def step_env(
        self,
        input_key: chex.PRNGKey,
        state: DP_RL_State,
        action: chex.Array,
        params: DP_RL_Params,
        private: Optional[bool] = True,
    ) -> Tuple[DP_RL_State, jnp.ndarray, Dict[Any, Any]]:
        raise NotImplementedError()

    def reset_env(self, key: chex.PRNGKey, params: DP_RL_Params) -> DP_RL_State:
        raise NotImplementedError()

    def is_terminal(
        self,
        state: DP_RL_State,
        params: DP_RL_Params,
        action: chex.Array
    ) -> jnp.ndarray:
        raise NotImplementedError()


class PrivateStepTaker(StepTaker):
    def step_env(
        self,
        input_key: chex.PRNGKey,
        state: DP_RL_State,
        action: chex.Array,
        params: DP_RL_Params,
        private: bool = True,
    ) -> Tuple[DP_RL_State, jnp.ndarray, Dict[Any, Any]]:
        """Perform single timestep state transition."""
        # Use 0 variance if non-private
        # Add action to privacy budget, if private, else keep the same
        action = lax.select(private, action, jnp.zeros_like(action))

        # # Update privacy accountant w/ action
        pa = params.privacy_accountant
        pas = state.privacy_accountant_state
        action, new_state = pa.get_correct_noise(pas, action, return_new_state=True)
        new_pas = PrivacyAccountantState(
            lax.select(private, new_state.moments, pas.moments)
        )

        # Add spherical noise to gradients
        input_key, _used_key = jr.split(input_key)
        noised_grads = add_spherical_noise(
            state.grads, action, _used_key, params.C, params.dummy_batch.shape[0]
        )

        # Add noisy gradients, update model and optimizer
        updates, new_opt_state = optax.adam(params.lr).update(
            noised_grads, state.opt_state, eqx.filter(state.model, eqx.is_array)
        )
        new_model = eqx.apply_updates(state.model, updates)

        # Subsample each with probability p
        input_key, _used_key = jr.split(input_key)
        new_loss, grads, average_grads = dp_cce_loss_poisson(
            new_model, params.X, params.y, _used_key, params.dummy_batch, params.C
        )
        input_key, _used_key = jr.split(input_key)
        accuracy = subset_classification_accuracy(
            new_model, params.X, params.y, 0.01, _used_key
        )

        # Create new state
        new_state = DP_RL_State(
            grads=grads,
            average_grads=average_grads,
            model=new_model,
            loss=new_loss,
            initial_accuracy=state.initial_accuracy,
            accuracy=accuracy,
            privacy_accountant_state=new_pas,
            time=state.time + 1,
            action=action.squeeze(),
            opt_state=new_opt_state,
        )

        # Determine if complete
        done = self.is_terminal(new_state, params, action)

        # Return observation, state, reward, done, and additional information
        return (
            lax.stop_gradient(new_state),
            done,
            {},
        )

    def reset_env(
        self,
        key: chex.PRNGKey,
        params: DP_RL_Params,
    ) -> DP_RL_State:
        """Reset environment state by sampling initial position."""

        # Create key
        key, _key = jr.split(key)

        # Create network
        network = reinit_model(params.network, _key)

        # Create grads
        key, _key = jr.split(key)
        loss, grads, average_grads = dp_cce_loss_poisson(
            network, params.X, params.y, _key, params.dummy_batch, params.C
        )
        reward = jnp.zeros_like(loss)

        key, _key = jr.split(key)
        accuracy = subset_classification_accuracy(
            network, params.X, params.y, 0.01, _key
        )

        # Create state
        state = DP_RL_State(
            grads=grads,
            average_grads=average_grads,
            model=network,
            loss=loss,
            initial_accuracy=accuracy,
            accuracy=accuracy,
            privacy_accountant_state=params.privacy_accountant.reset_state(),
            time=0,
            action=params.action.squeeze(),
            opt_state=optax.adam(params.lr).init(eqx.filter(network, eqx.is_array)),
        )

        return state

    def is_terminal(
        self,
        state: DP_RL_State,
        params: DP_RL_Params,
        action: chex.Array
    ) -> jnp.ndarray:
        """Check whether state is terminal."""
        return jnp.logical_or(
            state.time >= params.max_steps_in_episode,
            params.privacy_accountant.is_done(state.privacy_accountant_state),
        ).squeeze()



class NonPrivateStepTaker(StepTaker):
    def step_env(
        self,
        input_key: chex.PRNGKey,
        state: DP_RL_State,
        action: chex.Array,
        params: DP_RL_Params,
        private: bool = True,
    ) -> Tuple[DP_RL_State, jnp.ndarray, Dict[Any, Any]]:
        """Perform single timestep state transition."""
        # Update privacy accountant w/ action
        new_pas = state.privacy_accountant_state

        # Add spherical noise to gradients
        input_key, _used_key = jr.split(input_key)
        noised_grads = add_spherical_noise(
            state.grads, action, _used_key, params.C, params.dummy_batch.shape[0]
        )

        # Add noisy gradients, update model and optimizer
        updates, new_opt_state = optax.adam(params.lr).update(
            noised_grads, state.opt_state, eqx.filter(state.model, eqx.is_array)
        )
        new_model = eqx.apply_updates(state.model, updates)

        # Subsample each with probability p
        input_key, _used_key = jr.split(input_key)
        new_loss, grads, average_grads = dp_cce_loss_poisson(
            new_model, params.X, params.y, _used_key, params.dummy_batch, params.C
        )
        input_key, _used_key = jr.split(input_key)
        accuracy = subset_classification_accuracy(
            new_model, params.X, params.y, 0.01, _used_key
        )

        # Create new state
        new_state = DP_RL_State(
            grads=grads,
            average_grads=average_grads,
            model=new_model,
            loss=new_loss,
            initial_accuracy=state.initial_accuracy,
            accuracy=accuracy,
            privacy_accountant_state=new_pas,
            time=state.time + 1,
            action=action.squeeze(),
            opt_state=new_opt_state,
        )

        # Determine if complete
        done = self.is_terminal(new_state, params, action)

        # Return observation, state, reward, done, and additional information
        return (
            lax.stop_gradient(new_state),
            done,
            {},
        )

    def reset_env(
        self,
        key: chex.PRNGKey,
        params: DP_RL_Params,
    ) -> DP_RL_State:
        """Reset environment state by sampling initial position."""

        # Create key
        key, _key = jr.split(key)

        # Create network
        network = reinit_model(params.network, _key)

        # Create grads
        key, _key = jr.split(key)
        loss, grads, average_grads = dp_cce_loss_poisson(
            network, params.X, params.y, _key, params.dummy_batch, params.C
        )

        key, _key = jr.split(key)
        accuracy = subset_classification_accuracy(
            network, params.X, params.y, 0.01, _key
        )

        # Create state
        state = DP_RL_State(
            grads=grads,
            average_grads=average_grads,
            model=network,
            loss=loss,
            initial_accuracy=accuracy,
            accuracy=accuracy,
            privacy_accountant_state=params.privacy_accountant.reset_state(),
            time=0,
            action=params.action.squeeze(),
            opt_state=optax.adam(params.lr).init(eqx.filter(network, eqx.is_array)),
        )

        return state

    def is_terminal(
        self,
        state: DP_RL_State,
        params: DP_RL_Params,
        action: chex.Array
    ) -> jnp.ndarray:
        """Check whether state is terminal."""
        return state.time >= params.max_steps_in_episode


class AveragedRewardStepTaker(PrivateStepTaker):
    def step_env(
        self,
        key: chex.PRNGKey,
        orig_state: DP_RL_State,
        action: chex.Array,
        params: DP_RL_Params,
        private: bool = True,
    ) -> Tuple[DP_RL_State, jnp.ndarray, Dict[Any, Any]]:

        num_iters = 10
        key, vmap_keys = jr.split(key)
        keys = jr.split(vmap_keys, num_iters)
        vmapped_step = vmap(super().step_env, in_axes=(0, None, None, None, None, None))

        _, _, _ = vmapped_step(
            keys, orig_state, action, params, private
        )

        state, done, info = super().step_env(key, orig_state, action, params, private)
        return state, done, info


class Sticky_Loop_Carry(eqx.Module):
    num_steps: int
    key: chex.PRNGKey
    state: DP_RL_State
    done: bool


class StickyActionStepTaker(PrivateStepTaker):
    def step_env(
        self,
        key: chex.PRNGKey,
        orig_state: DP_RL_State,
        action: chex.Array,
        params: DP_RL_Params,
        private: bool = True,
    ) -> Tuple[DP_RL_State, jnp.ndarray, Dict[Any, Any]]:
        """Perform single timestep state transition."""

        def body_fn(carry: Sticky_Loop_Carry):
            new_key, key_ = jr.split(carry.key)
            state, done, info = super(StickyActionStepTaker, self).step_env(
                key_, carry.state, action, params, private,
            )
            return Sticky_Loop_Carry(
                state=state, key=new_key, done=done, num_steps=carry.num_steps + 1
            )

        carry = Sticky_Loop_Carry(
            state=orig_state,
            done=False,
            key=key,
            num_steps=0,
        )

        result = lax.while_loop(
            lambda carry: jnp.logical_and(
                carry.num_steps < 10, jnp.logical_not(carry.done)
            ),
            body_fn,
            carry,
        )

        return (
            lax.stop_gradient(result.state),
            result.done,
            {},
        )


class PrivacyPercentageStepTaker(PrivateStepTaker):
    """
    In this environment, the action is a vector of length 'max_steps_in_episode', in which
    each element is the percentage of privacy budget to spend at that step.
    The action is a vector of floats in the range [0, 1], and the cumulative sum is 1.0
    """

    def step_env(
        self,
        input_key: chex.PRNGKey,
        state: DP_RL_State,
        action: chex.Array,
        params: DP_RL_Params,
        private: bool = True,
    ) -> Tuple[DP_RL_State, jnp.ndarray, Dict[Any, Any]]:
        def scan_body(carry, eps):
            state, key, done, info = carry
            # get privacy for this step
            pas = state.privacy_accountant_state
            pa = params.privacy_accountant.replace(eps_bound=eps)

            # get according noise parameter for the privacy setting
            noise_action = pa.get_correct_noise(pas, 1e-10, return_new_state=False)

            # step the environment with this amount of noise
            key, _key = jr.split(key)
            state, done, info = super(PrivacyPercentageStepTaker, self).step_env(
                _key, state, noise_action, params, private
            )

            return (state, key, done, info), state.loss
    
        (final_state, _, done, info), _ = lax.scan(
            scan_body,
            init=(state, input_key, False, {}),
            xs=action,
            length=action.shape[0], # type: ignore
        )

        return (
            final_state,
            done,
            info,
        )


StepTakers = {
    "private": PrivateStepTaker,
    "non-private": NonPrivateStepTaker,
    "averaged-reward": AveragedRewardStepTaker,
    "sticky-actions": StickyActionStepTaker,
    "privacy-percentage": PrivacyPercentageStepTaker,
}
