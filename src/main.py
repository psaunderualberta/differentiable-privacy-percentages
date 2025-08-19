from conf.singleton_conf import SingletonConfig
from jax import random as jr, vmap, numpy as jnp, jit, value_and_grad, debug
from functools import partial
from util.dataloaders import DATALOADERS
from networks.net_factory import net_factory
from networks.nets import MLP
from environments.dp import DP_RL
from environments.dp_params import DP_RL_Params
from environments.dp_state import DP_RL_State
import chex
from typing import Tuple, Dict, Any


def main():
    experiment_config = SingletonConfig.get_experiment_config_instance()
    sweep_config = SingletonConfig.get_sweep_config_instance()
    environment_config = SingletonConfig.get_environment_config_instance()
    wandb_config = SingletonConfig.get_wandb_config_instance()

    total_timesteps = experiment_config.total_timesteps
    num_configs = experiment_config.num_configs
    cfg_prng_seed = experiment_config.cfg_prng_seed
    env_prng_seed = experiment_config.env_prng_seed

    print("Starting...")

    # Initialize dataset
    X, y = DATALOADERS[experiment_config.dataset](experiment_config.dataset_poly_d)
    print(f"Dataset shape: {X.shape}, {y.shape}")

    # Initialize Policy model
    policy_input = jnp.zeros((1, 1))
    policy_model = net_factory(
        input_shape=policy_input.shape,
        output_shape=(1, environment_config.max_steps_in_episode),
        conf=sweep_config.policy.network,
    )

    private_network_arch = net_factory(
        input_shape=X.shape,
        output_shape=y.shape,
        conf=sweep_config.env.network,
    )

    # Initialize private environment
    env_params = DP_RL_Params.create(
        environment_config,
        network_arch=private_network_arch,
        X=X,
        y=y,
    )

    env = DP_RL(
        step_taker=environment_config.step_taker,
        action_taker=environment_config.action_taker,
        obs_maker=environment_config.obs_maker,
        params=env_params,
    )

    @partial(value_and_grad, has_aux=True)
    def get_policy_loss(policy, policy_input, state, key) -> Tuple[chex.Array, Tuple[DP_RL_State, chex.Array]]:
        """Calculate the policy loss."""
        policy_output = vmap(policy)(policy_input)

        _, final_state, _, _, actions = env.step_env(
            key, state, policy_output, env_params, return_action=True
        )

        return final_state.loss, (final_state, actions)


    for timestep in range(total_timesteps):
        # Generate random key for the current timestep
        key = jr.PRNGKey(env_prng_seed + timestep)

        # Reset environment
        _, state = env.reset_env(key, env_params)

        # Get policy loss
        (loss, (final_state, actions)), grads = get_policy_loss(policy_model, policy_input, state, key) # type: ignore

        print(loss)
        print(grads)
        print(policy_model)
        print(actions)  # type: ignore
        # print(final_state)

        exit()

        # update policy
        
        
    


if __name__ == "__main__":
    main()