from conf.singleton_conf import SingletonConfig
from jax import random as jr, vmap, numpy as jnp, jit, value_and_grad
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
    policy_model = MLP.from_config(experiment_config.policy_model_config)

    network_arch = net_factory(
        input_shape=X.shape,
        output_shape=y.shape,
        conf=experiment_config.private_model_config,
    )

    # Initialize private environment
    env_params = DP_RL_Params.create(
        environment_config,
        network_arch=network_arch,
        X=X,
        y=y,
    )

    env = DP_RL(
        step_taker=environment_config.step_taker,
        action_taker=environment_config.action_taker,
        obs_maker=environment_config.obs_maker,
        params=env_params,
    )

    policy_input = jnp.zeros((1, env_params.input_dim))

    @partial(value_and_grad, has_aux=True)
    def get_policy_loss(policy, state, key) -> Tuple[chex.Array, Tuple[chex.Array, DP_RL_State, jnp.ndarray, Dict[Any, Any]]]:
        """Calculate the policy loss."""
        policy_output = vmap(policy)(policy_input)
        
        obs, state, reward, done, info = env.step_env(
            key, state, policy_output, env_params
        )

        return reward, (obs, state, done, info)


    for timestep in range(total_timesteps):
        # Generate random key for the current timestep
        key = jr.PRNGKey(env_prng_seed + timestep)

        # Reset environment
        obs, state = env.reset_env(key, env_params)

        # Get policy loss
        policy_grads, loss, (final_obs, state, done, info) = get_policy_loss(policy_model, state, key) # type: ignore

        exit()

        # update policy
        
    


if __name__ == "__main__":
    main()