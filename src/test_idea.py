from conf.singleton_conf import SingletonConfig
from jax import random as jr, vmap, numpy as jnp, jit, value_and_grad, grad, debug, nn as jnn, lax
from functools import partial
from networks.net_factory import net_factory
import chex
from typing import Tuple
import optax
import tqdm


def main():
    experiment_config = SingletonConfig.get_experiment_config_instance()
    sweep_config = SingletonConfig.get_sweep_config_instance()
    environment_config = SingletonConfig.get_environment_config_instance()
    wandb_config = SingletonConfig.get_wandb_config_instance()

    total_timesteps = experiment_config.total_timesteps
    env_prng_seed = experiment_config.env_prng_seed

    print("Starting...")

    # Initialize Policy model
    policy_input = jnp.zeros((1, 1))
    noises = jnp.asarray(1.0)

    @value_and_grad
    def mb_standin(x):
        """Return a noisy approximation of 'x^2'"""
        """Stand-in for mini-batch optimization"""
        return x**2

    @value_and_grad
    def get_policy_loss(actions, key) -> Tuple[chex.Array, chex.Array]:
        """Calculate the policy loss."""

        actions = jnn.softplus(actions)  # Ensure actions are positive

        key, _key = jr.split(key)
        x = jr.uniform(_key, minval=-1, maxval=1, shape=())

        key, _key = jr.split(key)
        y, x_grad = mb_standin(x)
        x_grad = x_grad + jr.normal(key, x_grad.shape) * actions
        x = x - 0.1 * x_grad
        
        return mb_standin(x)[0]


    # Initialize optimizer
    # optimizer = optax.sgd(learning_rate=experiment_config.sweep.policy.lr.min)
    # opt_state = optimizer.init(noises) # type: ignore

    iterator = tqdm.tqdm(
        range(total_timesteps),
        desc="Training Progress",
        total=total_timesteps
    )

    for timestep in iterator:
        # Generate random key for the current timestep
        key = jr.PRNGKey(env_prng_seed + timestep)

        # Get policy loss
        final_value, grads = get_policy_loss(noises, key) # type: ignore

        noises = noises - 0.1 * grads

        # Update policy model
        # updates, opt_state = optimizer.update(grads, opt_state, policy_model)  # type: ignore
        # policy_model = optax.apply_updates(policy_model, updates) # type: ignore

        iterator.set_description(
            f"Training Progress - Loss: {final_value:.4f}. Noises: {noises:.4f}"
        )



if __name__ == "__main__":
    main()