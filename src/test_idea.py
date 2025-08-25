from conf.singleton_conf import SingletonConfig
from jax import random as jr, vmap, numpy as jnp, jit, value_and_grad, grad, debug, nn as jnn, lax
from networks.net_factory import net_factory
import chex
from typing import Tuple
import optax
import tqdm
import os
from util.logger import ExperimentLogger
from privacy.gdp_privacy import approx_to_gdp, gdp_to_sigma, mu_to_poisson_subsampling_shedule
import wandb


def main():
    experiment_config = SingletonConfig.get_experiment_config_instance()
    sweep_config = SingletonConfig.get_sweep_config_instance()
    environment_config = SingletonConfig.get_environment_config_instance()
    wandb_config = SingletonConfig.get_wandb_config_instance()

    total_timesteps = experiment_config.total_timesteps
    env_prng_seed = experiment_config.env_prng_seed

    print("Starting...")

    # Initialize Policy model
    policy_input = jnp.ones((1,1))
    policy_batch_size = sweep_config.policy.batch_size
    policy_model = net_factory(
        input_shape=policy_input.shape,
        output_shape=(1, environment_config.max_steps_in_episode),
        conf=sweep_config.policy.network,
    )

    directories = [os.path.join(".", "logs", "0")]
    columns = ["step", "loss", "actions"]
    large_columns = ["actions"]
    logger = ExperimentLogger(directories, columns, large_columns)


    epsilon = experiment_config.sweep.env.eps
    delta = experiment_config.sweep.env.delta

    mu = approx_to_gdp(epsilon, delta)
    p = experiment_config.sweep.env.batch_size / 60000  # Assuming MNIST dataset size
    T = environment_config.max_steps_in_episode

    print("Privacy parameters:")
    print(f"\t(epsilon, delta)-DP: ({epsilon}, {delta})")
    print(f"\tmu-GDP: {mu}")


    def vec_to_simplex(x: chex.Array, order=None, axis=1) -> chex.Array:
        return x / jnp.linalg.norm(x, ord=1, keepdims=True)

    def positive_actions(x: chex.Array) -> chex.Array:
        return jnn.softplus(x)

    def simplex_to_noise_schedule(x: chex.Array) -> chex.Array:
        """Convert a simplex vector to a noise schedule."""
        mu_schedule = mu_to_poisson_subsampling_shedule(mu, x, p, T)
        return gdp_to_sigma(mu_schedule)

    def pipeline(x: chex.Array) -> chex.Array:
        x = positive_actions(x)
        x = vec_to_simplex(x)
        x = simplex_to_noise_schedule(x)
        return x

    @value_and_grad
    def mb_standin(x):
        """Return a noisy approximation of 'x^2'"""
        """Stand-in for mini-batch optimization"""
        return x**2

    def apply_actions(actions, x, key):
        """Applies a sequence of GD updates"""

        def loop_scan(carry, action):
            x, key = carry
            key, _key = jr.split(key)
            y, x_grad = mb_standin(x)
            x_grad = x_grad + jr.normal(_key, x_grad.shape) * action
            x = x - 0.1 * x_grad

            return (x, key), y
        
        (x, _), _ = lax.scan(loop_scan, (x, key), actions)
        
        return mb_standin(x)[0]

    @value_and_grad
    def get_policy_loss(policy, policy_input, key) -> Tuple[chex.Array, chex.Array]:
        """Calculate the policy loss."""
        # Ensure positive actions
        policy_output = vmap(policy)(policy_input)
        actions = vmap(pipeline)(policy_output).squeeze()

        key, _key = jr.split(key)
        keys = jr.split(key, policy_batch_size)
        initial_x = jr.uniform(_key, minval=-1, maxval=-1+1e-5, shape=(policy_batch_size,))

        return vmap(apply_actions, in_axes=(None, 0, 0))(actions, initial_x, keys).mean()

    # Initialize optimizer
    optimizer = optax.adam(learning_rate=experiment_config.sweep.policy.lr.min)
    opt_state = optimizer.init(policy_model) # type: ignore

    iterator = tqdm.tqdm(
        range(total_timesteps),
        desc="Training Progress",
        total=total_timesteps
    )

    for timestep in iterator:
        # Generate random key for the current timestep
        key = jr.PRNGKey(env_prng_seed + timestep)

        # Get policy loss
        loss, grads = get_policy_loss(policy_model, policy_input, key) # type: ignore

        # Update policy model
        updates, opt_state = optimizer.update(grads, opt_state, policy_model)  # type: ignore
        policy_model = optax.apply_updates(policy_model, updates) # type: ignore

        new_noise = pipeline(vmap(policy_model)(policy_input))[0] # type: ignore
        logger.log(0, {"step": timestep, "loss": loss, "actions": new_noise})

        iterator.set_description(
            f"Training Progress - Loss: {loss:.4f}"
        )

    # Log to wandb, if enabled
    if wandb_config.mode != "disabled":
        run_ids = []
        log_to_wandb = (wandb_config.mode != "disabled")
        with wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            config={
                "policy": sweep_config.policy.to_wandb(),
                "env": sweep_config.env.to_wandb(),
            },
            mode=wandb_config.mode,
        ) as run:
            run_ids.append(run.id)
            logger.create_plots(0, log_to_wandb=log_to_wandb, with_baselines=sweep_config.with_baselines)
            logger.get_csv(0, log_to_wandb=log_to_wandb)

        # sweep
        # print(run_ids)
        # wandb.sweep(
        #     sweep_config.to_wandb(),
        #     project=wandb_config.project,
        #     entity=wandb_config.entity,
        #     prior_runs=run_ids
        # )


if __name__ == "__main__":
    main()