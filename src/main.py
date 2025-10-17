import os
from functools import partial

import chex
import equinox as eqx
import optax
import tqdm
from jax import devices

try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map

from jax import numpy as jnp
from jax import random as jr
from jax import lax as jlax
from jax.experimental import checkify
from jax.sharding import Mesh, PartitionSpec as P
from jaxtyping import Array, PRNGKeyArray

import wandb
from conf.singleton_conf import SingletonConfig
from environments.dp import train_with_noise
from environments.dp_params import DP_RL_Params
from networks.net_factory import net_factory
from privacy.gdp_privacy import (
    approx_to_gdp,
    project_weights,
    weights_to_sigma_schedule,
    weights_to_mu_schedule,
)
from util.baselines import Baseline
from util.dataloaders import DATALOADERS
from util.logger import WandbTableLogger
from util.util import determine_optimal_num_devices, ensure_valid_pytree


def main():
    sweep_config = SingletonConfig.get_sweep_config_instance()
    environment_config = SingletonConfig.get_environment_config_instance()
    wandb_config = SingletonConfig.get_wandb_config_instance()

    total_timesteps = sweep_config.total_timesteps
    env_prng_seed = sweep_config.env_prng_seed

    print("Starting...")
    run = wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        id=wandb_config.restart_run_id,
        mode=wandb_config.mode,
        resume="allow",
    )

    # Initialize dataset
    X, y = DATALOADERS[sweep_config.dataset](sweep_config.dataset_poly_d)
    print(f"Dataset shape: {X.shape}, {y.shape}")

    epsilon = sweep_config.env.eps
    delta = sweep_config.env.delta
    mu_tot = approx_to_gdp(epsilon, delta)
    p = sweep_config.env.batch_size / X.shape[0]  # Assuming MNIST dataset size
    T = environment_config.max_steps_in_episode
    print("Privacy parameters:")
    print(f"\t(epsilon, delta)-DP: ({epsilon}, {delta})")
    print(f"\tmu-GDP: {mu_tot}")

    # Initialize Policy model
    policy = project_weights(jnp.ones((T,), dtype=jnp.float32), mu_tot, p, T)
    policy_batch_size = sweep_config.policy.batch_size

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

    logger = WandbTableLogger(
        {
            "policy": ["step", *(str(step) for step in range(T))],
            "mu": ["step", *(str(step) for step in range(T))],
            "losses": ["step", "losses"],
            "accuracies": ["step", "accuracies"],
            "actions": ["step", *(str(step) for step in range(T))],
            "grads": ["step", *(str(step) for step in range(T))],
        },
        {
            "policy": total_timesteps // sweep_config.plotting_steps,
            "mu": total_timesteps // sweep_config.plotting_steps,
            "actions": total_timesteps // sweep_config.plotting_steps,
            "losses": total_timesteps // sweep_config.plotting_steps,
            "accuracies": total_timesteps // sweep_config.plotting_steps,
            "grads": total_timesteps // sweep_config.plotting_steps,
        },
    )

    _, num_gpus = determine_optimal_num_devices(devices("gpu"), policy_batch_size)
    mesh = Mesh(devices("gpu")[:num_gpus], "x")
    vmapped_train_with_noise = eqx.filter_vmap(
        train_with_noise, in_axes=(None, None, None, None, 0)
    )

    # @partial(checkify.checkify, errors=checkify.nan_checks)
    @eqx.filter_jit
    @partial(eqx.filter_value_and_grad, has_aux=True)
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P(), P(), P("x")),
        out_specs=(P(), (P("x"), P("x"))),
        check_vma=False,
    )
    def get_policy_loss(
        policy: Array,
        mb_key: PRNGKeyArray,
        init_key: PRNGKeyArray,
        noise_keys: PRNGKeyArray,
    ) -> tuple[chex.Array, tuple[chex.Array, chex.Array]]:
        """Calculate the policy loss."""

        sigmas = weights_to_sigma_schedule(policy, mu_tot, p, T).squeeze()

        # Ensure privacy loss on each iteration is a real number (i.e. \sigma > 0)
        sigmas = eqx.error_if(sigmas, jnp.any(sigmas <= 0), "Some sigmas are <= zero!")

        # Train all networks
        _, losses, accuracies = vmapped_train_with_noise(
            sigmas, env_params, mb_key, init_key, noise_keys
        )

        # Average over all shard-mapped networks
        final_loss = jnp.mean(losses[:, -1])
        final_loss = jlax.pmean(final_loss, "x").squeeze()

        # losses = losses.reshape(-1, T + 1)
        # accuracies = accuracies.reshape(-1, T)

        # derivatives = derivatives.reshape(-1, T)
        # # derivatives = derivatives * dsigma_dweight(sigmas, mu, p, T)
        return final_loss, (losses, accuracies)

    optimizer = optax.adamw(learning_rate=sweep_config.policy.lr)
    opt_state = optimizer.init(policy)  # type: ignore

    iterator = tqdm.tqdm(
        range(total_timesteps), desc="Training Progress", total=total_timesteps
    )

    key = jr.PRNGKey(env_prng_seed)
    try:
        for timestep in iterator:
            timestep_dict = {"step": timestep}
            # Generate random key for the current timestep
            key, _ = jr.split(key)

            # Log policy & sigmas for this iteration
            mu_sched = weights_to_mu_schedule(mu_tot, policy, p, T).squeeze()
            new_sigmas = weights_to_sigma_schedule(policy, mu_tot, p, T).squeeze()  # type: ignore
            _ = logger.log_array("policy", policy, timestep_dict, plot=True)
            _ = logger.log_array("mu", mu_sched, timestep_dict, plot=True)
            _ = logger.log_array("actions", new_sigmas, timestep_dict, plot=True)

            # Get policy loss
            key, init_key, mb_key, noise_key = jr.split(key, 4)
            (loss, (losses, accuracies)), grads = get_policy_loss(
                policy, mb_key, init_key, jr.split(noise_key, policy_batch_size)
            )

            # log grads
            _ = logger.log_array("grads", grads, timestep_dict, plot=True)

            # Log iteration results to file
            _ = logger.log("losses", timestep_dict | {"losses": losses})
            _ = logger.log("accuracies", timestep_dict | {"accuracies": accuracies})

            # Log metrics for monitoring run
            wandb.log({"loss": loss, "accuracy": accuracies[:, -1].mean()})

            # Ensure gradients are real numbers
            loss = ensure_valid_pytree(loss, "loss in main")
            grads = ensure_valid_pytree(grads, "grads in main")

            # Update policy
            updates, opt_state = optimizer.update(grads, opt_state, policy)
            policy = optax.apply_updates(policy, updates)
            assert isinstance(policy, jnp.ndarray), "Policy is not an array"
            policy = ensure_valid_pytree(policy, "policy in main")
            policy = project_weights(policy, mu_tot, p, T)

            # Get new sigmas, ensure still valid
            new_sigmas = weights_to_sigma_schedule(policy, mu_tot, p, T).squeeze()  # type: ignore
            new_sigmas = ensure_valid_pytree(new_sigmas, "sigmas in main")

            # self-explanatory
            iterator.set_description(f"Training Progress - Loss: {loss:.4f}")

    except Exception as e:
        mu_sched = weights_to_mu_schedule(mu_tot, policy, p, T).squeeze()
        _ = logger.log_array("policy", policy, timestep_dict, force=True, plot=True)
        _ = logger.log_array("mu", mu_sched, timestep_dict, force=True, plot=True)
        _ = logger.log_array(
            "actions", new_sigmas, timestep_dict, force=True, plot=True
        )
        _ = logger.log_array("grads", grads, timestep_dict, force=True, plot=True)

        print("WARNING: Error raised during training: ")
        print(e.args[0])

        if not isinstance(e, KeyboardInterrupt):
            raise e

    # Generate final results with lots of iterations
    sigmas = weights_to_sigma_schedule(policy, mu_tot, p, T).squeeze()  # type: ignore
    eval_num_iterations = 100
    eval_key = jr.PRNGKey(0)

    # Generate baseline if directed
    if sweep_config.with_baselines:
        baseline = Baseline(env_params, mu_tot, eval_num_iterations)
        _ = baseline.generate_baseline_data(eval_key)
        eval_df = baseline.generate_schedule_data(sigmas, "Learned Policy", eval_key)
        final_loss_fig = baseline.baseline_comparison_final_loss_plotter(eval_df)
        accuracy_fig = baseline.baseline_comparison_accuracy_plotter(eval_df)
        wandb.log(
            {
                "Baseline - Final Losses": final_loss_fig,
                "Baseline - Accuracy": accuracy_fig,
            }
        )

    # Cleanup, finish wandb run
    for multi_line_table_name in ["actions", "policy", "grads", "mu"]:
        logger.line_plot(multi_line_table_name)
    for bulk_line_table_name in ["losses", "accuracies"]:
        logger.bulk_line_plots(bulk_line_table_name)

    logger.finish()
    run.finish()


if __name__ == "__main__":
    main()
