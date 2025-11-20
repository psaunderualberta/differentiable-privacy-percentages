import os
from functools import partial

import equinox as eqx
import optax
import tqdm
from jax import devices
from jax.experimental.shard_map import shard_map

from jax import numpy as jnp
from jax import random as jr
from jax import lax as jlax
from jax.sharding import Mesh, PartitionSpec as P
from jaxtyping import Array, PRNGKeyArray

import wandb
from conf.singleton_conf import SingletonConfig
from environments.dp import train_with_noise, lookahead_train_with_noise
from environments.dp_params import DP_RL_Params
from networks.net_factory import net_factory
from privacy.gdp_privacy import (
    approx_to_gdp,
    project_weights,
    weights_to_sigma_schedule,
    weights_to_mu_schedule,
    sigma_schedule_to_weights
)
from privacy.schedules import AbstractNoiseSchedule, LinearInterpSigmaNoiseSchedule, LinearInterpPolicyNoiseSchedule
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



    # Initialize dataset
    X, y = DATALOADERS[sweep_config.dataset](sweep_config.dataset_poly_d)
    X_test, y_test = DATALOADERS[sweep_config.dataset](sweep_config.dataset_poly_d, test=True)
    print(f"Dataset shape: {X.shape}, {y.shape}")
    print(f"Test Dataset shape: {X_test.shape}, {y_test.shape}")

    epsilon = sweep_config.env.eps
    delta = sweep_config.env.delta
    mu_tot = approx_to_gdp(epsilon, delta)
    p = sweep_config.env.batch_size / X.shape[0]  # Assuming MNIST dataset size
    T = environment_config.max_steps_in_episode
    print("Privacy parameters:")
    print(f"\t(epsilon, delta)-DP: ({epsilon}, {delta})")
    print(f"\tmu-GDP: {mu_tot}")

    # Initialize Policy model
    keypoints = jnp.arange(0, T + 1, step=50, dtype=jnp.int32)
    values = jnp.ones_like(keypoints, dtype=jnp.float32)
    policy = LinearInterpSigmaNoiseSchedule(keypoints=keypoints, values=values, T=T)
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
        valX=X_test,
        valy=y_test,
    )

    logger = WandbTableLogger(
        {
            "policy": ["step", *(str(step) for step in range(T))],
            "losses": ["step", "losses"],
            "accuracies": ["step", "accuracies"],
            "actions": ["step", *(str(step) for step in range(T))],
        },
        {
            "policy": total_timesteps // sweep_config.plotting_steps,
            "actions": total_timesteps // sweep_config.plotting_steps,
            "losses": total_timesteps // sweep_config.plotting_steps,
            "accuracies": total_timesteps // sweep_config.plotting_steps,
        },
    )

    _, num_gpus = determine_optimal_num_devices(devices("gpu"), policy_batch_size)
    mesh = Mesh(devices("gpu")[:num_gpus], "x")
    vmapped_train_with_noise = eqx.filter_vmap(
        train_with_noise, in_axes=(None, None, None, None, 0)
    )

    print("Starting...")
    run = wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        id=wandb_config.restart_run_id,
        mode=wandb_config.mode,
        resume="allow",
    )

    # @partial(checkify.checkify, errors=checkify.nan_checks)
    @eqx.filter_jit
    @partial(eqx.filter_value_and_grad, has_aux=True)
    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P(), P(), P(), P("x")),
        out_specs=(P(), (P("x"), P("x"), P("x"))),
        check_rep=False,
    )
    def get_policy_loss(
        schedule: AbstractNoiseSchedule,
        mb_key: PRNGKeyArray,
        init_key: PRNGKeyArray,
        noise_keys: PRNGKeyArray,
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        """Calculate the policy loss."""
        sigmas = schedule.get_private_sigmas(mu_tot, p, T)

        # Ensure privacy loss on each iteration is a real number (i.e. \sigma > 0)
        sigmas = eqx.error_if(sigmas, jnp.any(sigmas <= 0), "Some sigmas are <= zero!")
        sigmas = ensure_valid_pytree(sigmas, "sigmas in get_policy_loss")

        # Train all networks
        _, to_diff, losses, accuracies, val_acc = vmapped_train_with_noise(
            sigmas, env_params, mb_key, init_key, noise_keys
        )

        # Average over all shard-mapped networks
        to_diff = jnp.mean(to_diff)
        to_diff = jlax.pmean(to_diff, "x").squeeze()

        # losses = losses.reshape(-1, T + 1)
        # accuracies = accuracies.reshape(-1, T)

        # derivatives = derivatives.reshape(-1, T)
        # # derivatives = derivatives * dsigma_dweight(sigmas, mu, p, T)
        return to_diff, (losses, accuracies, val_acc)

    optimizer = optax.sgd(learning_rate=sweep_config.policy.lr.sample())
    opt_state = optimizer.init(policy)  # type: ignore

    iterator = tqdm.tqdm(
        range(total_timesteps), desc="Training Progress", total=total_timesteps
    )

    key = jr.PRNGKey(env_prng_seed)
    key, init_key, mb_key = jr.split(key, 3)
    try:
        for timestep in iterator:
            timestep_dict = {"step": timestep}
            # Generate random key for the current timestep
            key, _ = jr.split(key)

            # Log policy & sigmas for this iteration
            sigmas = policy.get_private_sigmas(mu_tot, p, T)
            schedule = policy.get_private_schedule(mu_tot, p, T)
            _ = logger.log_array("policy", schedule, timestep_dict, plot=True)
            _ = logger.log_array("actions", sigmas, timestep_dict, plot=True)

            # Get policy loss
            key, mb_key, noise_key = jr.split(key, 3)
            if not sweep_config.train_on_single_network:
                key, init_key = jr.split(key)
            (loss, (losses, accuracies, val_accs)), grads = get_policy_loss(
                policy, mb_key, init_key, jr.split(noise_key, policy_batch_size)
            )

            # Log iteration results to file
            _ = logger.log("losses", timestep_dict | {"losses": losses})
            _ = logger.log("accuracies", timestep_dict | {"accuracies": accuracies})

            # Log metrics for monitoring run
            wandb.log({"loss": loss, "accuracy": val_accs.mean()})

            # Ensure gradients are real numbers
            loss = ensure_valid_pytree(loss, "loss in main")

            # Update policy
            updates, opt_state = optimizer.update(grads, opt_state, policy)
            policy = eqx.apply_updates(policy, updates)

            # Ensure no Infs or NaNs were introduced
            policy = ensure_valid_pytree(policy, "policy in main")

            # self-explanatory
            iterator.set_description(f"Training Progress - Loss: {loss:.4f}")

    except Exception as e:
        sigmas = policy.get_private_sigmas(mu_tot, p, T)
        schedule = policy.get_private_schedule(mu_tot, p, T)
        _ = logger.log_array("policy", schedule, timestep_dict, force=True, plot=True)
        _ = logger.log_array(
            "actions", sigmas, timestep_dict, force=True, plot=True
        )

        print("WARNING: Error raised during training: ")
        print(e.args[0])

        if not isinstance(e, KeyboardInterrupt):
            raise e

    # Generate final results with lots of iterations
    eval_num_iterations = 100
    eval_key = jr.PRNGKey(0)

    # Generate baseline if directed
    if sweep_config.with_baselines:
        baseline = Baseline(env_params, mu_tot, eval_num_iterations)
        _ = baseline.generate_baseline_data(eval_key)
        sigmas = policy.get_private_sigmas(mu_tot, p, T)
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
    for multi_line_table_name in ["actions", "policy"]:
        logger.line_plot(multi_line_table_name)
    for bulk_line_table_name in ["losses", "accuracies"]:
        logger.bulk_line_plots(bulk_line_table_name)

    logger.finish()
    run.finish()


if __name__ == "__main__":
    main()
