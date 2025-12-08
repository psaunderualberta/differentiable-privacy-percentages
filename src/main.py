from functools import partial

import equinox as eqx
import optax
import tqdm
from jax import devices
from jax import lax as jlax
from jax import numpy as jnp
from jax import random as jr
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PRNGKeyArray

import wandb
from conf.singleton_conf import SingletonConfig
from environments.dp import (
    get_private_model_training_schemas,
    train_with_noise,
)
from environments.dp_params import DP_RL_Params
from privacy.base_schedules import InterpolatedExponentialSchedule
from privacy.gdp_privacy import get_privacy_params
from privacy.schedules import (
    AbstractNoiseAndClipSchedule,
    PolicyAndClipSchedule,
    SigmaAndClipSchedule,
)
from util.baselines import Baseline
from util.dataloaders import get_dataset_shapes
from util.logger import Loggable, WandbTableLogger
from util.util import ensure_valid_pytree, get_optimal_mesh


def main():
    sweep_config = SingletonConfig.get_sweep_config_instance()
    wandb_config = SingletonConfig.get_wandb_config_instance()

    total_timesteps = sweep_config.total_timesteps
    env_prng_seed = sweep_config.env_prng_seed

    # Initialize dataset
    X_shape, y_shape, X_test_shape, y_test_shape = get_dataset_shapes()
    print(f"Dataset shape: {X_shape}, {y_shape}")
    print(f"Test Dataset shape: {X_test_shape}, {y_test_shape}")

    # Get privacy parameters
    gdp_params = get_privacy_params(X_shape[0])
    T = gdp_params.T
    mu_tot = gdp_params.mu
    print("Privacy parameters:")
    print(f"\t(epsilon, delta)-DP: ({gdp_params.eps}, {gdp_params.delta})")
    print(f"\tmu-GDP: {mu_tot}")

    # Initialize Policy model
    # TODO: Creatable via config
    keypoints = jnp.arange(0, T + 1, step=T // 50, dtype=jnp.int32)
    values = jnp.zeros_like(keypoints, dtype=jnp.float32)
    policy_schedule = InterpolatedExponentialSchedule(
        keypoints.copy(), values=values.copy(), T=T
    )
    clip_schedule = InterpolatedExponentialSchedule(
        keypoints=keypoints.copy(), values=values.copy(), T=T
    )
    schedule = PolicyAndClipSchedule(
        policy_schedule=policy_schedule,
        clip_schedule=clip_schedule,
        privacy_params=gdp_params,
    )
    schedule = schedule.__class__.project(schedule)
    policy_batch_size = sweep_config.policy.batch_size

    # Initialize private environment
    env_params = DP_RL_Params.create_direct_from_config()

    # Initialize logger
    logger = WandbTableLogger()
    for schema in schedule.get_logging_schemas() + get_private_model_training_schemas():
        logger.add_schema(schema)

    # Get mesh over accessible GPUs
    mesh = get_optimal_mesh(devices("gpu"), policy_batch_size)
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
        schedule: AbstractNoiseAndClipSchedule,
        mb_key: PRNGKeyArray,
        init_key: PRNGKeyArray,
        noise_keys: PRNGKeyArray,
    ) -> tuple[Array, tuple[Array, Array, Array]]:
        """Calculate the policy loss."""

        # Train all networks
        _, to_diff, losses, accuracies, val_acc = vmapped_train_with_noise(
            schedule, env_params, mb_key, init_key, noise_keys
        )

        # Average over all shard-mapped networks
        to_diff = jnp.mean(to_diff)
        to_diff = jlax.pmean(to_diff, "x").squeeze()
        return to_diff, (losses, accuracies, val_acc)

    optimizer = optax.sgd(learning_rate=sweep_config.policy.lr.sample(), momentum=0.5)
    opt_state = optimizer.init(schedule)  # type: ignore

    iterator = tqdm.tqdm(
        range(total_timesteps), desc="Training Progress", total=total_timesteps
    )

    key = jr.PRNGKey(env_prng_seed)
    key, init_key, mb_key = jr.split(key, 3)
    try:
        for _ in iterator:
            # Generate random key for the current timestep
            key, _ = jr.split(key)

            # Log policy & sigmas for this iteration
            for loggable_item in schedule.get_loggables():
                _ = logger.log(loggable_item)

            # Get policy loss
            key, mb_key, noise_key = jr.split(key, 3)
            if not sweep_config.train_on_single_network:
                key, init_key = jr.split(key)
            (loss, (losses, accuracies, val_accs)), grads = get_policy_loss(
                schedule, mb_key, init_key, jr.split(noise_key, policy_batch_size)
            )

            loggable_losses = Loggable(table_name="train_loss", data={"losses": losses})
            loggable_accuracies = Loggable(
                table_name="accuracy", data={"accuracies": accuracies}
            )
            loggable_accuracies = Loggable(
                table_name="accuracy", data={"accuracies": accuracies}
            )
            # Log iteration results to file
            _ = logger.log(loggable_losses)
            _ = logger.log(loggable_accuracies)

            # Log metrics for monitoring run
            wandb.log({"loss": loss, "accuracy": val_accs.mean()})

            # Ensure gradients are real numbers
            loss = ensure_valid_pytree(loss, "loss in main")

            # Update policy
            grads = ensure_valid_pytree(grads, "grads in main")
            updates, opt_state = optimizer.update(grads, opt_state, schedule)
            schedule = eqx.apply_updates(schedule, updates)

            # Project schedule back to valid space
            schedule = schedule.__class__.project(schedule)

            # Ensure no Infs or NaNs were introduced
            schedule = ensure_valid_pytree(schedule, "policy in main")

            # self-explanatory
            iterator.set_description(f"Training Progress - Loss: {loss:.4f}")

    except Exception as e:
        # print("WARNING: Error raised during training: ", e.args[0])

        # if not isinstance(e, KeyboardInterrupt):
        #     raise e
        raise e

    # Plot final learnables
    for loggable_item in schedule.get_loggables(force=True):
        _ = logger.log(loggable_item)

    # Generate final results with lots of iterations
    eval_num_iterations = 100
    eval_key = jr.PRNGKey(0)

    # Generate baseline if directed
    if sweep_config.with_baselines:
        baseline = Baseline(env_params, gdp_params, eval_num_iterations)
        _ = baseline.generate_baseline_data(eval_key)
        eval_df = baseline.generate_schedule_data(schedule, "Learned Policy", eval_key)
        final_loss_fig = baseline.baseline_comparison_final_loss_plotter(eval_df)
        accuracy_fig = baseline.baseline_comparison_accuracy_plotter(eval_df)
        wandb.log(
            {
                "Baseline - Final Losses": final_loss_fig,
                "Baseline - Accuracy": accuracy_fig,
            }
        )

    # Cleanup, finish wandb run
    for multi_line_table in schedule.get_logging_schemas():
        logger.line_plot(multi_line_table.table_name)
    for bulk_line_table in get_private_model_training_schemas():
        logger.bulk_line_plots(bulk_line_table.table_name)

    logger.finish()
    run.finish()


if __name__ == "__main__":
    main()
