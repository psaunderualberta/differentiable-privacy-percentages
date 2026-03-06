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
from policy.factory import policy_factory
from policy.schedules.abstract import AbstractNoiseAndClipSchedule
from privacy.gdp_privacy import get_privacy_params
from util.baselines import Baseline
from util.checkpointing import load_checkpoint, make_state, save_checkpoint
from util.dataloaders import get_dataset_shapes
from util.logger import Loggable, WandbTableLogger
from util.util import ensure_valid_pytree, get_optimal_mesh


def main():
    """Run the outer gradient-based RL loop that learns the DP-SGD noise/clip schedule."""
    sweep_config = SingletonConfig.get_sweep_config_instance()
    wandb_config = SingletonConfig.get_wandb_config_instance()

    total_timesteps = sweep_config.total_timesteps
    env_prng_seed = sweep_config.prng_seed.sample()

    # Initialize dataset
    X_shape, y_shape, X_test_shape, y_test_shape = get_dataset_shapes()
    print(f"Dataset shape: {X_shape}, {y_shape}")
    print(f"Test Dataset shape: {X_test_shape}, {y_test_shape}")

    # Get privacy parameters
    gdp_params = get_privacy_params(X_shape[0])
    mu_tot = gdp_params.mu
    print("Privacy parameters:")
    print(f"\t(epsilon, delta)-DP: ({gdp_params.eps}, {gdp_params.delta})")
    print(f"\tmu-GDP: {mu_tot}")

    # Initialize Policy model
    schedule_conf = SingletonConfig.get_policy_config_instance().schedule
    schedule = policy_factory(schedule_conf, gdp_params)
    schedule = schedule.project()
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
        train_with_noise,
        in_axes=(None, None, None, None, 0),
    )

    # Initialise optimizer and PRNG keys.  These may be overwritten below if a
    # checkpoint is restored.
    optimizer = optax.sgd(
        learning_rate=sweep_config.policy.lr.sample(),
        momentum=sweep_config.policy.momentum.sample(),
    )
    opt_state = optimizer.init(schedule)  # type: ignore

    key = jr.PRNGKey(env_prng_seed)
    key, init_key, mb_key = jr.split(key, 3)

    # --- Checkpoint restore (happens before wandb.init so we know start_step) ---
    start_step = 0
    if wandb_config.checkpoint_run_id is not None:
        state_template = make_state(schedule, opt_state, key, init_key, 0)
        result = load_checkpoint(
            wandb_config.checkpoint_run_id,
            wandb_config.checkpoint_step,
            state_template,
            wandb_config.entity,
            wandb_config.project,  # always look in the original project
        )
        if result is not None:
            restored_state, start_step = result
            schedule = restored_state["schedule"]
            opt_state = restored_state["opt_state"]
            key = restored_state["key"]
            init_key = restored_state["init_key"]

    # --- W&B init ---
    # Three cases:
    #   1. Branching from a specific historical step → new run in {project}-branched
    #   2. Resuming the latest checkpoint of an existing run → continue that run
    #   3. Fresh start → new run
    print("Starting...")
    is_branching = (
        wandb_config.checkpoint_run_id is not None and wandb_config.checkpoint_step is not None
    )

    if is_branching:
        branch_project = (wandb_config.project or "runs") + "-branched"
        notes = (
            f"Branched from run {wandb_config.checkpoint_run_id} "
            f"at step {wandb_config.checkpoint_step} "
            f"(original project: {wandb_config.project})"
        )
        run = wandb.init(
            project=branch_project,
            entity=wandb_config.entity,
            mode=wandb_config.mode,
            config=sweep_config.to_wandb_sweep(),
            notes=notes,
        )
    elif wandb_config.restart_run_id is None:
        run = wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            id=wandb_config.restart_run_id,
            mode=wandb_config.mode,
            config=sweep_config.to_wandb_sweep(),
            resume="allow",
        )
    else:
        # Don't overwrite config when resuming an existing run
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
            schedule,
            env_params,
            mb_key,
            init_key,
            noise_keys,
        )

        # Average over all shard-mapped networks
        to_diff = jnp.mean(to_diff)
        to_diff = jlax.pmean(to_diff, "x").squeeze()
        return to_diff, (losses, accuracies, val_acc)

    # --- Baseline setup ---
    eval_num_iterations = 100
    eval_key = jr.PRNGKey(0)
    baseline = Baseline(env_params, gdp_params, eval_num_iterations)
    log_baselines_during_training = (
        sweep_config.with_baselines and sweep_config.baseline_log_interval > 0
    )
    if log_baselines_during_training:
        _ = baseline.generate_baseline_data(eval_key)

    def _log_baseline_comparison(schedule):
        eval_df = baseline.generate_schedule_data(schedule, "Learned Policy", eval_key)
        final_loss_fig = baseline.baseline_comparison_final_loss_plotter(eval_df)
        accuracy_fig = baseline.baseline_comparison_accuracy_plotter(eval_df)
        wandb.log(
            {
                "Baseline vs. Losses": final_loss_fig,
                "Baseline vs. Accuracy": accuracy_fig,
            },
        )
        baseline.delete_non_baseline_data()

    iterator = tqdm.tqdm(
        range(start_step, total_timesteps),
        desc="Training Progress",
        total=total_timesteps - start_step,
    )

    try:
        for t in iterator:
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
                schedule,
                mb_key,
                init_key,
                jr.split(noise_key, policy_batch_size),
            )

            loggable_losses = Loggable(
                table_name="train_losses",
                data={"losses": losses},
            )
            loggable_accuracies = Loggable(
                table_name="accuracies",
                data={"accuracies": accuracies},
            )
            # Log iteration results to file
            _ = logger.log(loggable_losses)
            _ = logger.log(loggable_accuracies)

            # Log metrics for monitoring run
            wandb.log(
                {
                    "val-loss": loss,
                    "val-accuracy": val_accs.mean(),
                    "train-loss": losses[:, -1].mean(),
                    "train-accuracies": accuracies[:, -1].mean(),
                },
            )

            # Ensure gradients are real numbers
            loss = ensure_valid_pytree(loss, "loss in main")

            # Update policy
            grads = ensure_valid_pytree(grads, "grads in main")
            updates, opt_state = optimizer.update(grads, opt_state, schedule)
            schedule = schedule.apply_updates(updates)

            schedule = ensure_valid_pytree(schedule, "policy in main after updates")

            # Project schedule back to valid space
            schedule = schedule.project()

            # Ensure no Infs or NaNs were introduced
            schedule = ensure_valid_pytree(schedule, "policy in main after project")

            iterator.set_description(f"Training Progress - Loss: {loss:.4f}")

            # Save checkpoint every N steps (state captured after project())
            if (t + 1) % wandb_config.checkpoint_every == 0:
                save_checkpoint(make_state(schedule, opt_state, key, init_key, t), t, run)

            # Log baseline comparison at regular intervals during training
            if log_baselines_during_training and (t + 1) % sweep_config.baseline_log_interval == 0:
                _log_baseline_comparison(schedule)

    except Exception as e:
        raise e

    # Plot final learnables
    for loggable_item in schedule.get_loggables(force=True):
        _ = logger.log(loggable_item)

    # Generate final baseline comparison if directed
    if sweep_config.with_baselines:
        if baseline is None:
            # baseline_log_interval == 0: build baseline data now (old behaviour)
            baseline = Baseline(env_params, gdp_params, eval_num_iterations)
            _ = baseline.generate_baseline_data(eval_key)
        else:
            # Discard any accumulated learned-policy rows from mid-training logs
            baseline.delete_non_baseline_data()
        eval_df = baseline.generate_schedule_data(schedule, "Learned Policy", eval_key)
        final_loss_fig = baseline.baseline_comparison_final_loss_plotter(eval_df)
        accuracy_fig = baseline.baseline_comparison_accuracy_plotter(eval_df)
        wandb.log(
            {
                "Baseline vs. Losses": final_loss_fig,
                "Baseline vs. Accuracy": accuracy_fig,
            },
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
