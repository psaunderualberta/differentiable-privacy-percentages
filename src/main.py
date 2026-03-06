import optax
import tqdm
from jax import devices
from jax import random as jr

import wandb
from conf.singleton_conf import SingletonConfig
from environments.dp import get_private_model_training_schemas
from environments.dp_params import DP_RL_Params
from environments.outer_loop import make_policy_loss_fn
from policy.factory import policy_factory
from privacy.gdp_privacy import get_privacy_params
from util.baselines import Baseline
from util.checkpointing import load_checkpoint, make_state, save_checkpoint
from util.dataloaders import get_dataset_shapes
from util.logger import Loggable, WandbTableLogger
from util.util import ensure_valid_pytree, get_optimal_mesh
from util.wandb_init import init_wandb_run


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
    print("Privacy parameters:")
    print(f"\t(epsilon, delta)-DP: ({gdp_params.eps}, {gdp_params.delta})")
    print(f"\tmu-GDP: {gdp_params.mu}")

    # Initialize schedule (policy)
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

    # Build the JIT-compiled, shard-mapped policy loss function
    mesh = get_optimal_mesh(devices("gpu"), policy_batch_size)
    get_policy_loss = make_policy_loss_fn(mesh, env_params)

    # Initialise optimizer and PRNG keys (may be overwritten by checkpoint restore)
    optimizer = optax.sgd(
        learning_rate=sweep_config.policy.lr.sample(),
        momentum=sweep_config.policy.momentum.sample(),
    )
    opt_state = optimizer.init(schedule)  # type: ignore

    key = jr.PRNGKey(env_prng_seed)
    key, init_key, _ = jr.split(key, 3)

    # --- Checkpoint restore (happens before wandb.init so we know start_step) ---
    start_step = 0
    if wandb_config.checkpoint_run_id is not None:
        state_template = make_state(schedule, opt_state, key, init_key, 0)
        result = load_checkpoint(
            wandb_config.checkpoint_run_id,
            wandb_config.checkpoint_step,
            state_template,
            wandb_config.entity,
            wandb_config.project,
        )
        if result is not None:
            restored_state, start_step = result
            schedule = restored_state["schedule"]
            opt_state = restored_state["opt_state"]
            key = restored_state["key"]
            init_key = restored_state["init_key"]

    # --- W&B init ---
    print("Starting...")
    run = init_wandb_run(wandb_config, sweep_config)

    # --- Baseline setup ---
    eval_key = jr.PRNGKey(0)
    baseline = Baseline(env_params, gdp_params, num_reps=100)
    log_baselines_during_training = (
        sweep_config.with_baselines and sweep_config.baseline_log_interval > 0
    )
    if log_baselines_during_training:
        baseline.generate_baseline_data(eval_key)

    iterator = tqdm.tqdm(
        range(start_step, total_timesteps),
        desc="Training Progress",
        total=total_timesteps - start_step,
    )

    for t in iterator:
        key, _ = jr.split(key)

        # Log schedule parameters for this iteration
        for loggable_item in schedule.get_loggables():
            logger.log(loggable_item)

        # Compute policy loss and gradients
        key, mb_key, noise_key = jr.split(key, 3)
        if not sweep_config.train_on_single_network:
            key, init_key = jr.split(key)
        (loss, (losses, accuracies, val_accs)), grads = get_policy_loss(
            schedule,
            mb_key,
            init_key,
            jr.split(noise_key, policy_batch_size),
        )

        logger.log(Loggable(table_name="train_losses", data={"losses": losses}))
        logger.log(Loggable(table_name="accuracies", data={"accuracies": accuracies}))

        wandb.log(
            {
                "val-loss": loss,
                "val-accuracy": val_accs.mean(),
                "train-loss": losses[:, -1].mean(),
                "train-accuracies": accuracies[:, -1].mean(),
            },
        )

        # Validate and apply gradient update
        loss = ensure_valid_pytree(loss, "loss in main")
        grads = ensure_valid_pytree(grads, "grads in main")
        updates, opt_state = optimizer.update(grads, opt_state, schedule)
        schedule = schedule.apply_updates(updates)
        schedule = ensure_valid_pytree(schedule, "policy in main after updates")
        schedule = schedule.project()
        schedule = ensure_valid_pytree(schedule, "policy in main after project")

        iterator.set_description(f"Training Progress - Loss: {loss:.4f}")

        if (t + 1) % wandb_config.checkpoint_every == 0:
            save_checkpoint(make_state(schedule, opt_state, key, init_key, t), t, run)

        if log_baselines_during_training and (t + 1) % sweep_config.baseline_log_interval == 0:
            baseline.log_comparison(schedule, eval_key)

    # Final logging
    for loggable_item in schedule.get_loggables(force=True):
        logger.log(loggable_item)

    if sweep_config.with_baselines:
        baseline.log_comparison(schedule, eval_key)

    for multi_line_table in schedule.get_logging_schemas():
        logger.line_plot(multi_line_table.table_name)
    for bulk_line_table in get_private_model_training_schemas():
        logger.bulk_line_plots(bulk_line_table.table_name)

    logger.finish()
    run.finish()


if __name__ == "__main__":
    main()
