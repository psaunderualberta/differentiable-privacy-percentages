import jax
import numpy as np
import optax
import tqdm
from jax import device_put, devices
from jax import random as jr
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

import wandb
from conf.scope import RunContext, current, using
from conf.singleton_conf import SingletonConfig
from environments.dp import get_private_model_training_schemas
from environments.dp_params import DPTrainingParams
from environments.outer_loop import make_training_loss_fn
from policy.factory import make_schedule
from privacy.gdp_privacy import get_privacy_params
from util.baselines import Baseline
from util.checkpointing import (
    load_checkpoint,
    make_state,
    save_checkpoint,
)
from util.dataloaders import get_dataset_shapes
from util.eval import make_dp_psac_ref_cmd
from util.job_chain import (
    register_signal_handler,
    resubmit_if_requested,
    shutdown_requested,
    time_limit_approaching,
)
from util.logger import Loggable, WandbTableLogger
from util.util import ensure_valid_pytree, get_optimal_mesh
from util.wandb_init import init_wandb_run


def main():
    """Run the outer gradient-based loop that learns the DP-SGD noise/clip schedule."""
    sweep_config = current().config.sweep
    wandb_config = current().config.wandb_conf

    num_outer_steps = sweep_config.num_outer_steps
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

    # Initialize schedule
    schedule_conf = current().config.sweep.schedule_optimizer.schedule
    schedule = make_schedule(schedule_conf, gdp_params)
    schedule = schedule.project()
    if getattr(schedule, "use_fista", False):
        schedule = schedule.fista_extrapolate()
    schedule_batch_size = sweep_config.schedule_optimizer.batch_size

    # Initialize private environment
    env_params = DPTrainingParams.create_direct_from_config()

    # Initialize logger
    logger = WandbTableLogger()
    for schema in schedule.get_logging_schemas() + get_private_model_training_schemas():
        logger.add_schema(schema)

    # Build mesh (for sharding noise_keys) and the JIT-compiled training loss function
    mesh = get_optimal_mesh(devices("gpu"), schedule_batch_size)
    get_training_loss = make_training_loss_fn(env_params)

    # Initialise optimizer and PRNG keys (may be overwritten by checkpoint restore)
    optimizer = optax.sgd(
        learning_rate=sweep_config.schedule_optimizer.lr.sample(),
        momentum=sweep_config.schedule_optimizer.momentum.sample(),
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
            # Orbax restores arrays as device-committed (bound to device 0).
            # This conflicts with sharded noise_keys inside the JIT call.
            # Round-trip through numpy to produce uncommitted arrays, matching
            # what a fresh (non-checkpoint) run provides.
            schedule = jax.tree_util.tree_map(
                lambda x: jax.numpy.array(np.asarray(x)) if isinstance(x, jax.Array) else x,
                schedule,
            )
            opt_state = jax.tree_util.tree_map(
                lambda x: jax.numpy.array(np.asarray(x)) if isinstance(x, jax.Array) else x,
                opt_state,
            )
            key = jax.numpy.array(np.asarray(key))
            init_key = jax.numpy.array(np.asarray(init_key))

    # --- W&B init ---
    print("Starting...")
    run = init_wandb_run(wandb_config, sweep_config)
    register_signal_handler()

    _dp_psac_ref_cmd = make_dp_psac_ref_cmd(
        run_id=run.id,
        entity=wandb_config.entity,
        project=wandb_config.project,
        dataset=sweep_config.dataset,
        batch_size=sweep_config.env.batch_size,
        lr=env_params.lr,
        delta=gdp_params.delta,
        arch=type(env_params.network).__name__.lower(),
    )
    print(f"dp_psac_ref eval command:\n  {_dp_psac_ref_cmd}")

    # --- Baseline setup ---
    eval_key = jr.PRNGKey(0)
    baseline = Baseline(env_params, gdp_params, num_reps=32)
    log_baselines_during_training = (
        sweep_config.with_baselines and sweep_config.baseline_log_interval > 0
    )
    baseline_data_saved = False
    if sweep_config.with_baselines:
        # On checkpoint restarts, reuse the baseline data from the source run
        if wandb_config.checkpoint_run_id is not None and baseline.restore_from_cache(
            wandb_config.checkpoint_run_id,
            wandb_config.entity,
            wandb_config.project,
        ):
            print("Reusing cached baseline data from checkpoint run.")
            baseline_data_saved = True  # already persisted under source run_id
        elif log_baselines_during_training:
            # Pre-generate now so periodic log_comparison calls work immediately.
            # End-only case (baseline_log_interval == 0) is deferred to log_comparison().
            baseline.generate_baseline_data(eval_key)
            baseline.save(run.id, run)
            baseline_data_saved = True

    iterator = tqdm.tqdm(
        range(start_step, num_outer_steps),
        desc="Training Progress",
        total=num_outer_steps - start_step,
    )
    sharding = NamedSharding(mesh, P("x", None))
    for t in iterator:
        key, _ = jr.split(key)

        # Log schedule parameters for this iteration
        for loggable_item in schedule.get_loggables():
            logger.log(loggable_item)

        # Compute training loss and gradients
        key, mb_key, noise_key = jr.split(key, 3)
        if not sweep_config.train_on_single_network:
            key, init_key = jr.split(key)

        # mb_key = device_put(mb_key, sharding)
        # init_key = device_put(init_key, sharding)
        noise_keys = device_put(jr.split(noise_key, schedule_batch_size), sharding)
        (loss, (losses, accuracies, val_accs)), grads = get_training_loss(
            schedule,
            mb_key,
            init_key,
            noise_keys,
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
        schedule = ensure_valid_pytree(schedule, "schedule in main after updates")
        x_new = schedule.project()
        x_new = ensure_valid_pytree(x_new, "schedule in main after project")
        if getattr(schedule, "use_fista", False):
            schedule = schedule.fista_advance(x_new)
            schedule = schedule.fista_extrapolate()
        else:
            schedule = x_new
        schedule = ensure_valid_pytree(schedule, "schedule in main after fista")

        iterator.set_description(f"Training Progress - Loss: {loss:.4f}")

        if (t + 1) % wandb_config.checkpoint_every == 0:
            save_checkpoint(make_state(schedule, opt_state, key, init_key, t), t, run)

        if log_baselines_during_training and (t + 1) % sweep_config.baseline_log_interval == 0:
            baseline.log_comparison(schedule, eval_key, logger=logger)

        if shutdown_requested() or time_limit_approaching():
            print(f"Graceful shutdown at step {t}; checkpointing for job-chain resubmit")
            save_checkpoint(make_state(schedule, opt_state, key, init_key, t), t, run)
            break

    # Resubmit job (dependency ensures won't start until current ends)
    if shutdown_requested() or time_limit_approaching():
        resubmit_if_requested(run.id)
    else:
        # Final logging
        for loggable_item in schedule.get_loggables(force=True):
            logger.log(loggable_item)

        if sweep_config.with_baselines:
            baseline.log_comparison(schedule, eval_key, logger=logger)

        for multi_line_table in schedule.get_logging_schemas():
            logger.line_plot(multi_line_table.table_name)
        for bulk_line_table in get_private_model_training_schemas():
            logger.bulk_line_plots(bulk_line_table.table_name)

    # End-only case: log_comparison() lazily generated baseline data above;
    # save it now so future restarts can skip the sweep.
    if sweep_config.with_baselines and not baseline_data_saved:
        baseline.save(run.id, run)

    logger.finish()
    print(f"dp_psac_ref eval command:\n  {_dp_psac_ref_cmd}")
    run.finish()


if __name__ == "__main__":
    with using(RunContext(SingletonConfig.get_instance())):
        main()
