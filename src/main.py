import os

import jax.numpy as jnp
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
from environments.outer_loop import make_initial_es_state, make_training_loss_fn
from policy.factory import make_schedule
from privacy.gdp_privacy import get_privacy_params
from util.baselines import Baseline
from util.dataloaders import get_dataset_shapes
from util.eval import make_dp_psac_ref_cmd
from util.grad_guard import ConsecutiveNonfiniteGuard
from util.logger import Loggable, WandbTableLogger
from util.run_lifecycle import RunLifecycle, TrainingState
from util.util import (
    ensure_valid_pytree,
    get_optimal_mesh,
    pytree_has_inf,
    pytree_has_nan,
)
from util.wandb_init import (
    init_wandb_run,
    start_offline_sync_daemon,
    sync_offline_run,
)


def main():
    """Run the outer gradient-based loop that learns the DP-SGD noise/clip schedule."""
    sweep_config = current().config.sweep
    wandb_config = current().config.wandb_conf

    # Seed the global numpy RNG before any DistributionConfig.sample()
    # (assuming not done above)
    np.random.seed(sweep_config.master_seed)

    num_outer_steps = sweep_config.num_outer_steps
    key = jr.PRNGKey(sweep_config.prng_seed.sample())

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
    # Determine the sharded outer-loop axis size:
    #   * Analytic mode → schedule_batch_size (independent vmapped DP-SGD runs)
    #   * ES mode       → population_size // 2 (one CRN key per antithetic pair)
    es_conf = sweep_config.schedule_optimizer.es
    if es_conf.enabled:
        population_size = int(es_conf.population_size.sample())
        if population_size % 2 != 0:
            raise ValueError(
                f"ES population_size must be even (antithetic pairs); got {population_size}.",
            )
        num_gpus = len(devices("gpu"))
        if population_size % (2 * num_gpus) != 0:
            raise ValueError(
                f"ES population_size ({population_size}) must be divisible by "
                f"2 * num_gpus ({2 * num_gpus}) so antithetic pairs split evenly.",
            )
        parallel_axis_size = population_size // 2
        print(f"ES enabled: population_size={population_size}, half_pop={parallel_axis_size}")
    else:
        parallel_axis_size = sweep_config.schedule_optimizer.batch_size

    # Initialize private environment
    env_params = DPTrainingParams.create_direct_from_config()

    # Initialize logger
    logger = WandbTableLogger()
    for schema in schedule.get_logging_schemas() + get_private_model_training_schemas():
        logger.add_schema(schema)

    # Build mesh (for sharding noise_keys) and the JIT-compiled training loss function
    mesh = get_optimal_mesh(devices("gpu"), parallel_axis_size)
    get_training_loss = make_training_loss_fn(env_params)
    es_state = make_initial_es_state()  # None unless ES enabled

    # Initialise optimizer and PRNG keys (may be overwritten by checkpoint restore)
    # Robustness chain (applied to the gradient in order):
    #   clip_by_global_norm -> zero_nans -> sgd
    # A rare divergent inner DP-SGD run yields a finite forward loss but an
    # Inf/NaN backward. clip_by_global_norm collapses any Inf/NaN into a NaN
    # global-norm scale (so the whole step is poisoned uniformly), then
    # zero_nans rewrites those NaNs to 0 *before* the momentum trace — turning
    # the corrupt step into a no-op instead of crashing the run. Finite steps
    # are simply clipped to max_grad_norm. zero_nans must precede sgd so the
    # momentum trace never ingests a NaN.
    optimizer = optax.chain(
        optax.clip_by_global_norm(sweep_config.schedule_optimizer.max_grad_norm),
        optax.zero_nans(),
        optax.sgd(
            learning_rate=sweep_config.schedule_optimizer.lr.sample(),
            momentum=sweep_config.schedule_optimizer.momentum.sample(),
        ),
    )
    opt_state = optimizer.init(schedule)  # type: ignore

    key, init_key, _ = jr.split(key, 3)

    # ---
    # Checkpoint restore (happens before wandb.init so we know start_step).
    # RunLifecycle owns the restore-or-start decision and the device-uncommit.
    # ---
    lifecycle = RunLifecycle()
    template = TrainingState(schedule, opt_state, key, init_key, es_state, jnp.array(0, jnp.int32))
    restored, start_step = lifecycle.restore(template)
    if restored is not None:
        schedule, opt_state = restored.schedule, restored.opt_state
        key, init_key, es_state = restored.key, restored.init_key, restored.es_state

    # ---
    # W&B init
    # ---
    print("Starting...")
    run = init_wandb_run(wandb_config, sweep_config)
    lifecycle.attach(run)

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

    # ---
    # Baseline setup
    # ---
    key, eval_key = jr.split(key)
    baseline = Baseline(env_params, gdp_params, eval_key)
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

    if log_baselines_during_training:
        baseline.log_comparison(schedule, eval_key, logger=logger)

    # ---
    # Main Training Loop
    # ---
    # Capture the run directory before finish(): run.dir is the files/ subdir,
    # and `wandb sync` wants its parent (the run directory itself).
    run_dir = os.path.dirname(run.dir)
    # Offline mode only: keep the cloud dashboard near-live by syncing in the
    # background.  No-op for online/disabled runs.
    sync_daemon = start_offline_sync_daemon(
        wandb_config.mode, run_dir, wandb_config.wandb_sync_interval_secs
    )

    iterator = tqdm.tqdm(
        range(start_step, num_outer_steps),
        desc="Training Progress",
        total=num_outer_steps - start_step,
    )
    nonfinite_guard = ConsecutiveNonfiniteGuard(
        sweep_config.schedule_optimizer.max_consecutive_nonfinite_grads
    )
    sharding = NamedSharding(mesh, P("x", None))
    try:
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
            noise_keys = device_put(jr.split(noise_key, parallel_axis_size), sharding)
            (loss, statistics), grads, es_state = get_training_loss(
                schedule,
                mb_key,
                init_key,
                noise_keys,
                es_state,
            )

            logger.log(Loggable(table_name="train_losses", data={"losses": statistics.losses}))
            logger.log(
                Loggable(table_name="accuracies", data={"accuracies": statistics.accuracies})
            )

            grad_nonfinite = bool(pytree_has_nan(grads) | pytree_has_inf(grads))
            wandb.log(
                {
                    "val-loss": loss,
                    "val-accuracy": statistics.val_accuracy.mean(),
                    "test-loss": statistics.test_loss.mean(),
                    "test-accuracy": statistics.test_accuracy.mean(),
                    "train-loss": statistics.losses[:, -1].mean(),
                    "train-accuracies": statistics.accuracies[:, -1].mean(),
                    # Pre-clip gradient norm — use this to calibrate
                    # schedule_optimizer.max_grad_norm. `grad-nonfinite` is 1.0 on
                    # steps where the clip_by_global_norm -> zero_nans chain had to
                    # neutralise an Inf/NaN backward (i.e. a divergent inner run).
                    "grad-global-norm": optax.global_norm(grads),
                    "grad-nonfinite": grad_nonfinite * 1.0,
                },
            )
            # Abort loudly if the gradient has been non-finite for too many
            # consecutive steps: the zero_nans chain below would otherwise no-op
            # every update, silently wasting the whole run (see ConsecutiveNonfiniteGuard).
            nonfinite_guard.update(grad_nonfinite)

            # Validate and apply gradient update. We intentionally do NOT raise on
            # non-finite grads here: the optimizer chain (clip_by_global_norm ->
            # zero_nans) rewrites a corrupt step into a no-op so a rare divergent
            # inner DP-SGD run can't kill a multi-hour outer run. The downstream
            # schedule checks still guard against anything the chain misses.
            loss = ensure_valid_pytree(loss, "loss in main")
            updates, opt_state = optimizer.update(grads, opt_state, schedule)
            schedule = schedule.apply_updates(updates)
            schedule = ensure_valid_pytree(schedule, "schedule in main after updates")
            x_new = schedule.project()
            x_new = ensure_valid_pytree(x_new, "schedule in main after project")
            schedule = x_new

            iterator.set_description(f"Training Progress - Loss: {loss:.4f}")

            if log_baselines_during_training and (t + 1) % sweep_config.baseline_log_interval == 0:
                baseline.log_comparison(schedule, eval_key, logger=logger)

            state = TrainingState(
                schedule, opt_state, key, init_key, es_state, jnp.array(t, jnp.int32)
            )
            if lifecycle.should_stop():
                print(f"Graceful shutdown at step {t}; checkpointing for job-chain resubmit")
                lifecycle.checkpoint(state, force=True)
                break
            lifecycle.checkpoint(state)

        # Final logging (skipped on graceful shutdown, which resubmits instead)
        if not lifecycle.stopped_for_chain:
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
    except KeyboardInterrupt:
        # Ctrl+C: don't let the run die mid-flight.  Fall through to the finally
        # block so the run is finished and any offline data is synced; the
        # interrupt latch suppresses the job-chain resubmit in finalize().
        lifecycle.mark_interrupted()
        print("\nKeyboardInterrupt — finishing run and syncing offline data before exit...")
    finally:
        # Stop the background syncer before the final sync so the two don't run
        # `wandb sync` on the same dir concurrently.
        if sync_daemon is not None:
            sync_daemon.stop()

        logger.finish()
        # Offline runs never touched the network during training (so they can't
        # be marked "crashed" mid-run); push the buffered data to the cloud now,
        # while we're still inside the SLURM allocation and SLURM_TMPDIR exists.
        if sync_daemon is not None:
            sync_offline_run(wandb_config.mode, run_dir)

        print(f"dp_psac_ref eval command:\n  {_dp_psac_ref_cmd}")
        run.finish()

    # Resubmit the continuation job iff a job-chain stop was latched (no-op for
    # KeyboardInterrupt / normal completion).  Runs after teardown so the
    # dependency-chained successor won't start until this run has finished.
    lifecycle.finalize()


if __name__ == "__main__":
    with using(RunContext(SingletonConfig.get_instance())):
        main()
