import os
from conf.singleton_conf import SingletonConfig
from jax import random as jr, vmap, pmap, numpy as jnp, jit, value_and_grad, debug, devices
from functools import partial
from util.dataloaders import DATALOADERS
from networks.net_factory import net_factory
from environments.dp_params import DP_RL_Params
from networks.nets import MLP
import chex
import optax
import tqdm
import wandb
from util.logger import ExperimentLogger
from privacy.gdp_privacy import approx_to_gdp, vec_to_mu_schedule
from util.util import determine_optimal_num_devices
from typing import Tuple
from environments.dp import train_with_noise
from util.baselines import Baseline


def main():
    experiment_config = SingletonConfig.get_experiment_config_instance()
    sweep_config = SingletonConfig.get_sweep_config_instance()
    environment_config = SingletonConfig.get_environment_config_instance()
    wandb_config = SingletonConfig.get_wandb_config_instance()

    total_timesteps = experiment_config.total_timesteps
    env_prng_seed = experiment_config.env_prng_seed

    print("Starting...")

    # Initialize dataset
    X, y = DATALOADERS[experiment_config.dataset](experiment_config.dataset_poly_d)
    print(f"Dataset shape: {X.shape}, {y.shape}")

    # Initialize Policy model
    policy_input = jnp.ones((1,1))
    policy_batch_size = sweep_config.policy.batch_size
    policy_model = net_factory(
        input_shape=policy_input.shape,
        output_shape=(1, environment_config.max_steps_in_episode),
        conf=sweep_config.policy.network,
    )

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

    directories = [os.path.join(".", "logs", "0")]
    columns = ["step", "batch_idx", "loss", "accuracy", "actions", "losses", "accuracies"]
    large_columns = ["actions", "losses", "accuracies"]
    logger = ExperimentLogger(directories, columns, large_columns)

    epsilon = experiment_config.sweep.env.eps
    delta = experiment_config.sweep.env.delta

    mu = approx_to_gdp(epsilon, delta)
    p = experiment_config.sweep.env.batch_size / X.shape[0]  # Assuming MNIST dataset size
    T = environment_config.max_steps_in_episode

    print("Privacy parameters:")
    print(f"\t(epsilon, delta)-DP: ({epsilon}, {delta})")
    print(f"\tmu-GDP: {mu}")

    _, num_gpus = determine_optimal_num_devices(devices('gpu'), policy_batch_size)
    gpus = devices('gpu')[:num_gpus]


    @partial(value_and_grad, has_aux=True)
    def get_policy_loss(policy, policy_input, key) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
        """Calculate the policy loss."""
        # Ensure positive actions
        policy_output = vmap(policy)(policy_input)
        actions = vec_to_mu_schedule(policy_output, mu, p, T).squeeze()

        keys = jr.split(key, policy_batch_size)
        
        keys = keys.reshape(policy_batch_size // num_gpus, num_gpus, 2)
        vmapped_twn = pmap(train_with_noise, in_axes=(None, None, 0), devices=gpus)

        @partial(vmap, in_axes=(None, None, 0))
        def pmap_fn(actions, env_params, keys):
            return vmapped_twn(actions, env_params, keys)

        _, losses, accuracies = pmap_fn(actions, env_params, keys)
        final_losses = losses[:, :, -1]
        losses = losses.reshape(-1, T + 1)
        accuracies = accuracies.reshape(-1, T)

        return jnp.mean(final_losses), (losses, accuracies)

    optimizer = optax.adam(learning_rate=experiment_config.sweep.policy.lr.min)
    opt_state = optimizer.init(policy_model) # type: ignore

    iterator = tqdm.tqdm(
        range(total_timesteps),
        desc="Training Progress",
        total=total_timesteps
    )

    key = jr.PRNGKey(env_prng_seed)
    for timestep in iterator:
        # Generate random key for the current timestep
        key, _ = jr.split(key)

        # Get policy loss
        (loss, (losses, accuracies)), grads = get_policy_loss(policy_model, policy_input, key) # type: ignore

        # Update policy model
        updates, opt_state = optimizer.update(grads, opt_state, policy_model)  # type: ignore
        policy_model: MLP = optax.apply_updates(policy_model, updates) # type: ignore

        policy_out = vmap(policy_model)(policy_input)
        new_noise = vec_to_mu_schedule(policy_out, mu, p, T)[0] # type: ignore
        for i in range(losses.shape[0]):
            loss = losses[i, -1]
            accuracy = accuracies[i, -1]
            logger.log(0, {"step": timestep,
                            "batch_idx": i,
                            "loss": loss,
                            "accuracy" : accuracy,
                            "actions": new_noise,
                            "losses": losses[i, :],
                            "accuracies": accuracies[i, :],
                    })

        iterator.set_description(
            f"Training Progress - Loss: {loss:.4f}"
        )

    # Generate baseline if directed
    if experiment_config.sweep.with_baselines:
        baseline = Baseline(env_params, mu, 8)
        baseline.generate_baseline_data()
    else:
        baseline = None

    # Log to wandb, if enabled
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
        logger.create_plots(0, log_to_wandb=log_to_wandb, baseline=baseline)
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