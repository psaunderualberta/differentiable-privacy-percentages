import jax.numpy as jnp
import jax.random as jr
import plotly.graph_objects as go
from conf.singleton_conf import SingletonConfig
from util.dataloaders import DATALOADERS
from privacy.gdp_privacy import approx_to_gdp, weights_to_sigma_schedule, project_weights, sigma_schedule_to_weights
from environments.dp import train_with_noise
from environments.dp_params import DP_RL_Params
from networks.net_factory import net_factory
import os
import equinox as eqx



def main():
    sweep_config = SingletonConfig.get_sweep_config_instance()
    environment_config = SingletonConfig.get_environment_config_instance() 
    
    X, y = DATALOADERS[sweep_config.dataset](sweep_config.dataset_poly_d)
    print(f"Dataset shape: {X.shape}, {y.shape}")

    private_network_arch = net_factory(
        input_shape=X.shape,
        output_shape=y.shape,
        conf=sweep_config.env.network,
    )
    env_params = DP_RL_Params.create(
        environment_config,
        network_arch=private_network_arch,
        X=X,
        y=y,
    )

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

    num_grid_points_per_dim = 10
    sigmas = weights_to_sigma_schedule(policy, mu_tot, p, T).squeeze()  # type: ignore
    grid_points = jnp.linspace(0.1, 2 * sigmas.min(), num_grid_points_per_dim, dtype=jnp.float32)
    losses = []
    start_sigmas = []
    end_sigmas = []

    mb_key, init_key, noise_key = jr.split(jr.PRNGKey(sweep_config.env_prng_seed), 3)
    for start_i in range(num_grid_points_per_dim):
        for end_i in range(num_grid_points_per_dim):
            start_sigma = grid_points[start_i]
            end_sigma = grid_points[end_i]
            sigmas = jnp.linspace(
                start_sigma, end_sigma, T, dtype=jnp.float32
            )
            try:
                weights = sigma_schedule_to_weights(sigmas, mu_tot, p, T)
                weights = project_weights(weights, mu_tot, p, T)  # Project to valid mu-GDP schedule
                sigmas = weights_to_sigma_schedule(weights, mu_tot, p, T).squeeze()  # type: ignore
            except eqx.EquinoxRuntimeError:
                print(f"Invalid sigma schedule with start sigma {round(start_sigma, 3)} and end sigma {round(end_sigma, 3)}, skipping...")
                losses.append(jnp.nan)
                start_sigmas.append(start_sigma)
                end_sigmas.append(end_sigma)
                continue

            print(f"Training with start sigma {round(start_sigma, 3)}, end sigma {round(end_sigma, 3)}...")
            _, loss, _, _ = train_with_noise(sigmas, env_params, mb_key, init_key, noise_key)

            losses.append(loss)
            start_sigmas.append(start_sigma)
            end_sigmas.append(end_sigma)
    
    fig = go.Figure(
        data=go.Surface(
            z=jnp.array(losses).reshape(num_grid_points_per_dim, num_grid_points_per_dim),
            x=jnp.asarray(start_sigmas).reshape(num_grid_points_per_dim, num_grid_points_per_dim),
            y=jnp.asarray(end_sigmas).reshape(num_grid_points_per_dim, num_grid_points_per_dim),
        )
    )

    fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))

    fig.update_layout(
        title="Loss Landscape over Start and End Sigmas",
        scene=dict(
            xaxis_title="Start Sigma",
            yaxis_title="End Sigma",
            zaxis_title="Loss",
        ),
        width=500, height=500, autosize=False,
    )

    current_dir = os.getcwd()
    filepath = os.path.join(current_dir, "plots", "loss_landscape.pdf")
    fig.write_image(filepath)


if __name__ == "__main__":
    main()