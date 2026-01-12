import os

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import plotly.graph_objects as go
from privacy.schedules import (
    LinearInterpPolicyNoiseSchedule,
    LinearInterpSigmaNoiseSchedule,
)

from conf.singleton_conf import SingletonConfig
from environments.dp import train_with_noise
from environments.dp_params import DP_RL_Params
from networks.net_factory import net_factory
from privacy.gdp_privacy import (
    approx_to_gdp,
    project_weights,
    sigma_schedule_to_weights,
    weights_to_sigma_schedule,
)
from util.dataloaders import DATALOADERS


def main():
    sweep_config = SingletonConfig.get_sweep_config_instance()
    environment_config = SingletonConfig.get_environment_config_instance()

    X, y = DATALOADERS[sweep_config.dataset](sweep_config.dataset_poly_d)
    X_test, y_test = DATALOADERS[sweep_config.dataset](
        sweep_config.dataset_poly_d, test=True
    )
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
        valX=X_test,
        valy=y_test,
    )

    epsilon = sweep_config.env.eps
    delta = sweep_config.env.delta
    mu_tot = approx_to_gdp(epsilon, delta)
    p = sweep_config.env.batch_size / X.shape[0]  # Assuming MNIST dataset size
    T = environment_config.max_steps_in_episode
    print("Privacy parameters:")
    print(f"\t(epsilon, delta)-DP: ({epsilon}, {delta})")
    print(f"\tmu-GDP: {mu_tot}")

    num_grid_points_per_dim = 7
    grid_points = jnp.linspace(0, 3, num_grid_points_per_dim, dtype=jnp.float32)
    keypoints = jnp.linspace(0, T, num_grid_points_per_dim + 2, dtype=jnp.int32)
    base_sigma = 1 / T

    mb_key, init_key, noise_key = jr.split(
        jr.PRNGKey(sweep_config.prng_seed.sample()), 3
    )
    mb_keys = jr.split(mb_key, sweep_config.policy.batch_size)
    init_keys = jr.split(init_key, sweep_config.policy.batch_size)
    noise_keys = jr.split(noise_key, sweep_config.policy.batch_size)
    vmapped_train_with_noise = eqx.filter_vmap(
        train_with_noise, in_axes=(None, None, 0, 0, 0)
    )

    losses = []
    x_axis = []
    y_axis = []
    for keypoint in keypoints:
        for end_i in range(num_grid_points_per_dim):
            values = jnp.full(keypoints.shape, base_sigma)
            values = values.at[keypoints == keypoint].set(grid_points[end_i])
            sigmas = LinearInterpPolicyNoiseSchedule(
                keypoints, values, T
            ).get_private_sigmas(mu_tot, p, T)

            print(
                f"Training with Sigma Schedule Keypoint {keypoint}, New Sigma {grid_points[end_i]}"
            )
            _, loss, _, _, _ = vmapped_train_with_noise(
                sigmas, env_params, mb_keys, init_keys, noise_keys
            )

            losses.append(loss.mean())
            x_axis.append(keypoint)
            y_axis.append(grid_points[end_i])

    fig = go.Figure(
        data=go.Surface(
            z=jnp.asarray(losses).reshape(num_grid_points_per_dim, -1),
            x=jnp.asarray(x_axis).reshape(num_grid_points_per_dim, -1),
            y=jnp.asarray(y_axis).reshape(num_grid_points_per_dim, -1),
        )
    )

    fig.update_traces(
        contours_z=dict(
            show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
        )
    )
    fig.update_layout(
        title="Loss Landscape over Start and End Sigmas",
        scene=dict(
            xaxis_title="Keypoint",
            yaxis_title="New Sigma Value",
            zaxis_title="Loss",
        ),
    )

    current_dir = os.getcwd()
    filepath = os.path.join(current_dir, "plots", "linear-interp-policy-private.html")
    fig.write_html(filepath)


if __name__ == "__main__":
    main()
