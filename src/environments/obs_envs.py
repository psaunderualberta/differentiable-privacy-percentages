import chex
import equinox as eqx
import jax.numpy as jnp
from gymnax.environments import EnvParams, EnvState, spaces


class ObservationMaker(eqx.Module):
    def __init__(self, *args, **kwargs):
        pass

    def observation_space(self, params: EnvParams) -> spaces.Box | spaces.Discrete:
        """Number of actions possible in environment."""
        raise NotImplementedError()

    def get_obs(
        self, state: EnvState, params: EnvParams = None, key: chex.PRNGKey = None
    ) -> chex.Array:
        """Creates a continuous action space of the environment."""
        raise NotImplementedError()


class _IterationObs(ObservationMaker):
    """A DP_RL environment where the observation is only the training step of the neural network."""

    def get_obs(
        self, state: EnvState, params: EnvParams = None, key: chex.PRNGKey = None
    ) -> chex.Array:
        """Return observation from raw state class."""
        return state.time.reshape(-1)

    def observation_space(self, params: EnvParams) -> spaces.Box | spaces.Discrete:
        """Observation space of the environment."""

        return spaces.Box(
            low=0,
            high=params.max_steps_in_episode,
            shape=(1,),
            dtype=jnp.float32,
        )


class _AccuracyObs(ObservationMaker):
    def get_obs(
        self, state: EnvState, params: EnvParams = None, key: chex.PRNGKey = None
    ) -> chex.Array:
        """Return observation from raw state class."""
        eps, _ = params.privacy_accountant.get_privacy_expenditure(
            state.privacy_accountant_state
        )

        return jnp.array(
            [
                state.accuracy,
                state.time,
                state.loss,
                state.action,
                eps,
            ]
        )

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(
            low=0,
            high=jnp.inf,
            shape=(5,),
            dtype=jnp.float32,
        )


class _HiddenNodeGradObs(ObservationMaker):
    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """Return observation from raw state class."""

        return state.average_grads  # bias of last layer

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""

        return spaces.Box(
            low=-100,
            high=100,
            shape=(1,),
            dtype=jnp.float32,
        )


ObservationMakers = {
    "accuracy": _AccuracyObs,
    "iteration": _IterationObs,
    "hidden-node-grads": _HiddenNodeGradObs,
}
