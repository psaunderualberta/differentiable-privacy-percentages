from abc import abstractmethod

import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array
from optax import global_norm

from conf.singleton_conf import SingletonConfig
from privacy.gdp_privacy import GDPPrivacyParameters
from util.logger import Loggable, LoggableArray, LoggingSchema


class AbstractGradientDerivedNoiseAndClipSchedule:
    @abstractmethod
    def get_private_mus(self) -> Array:
        raise NotImplementedError("Subclasses must implement get_private_mus method.")

    def get_private_weights(self) -> Array:
        raise NotImplementedError(
            "Subclasses must implement get_private_weights method."
        )

    @abstractmethod
    def get_derived_noise_and_clip(
        self, grads: Array, iter: Array
    ) -> tuple[Array, Array]:
        raise NotImplementedError(
            "Subclasses must implement get_derived_noise_and_clip method."
        )

    @abstractmethod
    def get_logging_schemas(self) -> list[LoggingSchema]:
        raise NotImplementedError(
            "Subclasses must implement get_logging_schemas method."
        )

    @abstractmethod
    def get_loggables(self, force=False) -> list[Loggable | LoggableArray]:
        raise NotImplementedError("Subclasses must implement get_loggables method.")


class MedianGradientNoiseAndClipSchedule(AbstractGradientDerivedNoiseAndClipSchedule):
    weights: Array
    privacy_params: GDPPrivacyParameters

    def __init__(self, initial_weights: Array, privacy_params: GDPPrivacyParameters):
        self.weights = initial_weights
        self.privacy_params = privacy_params

    def get_private_mus(self) -> Array:
        return self.privacy_params.weights_to_mu_schedule(self.get_private_weights())

    def get_derived_noise_and_clip(
        self, grads: Array, iter: Array
    ) -> tuple[Array, Array]:
        norms = vmap(global_norm)(grads)
        median_norm = jnp.median(norms)
        mu = self.get_private_mus()[iter]
        noise = median_norm / mu

        return noise, median_norm

    def get_logging_schemas(self) -> list[LoggingSchema]:
        plot_interval = SingletonConfig.get_sweep_config_instance().plotting_interval
        col_names = [str(step) for step in range(len(self.get_private_mus()))]
        return [
            LoggingSchema(table_name="weights", cols=col_names, freq=plot_interval),
            LoggingSchema(table_name="mus", cols=col_names, freq=plot_interval),
        ]

    def get_loggables(self, force=False) -> list[Loggable | LoggableArray]:
        return [
            LoggableArray(
                table_name="weights",
                array=self.get_private_weights(),
                plot=True,
                force=force,
            ),
            LoggableArray(
                table_name="mus",
                array=self.get_private_mus(),
                plot=True,
                force=force,
            ),
        ]
