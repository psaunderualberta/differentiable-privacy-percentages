import dataclasses
from abc import abstractmethod
from typing import Self

import equinox as eqx
import jax
from jaxtyping import Array, PyTree

from conf.singleton_conf import SingletonConfig
from policy.base_schedules.abstract import AbstractSchedule
from util.logger import Loggable, LoggableArray, LoggingSchema


class AbstractNoiseAndClipSchedule(eqx.Module):
    @abstractmethod
    def get_private_sigmas(self) -> Array:
        raise NotImplementedError(
            "Subclasses must implement get_private_sigmas method.",
        )

    @abstractmethod
    def get_private_clips(self) -> Array:
        raise NotImplementedError("Subclasses must implement get_private_clips method.")

    @abstractmethod
    def get_private_weights(self) -> Array:
        raise NotImplementedError(
            "Subclasses must implement get_private_weights method.",
        )

    @abstractmethod
    def apply_updates(self, updates) -> Self:
        raise NotImplementedError(
            "Subclasses must implement get_private_weights method.",
        )

    @abstractmethod
    def project(self) -> Self:
        raise NotImplementedError("Subclasses must implement 'project' class method.")

    @abstractmethod
    def _get_log_arrays(self) -> dict[str, Array]:
        """Return ordered {table_name: array} pairs for logging."""
        raise NotImplementedError("Subclasses must implement _get_log_arrays method.")

    def get_logging_schemas(self) -> list[LoggingSchema]:
        plot_interval = SingletonConfig.get_sweep_config_instance().plotting_interval
        arrays = self._get_log_arrays()
        col_names = [str(i) for i in range(len(next(iter(arrays.values()))))]
        return [
            LoggingSchema(table_name=name, cols=col_names, freq=plot_interval) for name in arrays
        ]

    def es_filter(self) -> PyTree:
        """Return a filter spec (same PyTree structure as ``self``) marking
        leaves to be optimised by Evolutionary Strategies.

        Default composition: every leaf is False except for fields that are
        themselves ``AbstractSchedule`` instances, whose own ``es_filter``
        is spliced in (base-first composition).
        """
        spec = jax.tree.map(lambda _: False, self)
        for f in dataclasses.fields(self):
            val = getattr(self, f.name)
            if isinstance(val, AbstractSchedule):
                sub_spec = val.es_filter()
                spec = eqx.tree_at(
                    lambda s, name=f.name: getattr(s, name),
                    spec,
                    sub_spec,
                )
        return spec

    def get_loggables(self, force=False) -> list[Loggable | LoggableArray]:
        return [
            LoggableArray(table_name=name, array=arr, plot=True, force=force)
            for name, arr in self._get_log_arrays().items()
        ]
