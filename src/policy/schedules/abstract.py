from abc import abstractmethod
from typing import Self

import equinox as eqx
from jaxtyping import Array

from util.logger import Loggable, LoggableArray, LoggingSchema


class AbstractNoiseAndClipSchedule(eqx.Module):
    @abstractmethod
    def get_private_sigmas(self) -> Array:
        raise NotImplementedError(
            "Subclasses must implement get_private_sigmas method."
        )

    @abstractmethod
    def get_private_clips(self) -> Array:
        raise NotImplementedError("Subclasses must implement get_private_clips method.")

    @abstractmethod
    def get_private_weights(self) -> Array:
        raise NotImplementedError(
            "Subclasses must implement get_private_weights method."
        )

    @abstractmethod
    def apply_updates(self, updates) -> Self:
        raise NotImplementedError(
            "Subclasses must implement get_private_weights method."
        )

    @abstractmethod
    def project(self) -> Self:
        raise NotImplementedError("Subclasses must implement 'project' class method.")

    @abstractmethod
    def get_logging_schemas(self) -> list[LoggingSchema]:
        raise NotImplementedError(
            "Subclasses must implement get_logging_schemas method."
        )

    @abstractmethod
    def get_loggables(self, force=False) -> list[Loggable | LoggableArray]:
        raise NotImplementedError("Subclasses must implement get_loggables method.")
