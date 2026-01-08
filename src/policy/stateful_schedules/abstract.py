from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array

from util.logger import Loggable, LoggableArray, LoggingSchema


class AbstractScheduleState(eqx.Module):
    @abstractmethod
    def get_clip(self) -> Array:
        raise NotImplementedError("Subclasses must implement get_clip method.")

    @abstractmethod
    def get_noise(self) -> Array:
        raise NotImplementedError("Subclasses must implement get_noise method.")


class AbstractStatefulNoiseAndClipSchedule(eqx.Module):
    iteration_array: eqx.AbstractVar[Array]

    def get_iteration_array(self) -> Array:
        return self.iteration_array

    @abstractmethod
    def get_initial_state(self) -> AbstractScheduleState:
        raise NotImplementedError("Subclasses must implement get_initial_state method.")

    @abstractmethod
    def update_state(
        self,
        state: AbstractScheduleState,
        grads: Array,
        iter: Array,
        batch_x: Array,
        batch_y: Array,
    ) -> AbstractScheduleState:
        raise NotImplementedError("Subclasses must implement update_state method.")

    @abstractmethod
    def get_logging_schemas(self) -> list[LoggingSchema]:
        raise NotImplementedError(
            "Subclasses must implement get_logging_schemas method."
        )

    @abstractmethod
    def get_loggables(self, force=False) -> list[Loggable | LoggableArray]:
        raise NotImplementedError("Subclasses must implement get_loggables method.")
