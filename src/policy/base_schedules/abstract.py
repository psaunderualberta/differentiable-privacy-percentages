import abc

import equinox as eqx
from jaxtyping import Array


class AbstractSchedule(eqx.Module):
    @abc.abstractmethod
    def get_valid_schedule(self) -> Array:
        raise NotImplementedError(
            "Subclasses must implement get_private_sigmas method."
        )

    @abc.abstractmethod
    def get_raw_schedule(self) -> Array:
        raise NotImplementedError("Subclasses must implement get_raw_schedule method.")

    @classmethod
    @abc.abstractmethod
    def from_projection(
        cls, schedule: "AbstractSchedule", projection: Array
    ) -> "AbstractSchedule":
        raise NotImplementedError(
            "Subclasses must implement 'from_projection' class method."
        )
