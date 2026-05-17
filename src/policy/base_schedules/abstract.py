import abc

import equinox as eqx
import jax
from jaxtyping import Array, PyTree


class AbstractSchedule(eqx.Module):
    @abc.abstractmethod
    def get_valid_schedule(self) -> Array:
        raise NotImplementedError(
            "Subclasses must implement get_private_sigmas method.",
        )

    @abc.abstractmethod
    def get_raw_schedule(self) -> Array:
        raise NotImplementedError("Subclasses must implement get_raw_schedule method.")

    @classmethod
    @abc.abstractmethod
    def from_projection(
        cls,
        schedule: "AbstractSchedule",
        projection: Array,
    ) -> "AbstractSchedule":
        raise NotImplementedError(
            "Subclasses must implement 'from_projection' class method.",
        )

    def es_filter(self) -> PyTree:
        """Return a filter spec (same PyTree structure as ``self``) marking
        which leaves should be optimised by Evolutionary Strategies.

        The default implementation opts every leaf out (all-False). Subclasses
        with ES-supported parameters override this to mark them True.
        """
        return jax.tree.map(lambda _: False, self)
