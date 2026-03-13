import jax.lax as jlax
import jax.numpy as jnp
from jaxtyping import Array

from policy.base_schedules._registry import register
from policy.base_schedules.abstract import AbstractSchedule
from policy.base_schedules.config import ConstantScheduleConfig


@register(ConstantScheduleConfig)
class ConstantSchedule(AbstractSchedule):
    placeholder: Array
    value: Array

    def __init__(self, value: Array | float, T: int):
        """Initialise a constant schedule with a single learnable scalar.

        Args:
            value: Initial (and only) scalar value broadcast across all T steps.
            T: Total number of training steps; determines the length of the schedule.
        """
        assert jnp.asarray(value).size == 1, "'Value' must be a single number"
        self.placeholder = jnp.ones((1, T), dtype=jnp.float32).squeeze()
        self.value = jnp.asarray(value, dtype=jnp.float32)

    @classmethod
    def from_config(cls, conf: ConstantScheduleConfig, T: int) -> "ConstantSchedule":
        return cls(conf.init_value, T)

    def get_valid_schedule(self) -> Array:
        """Return the length-T schedule array (stop-gradient on the placeholder ones vector)."""
        return jlax.stop_gradient(self.placeholder) * self.value

    def get_raw_schedule(self) -> Array:
        """Return the raw (un-projected) length-T schedule array; identical to `get_valid_schedule` for a constant."""
        return jlax.stop_gradient(self.placeholder) * self.value

    @classmethod
    def from_projection(
        cls,
        schedule: "AbstractSchedule",
        projection: Array,
    ) -> "ConstantSchedule":
        return ConstantSchedule(projection.mean(), schedule.placeholder.size)
