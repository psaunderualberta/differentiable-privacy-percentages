import jax.lax as jlax
import jax.numpy as jnp
from jaxtyping import Array

from conf.singleton_conf import SingletonConfig
from policy.base_schedules.abstract import AbstractSchedule
from policy.base_schedules.config import ConstantScheduleConfig


class ConstantSchedule(AbstractSchedule):
    placeholder: Array
    value: Array

    def __init__(self, value: Array | float | int, T: int):
        assert jnp.asarray(value).size == 1, "'Value' must be a single number"
        self.placeholder = jnp.ones((1, T), dtype=jnp.float32).squeeze()
        self.value = jnp.asarray(value)

    @classmethod
    def from_config(cls, conf: ConstantScheduleConfig) -> "ConstantSchedule":
        T = SingletonConfig.get_environment_config_instance().max_steps_in_episode
        return cls(conf.value, T)

    def get_valid_schedule(self) -> Array:
        return jlax.stop_gradient(self.placeholder) * self.value

    def get_raw_schedule(self) -> Array:
        return jlax.stop_gradient(self.placeholder) * self.value

    @classmethod
    def from_projection(
        cls, schedule: "AbstractSchedule", projection: Array
    ) -> "ConstantSchedule":
        return ConstantSchedule(projection.mean(), schedule.placeholder.size)
