import jax.lax as jlax
import jax.numpy as jnp
from jaxtyping import Array

from conf.singleton_conf import SingletonConfig
from policy.base_schedules.abstract import AbstractSchedule
from policy.base_schedules.config import InterpolatedExponentialScheduleConfig


class InterpolatedExponentialSchedule(AbstractSchedule):
    keypoints: Array
    values: Array
    points: Array

    def __init__(self, keypoints: Array, values: Array, T: int):
        self.keypoints = keypoints
        self.values = values
        self.points = jnp.arange(T)

    @classmethod
    def from_config(
        cls, conf: InterpolatedExponentialScheduleConfig
    ) -> "InterpolatedExponentialSchedule":
        T = SingletonConfig.get_environment_config_instance().max_steps_in_episode
        keypoints = jnp.arange(0, T + 1, step=T // conf.num_keypoints, dtype=jnp.int32)
        values = jnp.zeros_like(keypoints, dtype=jnp.float32) + conf.init_value

        return cls(keypoints, values, T)

    def get_valid_schedule(self) -> Array:
        return jnp.interp(
            self.points,
            jlax.stop_gradient(self.keypoints),
            jnp.log(jnp.exp(self.values) + 1),
        )

    def get_raw_schedule(self) -> Array:
        return jnp.interp(self.points, jlax.stop_gradient(self.keypoints), self.values)

    @classmethod
    def from_projection(
        cls, schedule: "InterpolatedExponentialSchedule", projection: Array
    ) -> "InterpolatedExponentialSchedule":
        reset_projection = jnp.log(jnp.exp(projection) - 1 + 1e-6)
        return InterpolatedExponentialSchedule(
            keypoints=schedule.keypoints,
            values=reset_projection[schedule.keypoints.astype(int)],
            T=len(schedule.points),
        )
