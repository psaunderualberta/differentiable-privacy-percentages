import jax.lax as jlax
import jax.numpy as jnp
from jaxtyping import Array

from conf.singleton_conf import SingletonConfig
from policy.base_schedules.abstract import AbstractSchedule
from policy.base_schedules.config import InterpolatedClippedScheduleConfig


class InterpolatedClippedSchedule(AbstractSchedule):
    keypoints: Array
    values: Array
    points: Array
    eps: float = 1e-6

    def __init__(self, keypoints: Array, values: Array, T: int, eps: float = 1e-6):
        self.keypoints = keypoints
        self.values = values
        self.points = jnp.arange(T)
        self.eps = eps

    @classmethod
    def from_config(
        cls, conf: InterpolatedClippedScheduleConfig
    ) -> "InterpolatedClippedSchedule":
        T = SingletonConfig.get_environment_config_instance().max_steps_in_episode
        keypoints = jnp.linspace(0, T, conf.num_keypoints, dtype=jnp.int32)
        values = jnp.zeros_like(keypoints, dtype=jnp.float32) + conf.init_value

        return cls(keypoints, values, T)

    def get_valid_schedule(self) -> Array:
        clipped_values = jnp.clip(self.values, min=self.eps)
        return jnp.interp(
            self.points, jlax.stop_gradient(self.keypoints), clipped_values
        )

    def get_raw_schedule(self) -> Array:
        return jnp.interp(self.points, jlax.stop_gradient(self.keypoints), self.values)

    @classmethod
    def from_projection(
        cls, schedule: "InterpolatedClippedSchedule", projection: Array
    ) -> "InterpolatedClippedSchedule":
        return InterpolatedClippedSchedule(
            keypoints=schedule.keypoints,
            values=projection[schedule.keypoints.astype(int)],
            T=len(schedule.points),
            eps=schedule.eps,
        )
