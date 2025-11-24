import equinox as eqx
from jaxtyping import Array
import jax.numpy as jnp
from jax import lax as jlax
import abc


class AbstractSchedule(eqx.Module):
    @abc.abstractmethod
    def get_valid_schedule(self) -> Array:
        raise NotImplementedError(
            "Subclasses must implement get_private_sigmas method."
        )

    @abc.abstractmethod
    def get_raw_schedule(self) -> Array:
        raise NotImplementedError(
            "Subclasses must implement get_private_schedule method."
        )


class ClippedSchedule(AbstractSchedule):
    schedule: Array
    min_value: float

    def __init__(self, schedule: Array, min_value: float):
        self.schedule = schedule
        self.min_value = min_value

    def get_valid_schedule(self) -> Array:
        return jnp.clip(self.schedule, a_min=jlax.stop_gradient(self.min_value))

    def get_raw_schedule(self) -> Array:
        return self.schedule


class ExponentialSchedule(AbstractSchedule):
    schedule: Array

    def __init__(self, schedule: Array):
        self.schedule = schedule

    def get_valid_schedule(self) -> Array:
        return jnp.exp(self.schedule)

    def get_raw_schedule(self) -> Array:
        return self.schedule


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

    def get_valid_schedule(self) -> Array:
        clipped_values = jnp.clip(self.values, min=self.eps)
        return jnp.interp(
            self.points, jlax.stop_gradient(self.keypoints), clipped_values
        )

    def get_raw_schedule(self) -> Array:
        return jnp.interp(self.points, jlax.stop_gradient(self.keypoints), self.values)


class InterpolatedExponentialSchedule(AbstractSchedule):
    keypoints: Array
    values: Array
    points: Array

    def __init__(self, keypoints: Array, values: Array, T: int):
        self.keypoints = keypoints
        self.values = values
        self.points = jnp.arange(T)

    def get_valid_schedule(self) -> Array:
        return jnp.interp(
            self.points, jlax.stop_gradient(self.keypoints), jnp.exp(self.values)
        )

    def get_raw_schedule(self) -> Array:
        return jnp.interp(self.points, jlax.stop_gradient(self.keypoints), self.values)
