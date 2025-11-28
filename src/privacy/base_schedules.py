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
            "Subclasses must implement get_raw_schedule method."
        )

    @classmethod
    @abc.abstractmethod
    def from_projection(
        cls, schedule: "AbstractSchedule", projection: Array
    ) -> "AbstractSchedule":
        raise NotImplementedError(
            "Subclasses must implement 'from_projection' class method."
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

    @classmethod
    def from_projection(
        cls, schedule: "ClippedSchedule", projection: Array
    ) -> "ClippedSchedule":
        return ClippedSchedule(schedule=projection, min_value=schedule.min_value)


class ExponentialSchedule(AbstractSchedule):
    schedule: Array

    def __init__(self, schedule: Array):
        self.schedule = schedule

    def get_valid_schedule(self) -> Array:
        return jnp.log(jnp.exp(self.schedule) + 1)

    def get_raw_schedule(self) -> Array:
        return self.schedule

    @classmethod
    def from_projection(
        cls, schedule: "ExponentialSchedule", projection: Array
    ) -> "ExponentialSchedule":
        reset_projection = jnp.log(jnp.exp(projection) - 1 + 1e-6)
        return ExponentialSchedule(schedule=reset_projection)


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
            self.points, jlax.stop_gradient(self.keypoints), jnp.log(jnp.exp(self.values) + 1)
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
