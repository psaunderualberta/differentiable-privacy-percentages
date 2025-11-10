import equinox as eqx
from jaxtyping import Array
import jax.numpy as jnp
from privacy.gdp_privacy import sigma_schedule_to_weights, weights_to_sigma_schedule, project_weights
import abc

class AbstractNoiseSchedule(eqx.Module):
    @abc.abstractmethod
    def get_private_sigmas(self, mu: Array, p: Array, T: Array) -> Array:
        raise NotImplementedError("Subclasses must implement get_private_sigmas method.")

    @abc.abstractmethod
    def get_private_schedule(self, mu: Array, p: Array, T: Array) -> Array:
        raise NotImplementedError("Subclasses must implement get_private_schedule method.")

    
class SigmaNoiseSchedule(AbstractNoiseSchedule):
    sigmas: Array

    def __init__(self, sigma: float, length: int):
        self.sigmas = jnp.full((length,), sigma, dtype=jnp.float32)
    
    def get_private_sigmas(self, mu: Array, p: Array, T: Array) -> Array:
        weights = sigma_schedule_to_weights(self.schedule, mu, p, T)
        proj_weights = project_weights(weights, mu, p, T)
        private_sigmas = weights_to_sigma_schedule(proj_weights, mu, p, T)
        return private_sigmas
    
    def get_private_schedule(self, mu: Array, p: Array, T: Array) -> Array:
        weights = sigma_schedule_to_weights(self.sigmas, mu, p, T)
        proj_weights = project_weights(weights, mu, p, T)
        return proj_weights

    def get_sigmas(self) -> Array:
        return self.sigmas


class PolicyNoiseSchedule(AbstractNoiseSchedule):
    schedule: Array

    def __init__(self, schedule: Array):
        self.schedule = schedule
    
    def get_private_sigmas(self, mu: Array, p: Array, T: Array) -> Array:
        proj_weights = project_weights(self.schedule, mu, p, T)
        private_sigmas = weights_to_sigma_schedule(proj_weights, mu, p, T)
        return private_sigmas
    
    def get_private_schedule(self, mu: Array, p: Array, T: Array) -> Array:
        proj_weights = project_weights(self.schedule, mu, p, T)
        return proj_weights


class LinearInterpNoiseSchedule(AbstractNoiseSchedule):
    keypoints: Array
    values: Array
    eps: float = 1e-6

    def __init__(self, keypoints: Array, values: Array, eps: float = 1e-6):
        self.keypoints = keypoints
        self.values = values
        self.eps = eps
    
    def get_private_sigmas(self, mu: Array, p: Array, T: Array) -> Array:
        values = jnp.clip(self.values, min=self.eps, max=1-self.eps)
        schedule = jnp.interp(jnp.arange(T), self.keypoints, values)
        proj_weights = project_weights(schedule, mu, p, T)
        private_sigmas = weights_to_sigma_schedule(proj_weights, mu, p, T)
        return private_sigmas
    
    def get_private_schedule(self, mu: Array, p: Array, T: Array) -> Array:
        schedule = jnp.interp(jnp.arange(T), self.keypoints, self.values)
        proj_weights = project_weights(schedule, mu, p, T)
        return proj_weights
    
