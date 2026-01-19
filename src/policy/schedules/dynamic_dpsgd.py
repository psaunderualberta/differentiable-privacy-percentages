from typing import Self

import equinox as eqx
import jax.lax as jlax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from scipy import optimize

from conf.singleton_conf import SingletonConfig
from policy.schedules.abstract import AbstractNoiseAndClipSchedule
from policy.schedules.config import DynamicDPSGDScheduleConfig
from privacy.gdp_privacy import GDPPrivacyParameters
from util.logger import Loggable, LoggableArray, LoggingSchema


class DynamicDPSGDSchedule(AbstractNoiseAndClipSchedule):
    """
    http://arxiv.org/abs/2111.00173
    """

    __iters: Array
    __mu_0: Array
    privacy_params: GDPPrivacyParameters
    rho_mu: Array
    rho_C: Array
    C_0: Array
    __eps: Array

    def __init__(
        self,
        rho_mu: ArrayLike,
        rho_C: ArrayLike,
        c_0: ArrayLike,
        privacy_params: GDPPrivacyParameters,
        eps: Array | float = 0.01,
    ):
        T = privacy_params.T
        self.__iters = jnp.arange(1, T + 1)
        self.rho_mu = jnp.asarray(rho_mu)
        self.rho_C = jnp.asarray(rho_C)
        self.C_0 = jnp.asarray(c_0)
        self.__eps = jnp.asarray(eps)

        self.privacy_params = privacy_params
        self.__mu_0 = self.__find_mu_0()

    @classmethod
    def from_config(
        cls, conf: DynamicDPSGDScheduleConfig, privacy_params: GDPPrivacyParameters
    ) -> "DynamicDPSGDSchedule":
        return cls(conf.rho_mu, conf.rho_c, conf.c_0, privacy_params)

    def __find_mu_0(self, tol=1e-12):
        mu_tot = self.privacy_params.mu
        p = self.privacy_params.p
        pows = self.rho_mu ** (self.iters / self.iters.size)

        def f(current_mu_0):
            # Eq'n 10 in reference material
            current_mu_tot = jnp.sqrt(
                p**2 * jnp.sum(jnp.exp((pows * current_mu_0) ** 2) - 1)
            )
            return current_mu_tot - mu_tot

        found_mu = optimize.root_scalar(f, bracket=[tol, 100], method="brentq").root
        return jnp.asarray(found_mu)

    @property
    def iters(self) -> Array:
        return jlax.stop_gradient(self.__iters)

    @property
    def eps(self) -> Array:
        return jlax.stop_gradient(self.__eps)

    @property
    def mu_0(self) -> Array:
        return jlax.stop_gradient(self.__mu_0)

    def get_private_sigmas(self) -> Array:
        return (self.C_0 / self.mu_0) * (self.rho_mu * self.rho_C) ** (
            -self.iters / self.privacy_params.T
        )

    def get_private_clips(self) -> Array:
        return self.C_0 * self.rho_C ** (-self.iters / self.privacy_params.T)

    def get_private_weights(self) -> Array:
        private_sigmas = self.get_private_sigmas()
        clips = self.get_private_clips()

        weights = self.privacy_params.sigma_schedule_to_weights(clips, private_sigmas)
        proj_weights = self.privacy_params.project_weights(weights)
        return proj_weights.squeeze()

    def apply_updates(self, updates) -> Self:
        return eqx.apply_updates(self, updates)

    def project(self) -> Self:
        rho_mu = jnp.maximum(self.rho_mu, self.eps)
        rho_C = jnp.maximum(self.rho_C, self.eps)
        C_0 = jnp.maximum(self.C_0, self.eps)

        return self.__class__(rho_mu, rho_C, C_0, self.privacy_params, self.eps)

    def get_logging_schemas(self) -> list[LoggingSchema]:
        plot_interval = SingletonConfig.get_sweep_config_instance().plotting_interval
        col_names = [str(step) for step in range(len(self.get_private_sigmas()))]
        return [
            LoggingSchema(table_name="sigmas", cols=col_names, freq=plot_interval),
            LoggingSchema(table_name="clips", cols=col_names, freq=plot_interval),
            LoggingSchema(table_name="weights", cols=col_names, freq=plot_interval),
            LoggingSchema(table_name="mus", cols=col_names, freq=plot_interval),
        ]

    def get_loggables(self, force=False) -> list[Loggable | LoggableArray]:
        return [
            LoggableArray(
                table_name="sigmas",
                array=self.get_private_sigmas(),
                plot=True,
                force=force,
            ),
            LoggableArray(
                table_name="clips",
                array=self.get_private_clips(),
                plot=True,
                force=force,
            ),
            LoggableArray(
                table_name="weights",
                array=self.get_private_weights(),
                plot=True,
                force=force,
            ),
            LoggableArray(
                table_name="mus",
                array=self.privacy_params.weights_to_mu_schedule(
                    self.get_private_weights()
                ),
                plot=True,
                force=force,
            ),
        ]
