import jax.numpy as jnp
import optax
from jax import vmap
from jaxtyping import Array, ArrayLike

from conf.singleton_conf import SingletonConfig
from policy.stateful_schedules.abstract import (
    AbstractScheduleState,
    AbstractStatefulNoiseAndClipSchedule,
)
from policy.stateful_schedules.config import StatefulMedianGradientNoiseAndClipConfig
from privacy.gdp_privacy import GDPPrivacyParameters
from util.logger import Loggable, LoggableArray, LoggingSchema


class StatefulMedianGradientNoiseAndClipSchedule(AbstractStatefulNoiseAndClipSchedule):
    """
    https://proceedings.neurips.cc/paper_files/paper/2021/file/91cff01af640a24e7f9f7a5ab407889f-Paper.pdf
    """

    class MedianGradientScheduleState(AbstractScheduleState):
        C: Array
        sigma: Array

        def __init__(self, C: Array, sigma: Array):
            self.C = C
            self.sigma = sigma

        def get_clip(self) -> Array:
            return self.C

        def get_noise(self) -> Array:
            return self.sigma

    c_0: Array
    eta_c: Array
    iteration_array: Array
    privacy_params: GDPPrivacyParameters
    gamma: float = 0.5

    def __init__(
        self, c_0: ArrayLike, eta_c: ArrayLike, privacy_params: GDPPrivacyParameters
    ):
        self.c_0 = jnp.asarray(c_0)
        self.eta_c = jnp.asarray(eta_c)
        self.privacy_params = privacy_params
        self.iteration_array = jnp.arange(self.privacy_params.T)

    @classmethod
    def from_config(
        cls,
        conf: StatefulMedianGradientNoiseAndClipConfig,
        privacy_params: GDPPrivacyParameters,
    ) -> "StatefulMedianGradientNoiseAndClipSchedule":
        return cls(conf.c_0, conf.eta_c, privacy_params)

    def get_initial_state(self) -> MedianGradientScheduleState:
        sigma = self.c_0 / self.privacy_params.mu_0
        return self.MedianGradientScheduleState(C=self.c_0, sigma=sigma)

    def update_state(
        self,
        state: AbstractScheduleState,
        grads: Array,
        iter: Array,
        batch_x: Array,
        batch_y: Array,
    ) -> MedianGradientScheduleState:
        current_C = state.get_clip()
        norms = vmap(optax.tree.norm)(grads)
        b_bar = (norms <= current_C).mean()

        new_C = current_C * jnp.exp(-self.eta_c * (b_bar - self.gamma))
        new_sigma = new_C / self.privacy_params.mu_0

        return self.MedianGradientScheduleState(C=new_C, sigma=new_sigma)

    def get_logging_schemas(self) -> list[LoggingSchema]:
        plot_interval = SingletonConfig.get_sweep_config_instance().plotting_interval
        col_names = [str(step) for step in range(len(self.get_private_mus()))]
        return [
            LoggingSchema(table_name="weights", cols=col_names, freq=plot_interval),
            LoggingSchema(table_name="mus", cols=col_names, freq=plot_interval),
        ]

    def get_loggables(self, force=False) -> list[Loggable | LoggableArray]:
        return [
            LoggableArray(
                table_name="weights",
                array=self.get_private_weights(),
                plot=True,
                force=force,
            ),
            LoggableArray(
                table_name="mus",
                array=self.get_private_mus(),
                plot=True,
                force=force,
            ),
        ]
