from dataclasses import dataclass

from tyro.conf import Fixed

from conf.config_util import to_wandb_sweep_params


class AbstractStatefulScheduleConfig:
    pass


@dataclass
class StatefulMedianGradientNoiseAndClipConfig(AbstractStatefulScheduleConfig):
    c_0: float = 0.1
    eta_c: float = 0.2
    attrs: Fixed[tuple[str, ...]] = ("c_0", "eta_c")

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)
