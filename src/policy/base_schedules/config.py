from dataclasses import dataclass

from tyro.conf import Fixed

from conf.config_util import to_wandb_sweep_params


class AbstractScheduleConfig:
    pass


@dataclass
class ConstantScheduleConfig(AbstractScheduleConfig):
    init_value: float = 1.0
    attrs: Fixed[tuple[str, ...]] = ("init_value",)

    def to_wandb_sweep(self):
        return to_wandb_sweep_params(self)


@dataclass
class InterpolatedExponentialScheduleConfig(AbstractScheduleConfig):
    num_keypoints: int = 50
    init_value: float = 1.0
    attrs: Fixed[tuple[str, ...]] = ("num_keypoints", "init_value")

    def to_wandb_sweep(self):
        return to_wandb_sweep_params(self)


@dataclass
class InterpolatedClippedScheduleConfig(AbstractScheduleConfig):
    num_keypoints: int = 50
    init_value: float = 1.0
    attrs: Fixed[tuple[str, ...]] = ("num_keypoints", "init_value")

    def to_wandb_sweep(self):
        return to_wandb_sweep_params(self)
