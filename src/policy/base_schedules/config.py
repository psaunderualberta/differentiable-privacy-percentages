from dataclasses import dataclass

from tyro.conf import Fixed

from conf.config_util import to_wandb_sweep_params


@dataclass
class ConstantScheduleConfig:
    value: float = 1.0
    attrs: Fixed[tuple[str, ...]] = ("value",)

    def to_wandb_sweep(self):
        return to_wandb_sweep_params(self)


@dataclass
class InterpolatedExponentialScheduleConfig:
    num_keypoints: int = 50
    init_value: float = 1.0
    attrs: Fixed[tuple[str, ...]] = ("num_keypoints", "init_value")

    def to_wandb_sweep(self):
        return to_wandb_sweep_params(self)


@dataclass
class InterpolatedClippedScheduleConfig:
    num_keypoints: int = 50
    init_value: float = 1.0
    attrs: Fixed[tuple[str, ...]] = ("num_keypoints", "init_value")

    def to_wandb_sweep(self):
        return to_wandb_sweep_params(self)
