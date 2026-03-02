import dataclasses
from dataclasses import dataclass
from typing import Annotated, Union

import tyro

from conf.config_util import to_wandb_sweep_params


class AbstractScheduleConfig:
    pass


@dataclass
class ConstantScheduleConfig(AbstractScheduleConfig):
    init_value: float = 1.0

    def to_wandb_sweep(self):
        return to_wandb_sweep_params(self)


@dataclass
class InterpolatedExponentialScheduleConfig(AbstractScheduleConfig):
    num_keypoints: int = 50
    init_value: float = 1.0

    def to_wandb_sweep(self):
        return to_wandb_sweep_params(self)


@dataclass
class InterpolatedClippedScheduleConfig(AbstractScheduleConfig):
    num_keypoints: int = 50
    init_value: float = 1.0

    def to_wandb_sweep(self):
        return to_wandb_sweep_params(self)


# Union type used by schedule configs to select a base schedule parametrisation.
# tyro treats a Union of dataclasses as subcommands automatically; only the
# selected variant is ever constructed.
BaseScheduleConfig = Union[
    Annotated[ConstantScheduleConfig, tyro.conf.subcommand("constant")],
    Annotated[
        InterpolatedExponentialScheduleConfig, tyro.conf.subcommand("interplated-exp")
    ],
    Annotated[
        InterpolatedClippedScheduleConfig, tyro.conf.subcommand("interpolated-clipped")
    ],
]
