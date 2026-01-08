from dataclasses import dataclass
from typing import Annotated, Union

import tyro

from conf.config_util import to_wandb_sweep_params
from policy.base_schedules.config import (
    ConstantScheduleConfig,
    InterpolatedClippedScheduleConfig,
    InterpolatedExponentialScheduleConfig,
)


@dataclass
class SigmaAndClipScheduleConfig:
    noise: Union[
        Annotated[ConstantScheduleConfig, tyro.conf.subcommand("constant")],
        Annotated[InterpolatedClippedScheduleConfig, tyro.conf.subcommand("clipped")],
        Annotated[
            InterpolatedExponentialScheduleConfig, tyro.conf.subcommand("exponential")
        ],
    ]
    clip: Union[
        Annotated[ConstantScheduleConfig, tyro.conf.subcommand("constant")],
        Annotated[InterpolatedClippedScheduleConfig, tyro.conf.subcommand("clipped")],
        Annotated[
            InterpolatedExponentialScheduleConfig, tyro.conf.subcommand("exponential")
        ],
    ]
    attrs: tyro.conf.Fixed[tuple[str, ...]] = ("noise", "clip")

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)


@dataclass
class PolicyAndClipScheduleConfig:
    noise: Union[
        Annotated[ConstantScheduleConfig, tyro.conf.subcommand("constant")],
        Annotated[InterpolatedClippedScheduleConfig, tyro.conf.subcommand("clipped")],
        Annotated[
            InterpolatedExponentialScheduleConfig, tyro.conf.subcommand("exponential")
        ],
    ]
    clip: Union[
        Annotated[ConstantScheduleConfig, tyro.conf.subcommand("constant")],
        Annotated[InterpolatedClippedScheduleConfig, tyro.conf.subcommand("clipped")],
        Annotated[
            InterpolatedExponentialScheduleConfig, tyro.conf.subcommand("exponential")
        ],
    ]
    attrs: tyro.conf.Fixed[tuple[str, ...]] = ("noise", "clip")

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)


@dataclass
class AlternatingSigmaAndClipScheduleConfig:
    noise: Union[
        Annotated[ConstantScheduleConfig, tyro.conf.subcommand("constant")],
        Annotated[InterpolatedClippedScheduleConfig, tyro.conf.subcommand("clipped")],
        Annotated[
            InterpolatedExponentialScheduleConfig, tyro.conf.subcommand("exponential")
        ],
    ]
    clip: Union[
        Annotated[ConstantScheduleConfig, tyro.conf.subcommand("constant")],
        Annotated[InterpolatedClippedScheduleConfig, tyro.conf.subcommand("clipped")],
        Annotated[
            InterpolatedExponentialScheduleConfig, tyro.conf.subcommand("exponential")
        ],
    ]
    diff_clips_first: bool = False
    attrs: tyro.conf.Fixed[tuple[str, ...]] = ("noise", "clip", "diff_clips_first")

    def to_wandb_sweep(self) -> dict[str, object]:
        return to_wandb_sweep_params(self)


@dataclass
class DynamicDPSGDScheduleConfig:
    rho_mu: float = 0.5
    rho_c: float = 0.5
    c_0: float = 1.5
    attrs: tyro.conf.Fixed[tuple[str, ...]] = ("rho_mu", "rho_c", "c_0")

    def to_wandb_sweep(self):
        return to_wandb_sweep_params(self)
