from policy.base_schedules.abstract import AbstractSchedule
from policy.base_schedules.clipped import InterpolatedClippedSchedule
from policy.base_schedules.config import (
    ConstantScheduleConfig,
    InterpolatedClippedScheduleConfig,
    InterpolatedExponentialScheduleConfig,
)
from policy.base_schedules.constant import ConstantSchedule
from policy.base_schedules.exponential import InterpolatedExponentialSchedule


def base_schedule_factory(conf) -> AbstractSchedule:
    if isinstance(conf, ConstantScheduleConfig):
        return ConstantSchedule.from_config(conf)
    elif isinstance(conf, InterpolatedClippedScheduleConfig):
        return InterpolatedClippedSchedule.from_config(conf)
    elif isinstance(conf, InterpolatedExponentialScheduleConfig):
        return InterpolatedExponentialSchedule.from_config(conf)

    raise ValueError(f"Configuration not of expected type: {conf.__class__}")
