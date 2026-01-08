from policy.stateful_schedules.abstract import AbstractStatefulNoiseAndClipSchedule
from policy.stateful_schedules.config import StatefulMedianGradientNoiseAndClipConfig
from policy.stateful_schedules.median_gradient import (
    StatefulMedianGradientNoiseAndClipSchedule,
)
from src.privacy.gdp_privacy import GDPPrivacyParameters


def base_schedule_factory(
    conf, privacy_params: GDPPrivacyParameters
) -> AbstractStatefulNoiseAndClipSchedule:
    if isinstance(conf, StatefulMedianGradientNoiseAndClipConfig):
        return StatefulMedianGradientNoiseAndClipSchedule.from_config(
            conf, privacy_params
        )
    raise ValueError(f"Configuration not of expected type: {conf.__class__}")
