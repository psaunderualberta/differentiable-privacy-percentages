from policy.schedules.abstract import AbstractNoiseAndClipSchedule
from policy.schedules.alternating import AlternatingSigmaAndClipSchedule
from policy.schedules.config import (
    AlternatingSigmaAndClipScheduleConfig,
    DynamicDPSGDScheduleConfig,
    PolicyAndClipScheduleConfig,
    SigmaAndClipScheduleConfig,
)
from policy.schedules.dynamic_dpsgd import DynamicDPSGDSchedule
from policy.schedules.policy_and_clip import PolicyAndClipSchedule
from policy.schedules.sigma_and_clip import SigmaAndClipSchedule
from src.privacy.gdp_privacy import GDPPrivacyParameters


def schedule_factory(
    conf, privacy_params: GDPPrivacyParameters
) -> AbstractNoiseAndClipSchedule:
    if isinstance(conf, AlternatingSigmaAndClipScheduleConfig):
        return AlternatingSigmaAndClipSchedule.from_config(conf, privacy_params)
    elif isinstance(conf, SigmaAndClipScheduleConfig):
        return SigmaAndClipSchedule.from_config(conf, privacy_params)
    elif isinstance(conf, PolicyAndClipScheduleConfig):
        return PolicyAndClipSchedule.from_config(conf, privacy_params)
    elif isinstance(conf, DynamicDPSGDScheduleConfig):
        return DynamicDPSGDSchedule.from_config(conf, privacy_params)

    raise ValueError(f"Configuration not of expected type: {conf.__class__}")
