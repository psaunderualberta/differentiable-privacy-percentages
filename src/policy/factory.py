from policy.base_schedules.config import AbstractScheduleConfig
from policy.schedules.factory import schedule_factory
from policy.stateful_schedules.config import AbstractStatefulScheduleConfig
from policy.stateful_schedules.factory import stateful_schedule_factory
from privacy.gdp_privacy import GDPPrivacyParameters


def policy_factory(conf, privacy_params: GDPPrivacyParameters):
    if isinstance(conf, AbstractScheduleConfig):
        return schedule_factory(conf, privacy_params)
    elif isinstance(conf, AbstractStatefulScheduleConfig):
        return stateful_schedule_factory(conf, privacy_params)

    raise ValueError(f"Configuration not of expected type: {conf.__class__}")
