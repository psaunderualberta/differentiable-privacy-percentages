from policy.schedules.config import AbstractNoiseAndClipScheduleConfig
from policy.schedules.factory import schedule_factory
from policy.stateful_schedules.config import AbstractStatefulScheduleConfig
from policy.stateful_schedules.factory import stateful_schedule_factory
from privacy.gdp_privacy import GDPPrivacyParameters


def policy_factory(conf, privacy_params: GDPPrivacyParameters):
    """Instantiate the correct schedule (stateful or non-stateful) from a config.

    Args:
        conf: An `AbstractNoiseAndClipScheduleConfig` or `AbstractStatefulScheduleConfig`.
        privacy_params: GDP privacy budget and subsampling parameters.

    Returns:
        The constructed schedule object.

    Raises:
        ValueError: If `conf` is neither a schedule nor a stateful schedule config.
    """
    if isinstance(conf, AbstractNoiseAndClipScheduleConfig):
        return schedule_factory(conf, privacy_params)
    if isinstance(conf, AbstractStatefulScheduleConfig):
        return stateful_schedule_factory(conf, privacy_params)
    raise ValueError(
        f"Config type '{type(conf).__name__}' is neither a schedule nor a "
        f"stateful schedule config.",
    )
