"""Registration decorator and factory for noise-and-clip schedules.

Each concrete schedule class registers itself with @register(ConfigClass).
The factory build() dispatches to the correct class without an explicit if-chain.
"""

from __future__ import annotations

from privacy.gdp_privacy import GDPPrivacyParameters

_REGISTRY: dict[type, type] = {}


def register(config_cls: type):
    """Class decorator — register a schedule class against its config type.

    Usage::

        @register(AlternatingSigmaAndClipScheduleConfig)
        class AlternatingSigmaAndClipSchedule(AbstractNoiseAndClipSchedule):
            ...
    """

    def decorator(schedule_cls: type) -> type:
        _REGISTRY[config_cls] = schedule_cls
        return schedule_cls

    return decorator


def build(conf, privacy_params: GDPPrivacyParameters):
    """Construct the schedule corresponding to *conf*.

    Args:
        conf:           A concrete schedule config instance.
        privacy_params: GDP privacy parameters for the experiment.
    """
    cls = _REGISTRY.get(type(conf))
    if cls is None:
        known = [c.__name__ for c in _REGISTRY]
        raise ValueError(
            f"No schedule registered for config type '{type(conf).__name__}'. "
            f"Known types: {known}"
        )
    return cls.from_config(conf, privacy_params)
