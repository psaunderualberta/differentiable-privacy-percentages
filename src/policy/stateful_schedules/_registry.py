"""Registration decorator and factory for stateful noise-and-clip schedules."""

from __future__ import annotations

from privacy.gdp_privacy import GDPPrivacyParameters

_REGISTRY: dict[type, type] = {}


def register(config_cls: type):
    """Class decorator — register a stateful schedule class against its config type."""

    def decorator(schedule_cls: type) -> type:
        _REGISTRY[config_cls] = schedule_cls
        return schedule_cls

    return decorator


def build(conf, privacy_params: GDPPrivacyParameters):
    """Construct the stateful schedule corresponding to *conf*."""
    cls = _REGISTRY.get(type(conf))
    if cls is None:
        known = [c.__name__ for c in _REGISTRY]
        raise ValueError(
            f"No stateful schedule registered for config type '{type(conf).__name__}'. "
            f"Known types: {known}",
        )
    return cls.from_config(conf, privacy_params)
