"""Registration decorator and factory for base schedules (ConstantSchedule, etc.).

Each concrete base schedule class registers itself with @register(ConfigClass).
The factory build() dispatches to the correct class without an explicit if-chain.
"""

from __future__ import annotations

_REGISTRY: dict[type, type] = {}


def register(config_cls: type):
    """Class decorator — register a base schedule class against its config type.

    Usage::

        @register(ConstantScheduleConfig)
        class ConstantSchedule(AbstractSchedule): ...
    """

    def decorator(schedule_cls: type) -> type:
        _REGISTRY[config_cls] = schedule_cls
        return schedule_cls

    return decorator


def build(conf, T: int):
    """Construct the base schedule corresponding to *conf*.

    Args:
        conf: A concrete ``AbstractScheduleConfig`` instance.
        T:    Number of DP-SGD training steps (``num_training_steps``).
    """
    cls = _REGISTRY.get(type(conf))
    if cls is None:
        known = [c.__name__ for c in _REGISTRY]
        raise ValueError(
            f"No base schedule registered for config type '{type(conf).__name__}'. "
            f"Known types: {known}",
        )
    return cls.from_config(conf, T)
