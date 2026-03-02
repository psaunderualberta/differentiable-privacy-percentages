"""Public factory for stateful noise-and-clip schedules.

Importing this module triggers registration of all concrete stateful schedule
classes via their @register decorators.
"""

# Import triggers @register side-effect.
from policy.stateful_schedules import median_gradient  # noqa: F401
from policy.stateful_schedules._registry import build as stateful_schedule_factory

__all__ = ["stateful_schedule_factory"]
