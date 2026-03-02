"""Public factory for base schedules.

Importing this module triggers registration of all concrete base schedule
classes via their @register decorators.
"""

# Imports trigger @register side-effects — order does not matter.
from policy.base_schedules import clipped, constant, exponential  # noqa: F401
from policy.base_schedules._registry import build as base_schedule_factory

__all__ = ["base_schedule_factory"]
