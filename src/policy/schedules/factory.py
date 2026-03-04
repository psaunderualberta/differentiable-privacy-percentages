"""Public factory for noise-and-clip schedules.

Importing this module triggers registration of all concrete schedule classes
via their @register decorators.
"""

# Imports trigger @register side-effects — order does not matter.
from policy.schedules import (  # noqa: F401
    alternating,
    dynamic_dpsgd,
    policy_and_clip,
    sigma_and_clip,
    warmup_alternating,
)
from policy.schedules._registry import build as schedule_factory

__all__ = ["schedule_factory"]
