"""Registry for optax-derived optimizer config classes.

Mirrors networks/_registry.py.  Populated at import time by
``make_optimizer_config`` so ``_get_config_classes()`` in singleton_conf can
reconstruct the correct Union variant when restoring from a W&B run config.
"""

from __future__ import annotations

_REGISTRY: dict[str, type] = {}


def register(cls: type) -> type:
    _REGISTRY[cls.__name__] = cls
    return cls
