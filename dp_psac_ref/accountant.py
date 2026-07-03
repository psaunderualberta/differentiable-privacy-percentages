"""Privacy accountant: thin wrapper over Opacus RDPAccountant.

Supports per-step sigma schedules (one step() call per iteration).
"""

from __future__ import annotations

import numpy as np
from opacus.accountants import RDPAccountant


def epsilon_spent(sigmas: np.ndarray, sample_rate: float, delta: float) -> float:
    acc = RDPAccountant()
    for s in np.asarray(sigmas).tolist():
        acc.step(noise_multiplier=float(s), sample_rate=float(sample_rate))
    return float(acc.get_epsilon(delta=delta))
