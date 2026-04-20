"""Build a constant (sigmas, clips) schedule calibrated to a target (eps, delta).

Usage:
    uv run calibrate.py --target-eps 3 --delta 1e-5 --N 60000 --B 512 --T 2000 \\
                        --clip 0.1 --out-sigmas sigmas.npy --out-clips clips.npy

Uses Opacus's search to pick the constant noise multiplier sigma such that the
RDP accountant reports exactly (target_eps, delta) after T Poisson-subsampled
steps at sample_rate B/N. Writes two length-T .npy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
from opacus.accountants.utils import get_noise_multiplier


@dataclass
class Args:
    target_eps: float
    delta: float = 1e-5
    N: int = 60000
    B: int = 512
    T: int = 2000
    clip: float = 0.1
    out_sigmas: Path = Path("sigmas.npy")
    out_clips: Path = Path("clips.npy")


def main(a: Args) -> None:
    q = a.B / a.N
    sigma = get_noise_multiplier(
        target_epsilon=a.target_eps,
        target_delta=a.delta,
        sample_rate=q,
        steps=a.T,
        accountant="rdp",
    )
    sigmas = np.full(a.T, sigma, dtype=np.float32)
    clips = np.full(a.T, a.clip, dtype=np.float32)
    np.save(a.out_sigmas, sigmas)
    np.save(a.out_clips, clips)
    print(
        f"calibrated constant schedule: sigma={sigma:.4f} clip={a.clip} T={a.T} "
        f"q={q:.6f}  (eps={a.target_eps}, delta={a.delta})"
    )
    print(f"wrote {a.out_sigmas} and {a.out_clips}")


if __name__ == "__main__":
    main(tyro.cli(Args))
