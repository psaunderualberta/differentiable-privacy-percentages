"""Privacy accountant for the standalone DP-PSAC reference.

Independent re-implementation (via ``autodp``) of the *same* privacy model the
``src`` outer-loop budget uses, so the reference's reported epsilon is directly
comparable to the target the schedule was calibrated against:

    * fixed-size sampling **without replacement** (WOR) amplification, and
    * **replace-one** (substitution) adjacency, i.e. per-step Gaussian
      sensitivity ``2C`` — the added noise has std ``C * sigma_mult``, so the
      sensitivity-normalised Gaussian parameter is ``sigma_mult / 2``.

This mirrors ``src/privacy/rdp_accountant.py`` (base RDP ``2*alpha*w**2`` +
Wang-Balle-Kasiviswanathan WOR amplification) but is written against a
third-party library (``autodp``), so it is a genuine independent check rather
than a call back into ``src``. ``autodp`` optimises the RDP->(eps, delta)
conversion over a finer Renyi-order grid than ``src``'s fixed integer set, so it
typically returns a slightly *tighter* (smaller) epsilon for the same schedule.

Note: ``dp_psac.py`` samples *with replacement* (``jr.randint``), which matches
neither WOR nor Poisson exactly; WOR is used here because it is the model the
schedule's budget was defined under.
"""

from __future__ import annotations

import numpy as np
from autodp import mechanism_zoo, transformer_zoo


def epsilon_spent(
    noise_multipliers: np.ndarray,
    sample_rate: float,
    delta: float,
    sensitivity_factor: float = 2.0,
) -> float:
    """Realised ``eps(delta)`` for a per-step DP-SGD noise-multiplier schedule.

    Args:
        noise_multipliers: Per-step ``sigma / clip`` (noise std over clip norm),
            shape ``(T,)``.
        sample_rate: Subsampling ratio ``q = batch_size / n``.
        delta: Target delta.
        sensitivity_factor: Adjacency sensitivity in units of the clip norm ``C``.
            ``2.0`` (default) is replace-one / substitution (matches the ``src``
            budget); ``1.0`` is add/remove.

    Returns:
        Scalar ``eps(delta)`` under WOR-amplified, replace-one Gaussian composition.
    """
    mults = np.asarray(noise_multipliers, dtype=float).ravel()
    subsample = transformer_zoo.AmplificationBySampling(PoissonSampling=False)
    compose = transformer_zoo.Composition()

    mechanisms = []
    for m in mults.tolist():
        # sigma param = noise_std / sensitivity = (C * m) / (sensitivity_factor * C).
        gaussian = mechanism_zoo.GaussianMechanism(sigma=m / sensitivity_factor)
        # WOR amplification lemma requires a replace-one base mechanism.
        gaussian.neighboring = "replace_one"
        mechanisms.append(subsample.amplify(gaussian, sample_rate, improved_bound_flag=True))

    composed = compose(mechanisms, [1] * len(mechanisms))
    return float(composed.get_approxDP(delta))


def _constant_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    delta: float,
    steps: int,
    sensitivity_factor: float,
) -> float:
    """``eps(delta)`` for a *constant* noise-multiplier schedule of length ``steps``.

    Fast path: composes a single amplified mechanism ``steps`` times (integer
    coefficient) instead of building ``steps`` identical mechanisms.
    """
    gaussian = mechanism_zoo.GaussianMechanism(sigma=noise_multiplier / sensitivity_factor)
    gaussian.neighboring = "replace_one"
    subsample = transformer_zoo.AmplificationBySampling(PoissonSampling=False)
    amplified = subsample.amplify(gaussian, sample_rate, improved_bound_flag=True)
    composed = transformer_zoo.Composition()([amplified], [steps])
    return float(composed.get_approxDP(delta))


def calibrate_noise_multiplier(
    target_epsilon: float,
    sample_rate: float,
    delta: float,
    steps: int,
    sensitivity_factor: float = 2.0,
    tol: float = 1e-4,
) -> float:
    """Constant noise multiplier ``sigma / clip`` hitting ``target_epsilon`` at ``delta``.

    Calibrated under the *same* WOR + replace-one (2C) model as ``epsilon_spent``,
    so a schedule built from this multiplier re-accounts back to ``target_epsilon``
    (up to ``tol``). Replaces Opacus's ``get_noise_multiplier`` (which calibrates
    under Poisson + add/remove and would leave the reported epsilon off by the
    model gap). ``eps(delta)`` is monotone decreasing in the multiplier, so a
    bisection on ``[lo, hi]`` converges.
    """

    def eps(m: float) -> float:
        return _constant_epsilon(m, sample_rate, delta, steps, sensitivity_factor)

    lo, hi = 1e-3, 1.0
    while eps(hi) > target_epsilon:  # grow until over-noised enough (eps below target)
        lo, hi = hi, hi * 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if eps(mid) > target_epsilon:  # still under-noised -> need larger multiplier
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)
