"""Candidate space for *tuned* policy transfer (ADR 0024).

Direct transfer borrows a source's schedule verbatim and adapts nothing to the target
but ``seat_on_budget``'s single budget-binding scalar. Tuned transfer treats the
schedule's remaining free parameters as hyperparameters to search **on the target**,
the same way the native references in ``transfer_reference.py`` are already swept —
which is what makes the two arms of the matrix comparable.

Two knobs, in two nested stages:

*Stage A* — a joint ``(sigma, clip)`` **scale**, available to any transferred schedule
(curve or equation). :func:`apply_scale`.

*Stage B* — the distilled template's per-condition **shape constants**, available to
the equation arm only, which is the tunability an equation uniquely adds over a
resampled curve. :func:`constant_candidates`.

Kept numpy/pandas-only, no jax: ``transfer_launch.py`` sizes the candidate arrays from
here and must not drag a GPU framework into an off-cluster launcher (see that module's
docstring). The one place that would have needed jax — deciding whether a perturbed
constant actually moves the seated schedule — is instead settled analytically by
:func:`seats_identically`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np


@dataclasses.dataclass(frozen=True, order=True)
class Knobs:
    """One tuning candidate: what a transferred schedule is adapted by on the target.

    Both stages produce these, so the record/selector plumbing handles a single type.
    The constant fields are **overrides**, not full vectors: a stage-B candidate varies
    one named constant of one equation and inherits the borrowed condition's value for
    every other, which is what makes a candidate legible on the cell it wins
    (``"sigma.p2=1.34"``) rather than an opaque vector.

    Frozen and ordered so a candidate list is hashable, deduplicable and sorts into a
    stable enumeration — the index into that list *is* the SLURM task's identity.
    """

    scale: float = 1.0
    """Stage A: the joint ``(sigma, clip)`` factor. See :func:`apply_scale`."""
    sigma_constants: tuple[tuple[str, float], ...] = ()
    """Stage B: ``(name, value)`` overrides on the distilled sigma equation."""
    clip_constants: tuple[tuple[str, float], ...] = ()
    """Stage B: ``(name, value)`` overrides on the distilled clip equation."""


# ---------------------------------------------------------------------------
# Stage A: the joint (sigma, clip) scale
# ---------------------------------------------------------------------------

SCALE_GRID: tuple[float, ...] = tuple(float(2.0**k) for k in range(-6, 4))
"""The joint-scale grid stage A searches: powers of two from 1/64 to 8.

Anchored on the native reference's own search rather than guessed: a reference draws
its constant clip from U(0.1, 5.0) (``Baseline.candidate_schedules``) and seats sigma
to the budget, which is precisely this knob. Matching that support is what makes the
tuned transfer arm and the tuned reference arm comparable — the reason the reference is
in the matrix at all.

The scale is *relative* to the transferred curve, so the support it reaches depends on
the level the source learned, and the two arms are nearly a decade apart: the median
mean-clip over FirSweep's 959 runs is 5.1 at ``sgd-m0.0`` and 0.63 at ``sgd-m0.9``.
Spanning U(0.1, 5.0) from **either** level is what sets the endpoints, and it is why
the grid is asymmetric about 1 — learned clip levels sit above the reference's box more
often than below it, so the headroom that matters is downward.

Ratio-2 and log-spaced. Deterministic, so unlike the reference's uniform draw it can
simply cover the low decades instead of giving them a few percent of its samples.
Coarse on purpose: at one scoring rep per candidate the per-score noise is at or above
the 0.36pp cell-level sigma_eval, so a finer grid would only let the selector choose
between candidates it cannot actually tell apart.

``1.0`` is a grid point *exactly*, so the untuned schedule is always among the
candidates and "tuning helped" stays separable from "tuning moved things".
"""


def scale_candidates(grid: tuple[float, ...] = SCALE_GRID) -> list[Knobs]:
    """Stage A's candidate list: the joint scale alone, no constants perturbed.

    Available to **any** transferred schedule, curve as well as equation — the scale
    compensates for the target's gradient-norm regime, not for the source's shape, so
    a resampled curve has exactly the same free parameter a closed form does.
    """
    return [Knobs(scale=float(scale)) for scale in grid]


def apply_scale(sigmas, clips, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale a seated ``(sigma, clip)`` schedule by ``scale``, preserving its budget.

    The GDP budget is ``sum_i exp((C_i/sigma_i)^2)``, a function of the *multiplier*
    ``s_i = sigma_i/C_i`` alone, so multiplying both curves by the same factor is
    **exactly** privacy-neutral: a schedule seated on the boundary stays on it. That is
    what makes the scale a free parameter rather than a budget violation, and it is why
    this is applied *after* seating rather than folded into the shape.

    It is not, however, training-neutral. Under DP-PSAC the clip is a scale and not a
    ceiling (the multiplier ``C/(||g|| + 1/(||g||+1))`` is unbounded above), so the
    scale is how a transferred schedule adapts to the *target's* gradient-norm regime —
    the one thing the source shape cannot know. With the inner loop's SGD update this
    is equivalent to scaling the inner learning rate.
    """
    scale = float(scale)
    return np.asarray(sigmas, dtype=float) * scale, np.asarray(clips, dtype=float) * scale


# ---------------------------------------------------------------------------
# Stage B: the template's per-condition shape constants
# ---------------------------------------------------------------------------

CONSTANT_WIDEN = 0.5
"""How far past the fitted conditions' range each constant is swept, as a fraction of
that range's own span (per side). The target is a different dataset and may want a
constant no source condition happened to need; the validity screen catches an
overreach, so a moderate widening costs nothing but reach."""


def constant_ranges(
    constant_table: dict[str, np.ndarray], widen: float = CONSTANT_WIDEN
) -> dict[str, tuple[float, float]]:
    """The ``(low, high)`` box stage B sweeps each template constant over.

    Read off the fit — each constant's observed spread across the trained conditions
    (``_TemplatePredictor.constant_table``) — rather than guessed. That range is where
    the fitted template is *known* to produce sensible shapes, and it is literally the
    variation the template was fitted to express.

    It also handles what a relative box cannot. The K constants are heterogeneous in
    role — one may be an exponent, another a rate, another an offset — so a uniform
    percentage of each *value* means something different for each (and nothing at all
    for one that happens to sit near zero), while per-constant empirical ranges are
    automatically on the right scale.

    ``widen`` extends each range symmetrically by that fraction of its own span; see
    :data:`CONSTANT_WIDEN`.
    """
    ranges: dict[str, tuple[float, float]] = {}
    for name, values in constant_table.items():
        values = np.asarray(values, dtype=float)
        low, high = float(np.min(values)), float(np.max(values))
        margin = widen * (high - low)
        ranges[name] = (low - margin, high + margin)
    return ranges


CONSTANT_POINTS = 4
"""Points swept per template constant. Coarse for the same reason :data:`SCALE_GRID`
is: one scoring rep puts the per-candidate noise at or above the 0.36pp cell-level
sigma_eval, so a finer sweep buys resolution the selector cannot use."""


def constant_candidates(
    sigma_ranges: dict[str, tuple[float, float]],
    clip_ranges: dict[str, tuple[float, float]],
    scale: float = 1.0,
    points: int = CONSTANT_POINTS,
) -> list[Knobs]:
    """Stage B's candidate list: each template constant swept across its own range.

    **One at a time**, not a joint sample of the K-dimensional box. At this budget
    nothing could fit an interaction model anyway — a dozen points in 3-D, with a
    per-score noise at or above the 0.36pp cell-level sigma_eval — and OAT buys a
    per-constant sensitivity curve, which is the more useful result.

    **Nested inside stage A**: ``scale`` is the winning joint scale, fixed here rather
    than re-searched, because the full joint space is far too sparse at this many
    candidates. The unperturbed candidate (that scale, no constant moved) leads the
    list, so stage B's own pool contains the do-nothing option and "the shape constants
    helped" stays separable from "the scale did".

    Values are linearly spaced across each range: template constants are signed and
    routinely straddle zero, so a log grid is not available (unlike the scale, which is
    a strictly positive multiplier).
    """
    candidates = [Knobs(scale=float(scale))]
    for field, ranges in (("sigma_constants", sigma_ranges), ("clip_constants", clip_ranges)):
        for name, (low, high) in ranges.items():
            for value in np.linspace(float(low), float(high), points):
                overrides = ((str(name), float(value)),)
                candidates.append(Knobs(scale=float(scale), **{field: overrides}))
    return candidates


# ---------------------------------------------------------------------------
# Screening: which perturbations are worth a GPU-hour
# ---------------------------------------------------------------------------

Shape = tuple[np.ndarray, np.ndarray]
"""An unseated ``(f_sigma, f_clip)`` pair evaluated on the target's step grid."""


def is_evaluable(shape: Shape) -> bool:
    """Whether a candidate shape can be seated and trained at all.

    ``_TemplatePredictor`` evaluates under ``errstate(all="ignore")``, so a constant
    pushed outside the range the template was fitted over comes back as silent NaN/Inf
    rather than an exception — and an unscreened bad candidate costs a GPU-hour to
    discover. Seating computes ``exp(1/s^2)``, so both curves must additionally be
    strictly positive, not merely finite.
    """
    sigma, clip = (np.asarray(part, dtype=float) for part in shape)
    return bool(
        np.all(np.isfinite(sigma))
        and np.all(np.isfinite(clip))
        and np.all(sigma > 0.0)
        and np.all(clip > 0.0)
    )


def seats_identically(a: Shape, b: Shape, rtol: float = 1e-6) -> bool:
    """Whether two shape pairs seat onto the **same** ``(sigma, clip)`` schedule.

    ``seat_on_budget`` solves ``sum_i exp(1/(c*s_i)^2) = bound`` for a single scalar
    ``c`` on the multiplier ``s = f_sigma/f_clip``, so scaling ``f_sigma`` by ``b``
    sends ``s -> b*s`` and ``c -> c/b``, leaves ``c*s`` invariant, and returns a
    bit-identical seated curve — the invariance survives ``project_inverse_sigmas``,
    which only ever sees ``c*s``. The final schedule is then ``(seat(s)*f_clip,
    f_clip)``, so it depends on ``f_clip`` exactly and on ``f_sigma`` only up to a
    positive multiplicative constant.

    That is the whole test: same clip curve, and sigma curves that are constant
    multiples of one another. Deciding it here rather than by pushing both candidates
    through the real seater keeps this module jax-free (so the launcher can size its
    arrays from it) and avoids inventing a "did it move enough" threshold on top of the
    bisection's own 1e-6 tolerance.

    Perturbations that come back True are unidentifiable and drop out of the sweep —
    scoring one would spend a GPU-hour re-measuring a schedule already in the pool. Note
    this is *not* the stage-A scale, which moves ``f_clip`` too and so is never
    degenerate by this test.
    """
    sigma_a, clip_a = (np.asarray(part, dtype=float) for part in a)
    sigma_b, clip_b = (np.asarray(part, dtype=float) for part in b)
    if sigma_a.shape != sigma_b.shape or clip_a.shape != clip_b.shape:
        return False
    if not np.allclose(clip_a, clip_b, rtol=rtol, atol=0.0):
        return False
    with np.errstate(all="ignore"):
        ratio = sigma_b / sigma_a
    return bool(np.all(np.isfinite(ratio)) and np.allclose(ratio, ratio.flat[0], rtol=rtol))


def screen_candidates(candidates: list[Knobs], evaluate: Callable[[Knobs], Shape]) -> list[Knobs]:
    """The candidates worth submitting: evaluable, and distinct from one another.

    Runs entirely CPU-side at generation time — the whole point, since every candidate
    that survives to the cluster costs a GPU-hour whether or not it turns out to be NaN
    or a duplicate of one already in the pool.

    ``evaluate`` maps a candidate to the ``(f_sigma, f_clip)`` it would be seated from,
    and is the caller's business: the equation producer evaluates two distilled shapes,
    the curve producer scales one resampled pair. Passing it in keeps this module free
    of predictor loading — and jax-free, so the launcher can size its arrays from the
    same screened list the producer will index into.

    A candidate is dropped when it is not :func:`is_evaluable`, or when it
    :func:`seats_identically` to one already kept. Order is preserved and the first
    candidate always survives, so an unperturbed base leading the list stays the
    reference point every later candidate is compared against.
    """
    kept: list[Knobs] = []
    shapes: list[Shape] = []
    for knobs in candidates:
        shape = evaluate(knobs)
        if not is_evaluable(shape):
            continue
        if any(seats_identically(seen, shape) for seen in shapes):
            continue
        kept.append(knobs)
        shapes.append(shape)
    return kept
