"""Tuned transfer: the free constants of a transferred schedule are searched on the
target rather than borrowed verbatim from the source condition (ADR 0024)."""

import numpy as np
import pandas as pd

from privacy.gdp_privacy import GDPPrivacyParameters
from symbolic_regression_eval import _TemplatePredictor
from transfer_tuning import (
    Knobs,
    apply_scale,
    constant_candidates,
    constant_ranges,
    is_evaluable,
    scale_candidates,
    screen_candidates,
    seats_identically,
)
from util.transfer import seat_on_budget


def _spend(sigmas: np.ndarray, clips: np.ndarray) -> float:
    """The GDP budget a (sigma, clip) schedule consumes: ``sum_i exp((C_i/sigma_i)^2)``."""
    return float(np.sum(np.exp((np.asarray(clips) / np.asarray(sigmas)) ** 2)))


def _seated(pp: GDPPrivacyParameters, clips: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    """The raw sigma curve that seats ``multipliers`` onto ``pp``'s budget."""
    return np.asarray(seat_on_budget(multipliers, pp)) * clips


def _predictor(expr: str, **params: list[float]) -> _TemplatePredictor:
    """A template predictor for shape ``f`` with per-condition constants ``p1..pK``.

    In ``f``, ``#1`` is step_norm and ``#(k+1)`` is that condition's ``pk``; each
    ``pk`` list carries the constant for every 1-indexed category.
    """
    ordered = sorted(params.items(), key=lambda kv: int(kv[0][1:]))
    tail = "; ".join(f"{name} = [{', '.join(map(str, values))}]" for name, values in ordered)
    equations = pd.DataFrame({"equation": [f"f = {expr}; {tail}"], "selected": [True]})
    return _TemplatePredictor(equations, ["step_norm", "category"])


class TestConditionConstantsAreReadable:
    """Tuning starts from the condition's own fitted constants, so the borrowed
    vector has to be legible before it can be varied."""

    def test_returns_the_named_constants_of_one_condition(self):
        predictor = _predictor("#1 * #2 + #3", p1=[10.0, 20.0], p2=[0.5, 0.25])

        assert predictor.constants(category=1) == {"p1": 10.0, "p2": 0.5}
        assert predictor.constants(category=2) == {"p1": 20.0, "p2": 0.25}


class TestShapeEvaluatedUnderExplicitConstants:
    """The tuning primitive: the distilled *shape* is universal but its constants are
    free, so transfer must be able to evaluate ``f`` at a constant vector that belongs
    to no trained condition (ADR 0024). Today's producer can only borrow one verbatim."""

    def test_reproduces_predict_when_given_the_conditions_own_constants(self):
        predictor = _predictor("#1 * #2 + #3", p1=[10.0, 20.0], p2=[0.5, 0.25])
        step_norm = np.linspace(0.0, 1.0, 16)

        borrowed = predictor.predict(np.column_stack([step_norm, np.full(16, 2)]))
        explicit = predictor.predict_with_constants(step_norm, predictor.constants(category=2))

        assert np.allclose(explicit, borrowed)

    def test_an_off_condition_constant_vector_gives_a_different_shape(self):
        predictor = _predictor("#1 * #2 + #3", p1=[10.0, 20.0], p2=[0.5, 0.25])
        step_norm = np.linspace(0.0, 1.0, 16)

        tuned = predictor.predict_with_constants(step_norm, {"p1": 15.0, "p2": 0.5})

        # f = step*15 + 0.5 — a vector no trained condition carries.
        assert np.allclose(tuned, step_norm * 15.0 + 0.5)


class TestJointScaleIsPrivacyNeutral:
    """Stage A's knob. Scaling sigma and clip together leaves the multiplier
    ``s = sigma/clip`` — and therefore the GDP spend — exactly unchanged, so the scale
    is a free parameter the target may tune without touching its budget. What it does
    change is training: under DP-PSAC the clip is a scale, not a ceiling, so this is
    the transferred schedule's grip on the *target's* gradient-norm regime."""

    def test_scaling_moves_the_clip_level_but_not_the_budget_spend(self):
        pp = GDPPrivacyParameters(eps=1.0, delta=1e-5, p=0.01, T=8)
        clips = np.linspace(1.0, 2.0, 8)
        sigmas = _seated(pp, clips, np.linspace(0.8, 1.2, 8))

        for scale in (0.125, 4.0):
            scaled_sigmas, scaled_clips = apply_scale(sigmas, clips, scale)

            # The schedule really moved — a scale of 1/8 is a different training run.
            assert np.allclose(scaled_clips, clips * scale)
            assert not np.allclose(scaled_sigmas, sigmas)
            # ...but it spends the identical budget.
            assert np.isclose(_spend(scaled_sigmas, scaled_clips), _spend(sigmas, clips))

    def test_scaling_commutes_with_seating(self):
        """Scaling before or after ``seat_on_budget`` gives the same schedule, because
        seating is a function of the multiplier alone. The producers apply the scale
        *after* seating, where the tuned schedule is the obvious thing being scaled;
        the candidate screen applies it *before*, to the raw shapes it has in hand.
        Those must agree, or the screen would be filtering a different schedule than
        the one that gets trained."""
        pp = GDPPrivacyParameters(eps=1.0, delta=1e-5, p=0.01, T=8)
        clips, sigma_shape = np.linspace(1.0, 2.0, 8), np.linspace(0.9, 1.4, 8)

        for scale in (0.125, 4.0):
            after = apply_scale(_seated(pp, clips, sigma_shape / clips), clips, scale)

            pre_sigma, pre_clips = apply_scale(sigma_shape, clips, scale)
            before = (_seated(pp, pre_clips, pre_sigma / pre_clips), pre_clips)

            assert np.allclose(after[0], before[0], rtol=1e-5)
            assert np.allclose(after[1], before[1])


class TestScaleGridBracketsTheReferenceSweep:
    """The grid is anchored on the native reference's search, not guessed. A tuned
    transfer arm is only comparable to the tuned reference arm if the two search the
    same knob over the same support, and ``Baseline.candidate_schedules`` draws the
    reference's constant clip from U(0.1, 5.0)."""

    def test_is_a_ratio_two_grid_containing_the_untuned_schedule(self):
        scales = [knobs.scale for knobs in scale_candidates()]

        # 1.0 is in the grid *exactly*, so "tuning helped" stays separable from
        # "tuning moved things" — the untuned schedule is one of the candidates.
        assert 1.0 in scales
        # Log spacing, not the reference's uniform draw: a deterministic grid can just
        # cover the range, and uniform would give the 0.1-1 decade only ~18% of it.
        assert np.allclose(np.diff(np.log2(scales)), 1.0)

    def test_reaches_the_reference_clip_support_from_either_arms_learned_level(self):
        scales = [knobs.scale for knobs in scale_candidates()]

        # The scale multiplies the *transferred* clip curve, whose level is whatever
        # the source learned — and the two arms sit almost a decade apart (median
        # clip 5.1 at sgd-m0.0, 0.63 at sgd-m0.9, measured on FirSweep). So the grid
        # has to span the reference's U(0.1, 5.0) box seen from either level, which
        # is why it is asymmetric about 1: learned clips sit above that box more
        # often than below it, so the headroom that matters is downward.
        for learned_clip_level in (5.1, 0.63):
            reachable = [learned_clip_level * scale for scale in scales]
            assert min(reachable) <= 0.1
            assert max(reachable) >= 5.0

    def test_a_scale_only_candidate_perturbs_no_constants(self):
        # Stage A is available to any transferred schedule, curve included, so its
        # candidates must not carry equation-only knobs.
        assert all(
            not knobs.sigma_constants and not knobs.clip_constants for knobs in scale_candidates()
        )


class TestDegenerateCandidatesSeatOntoTheSameSchedule:
    """Not every constant is a knob. ``seat_on_budget`` binds the curve by solving on
    ``c * s`` where ``s = sigma/clip``, so the schedule it produces is a function of
    the clip shape *exactly* and the sigma shape only *up to a multiplicative
    constant*. A perturbation that just rescales the sigma equation is therefore an
    exact no-op, and burning a GPU-hour to score it would be pure waste."""

    def test_a_rescaled_sigma_shape_is_the_same_candidate(self):
        step = np.linspace(0.0, 1.0, 32)
        clip = 1.0 + step
        sigma = 2.0 - step

        assert seats_identically((sigma, clip), (7.5 * sigma, clip))

    def test_a_genuinely_different_sigma_shape_is_not(self):
        step = np.linspace(0.0, 1.0, 32)
        clip = 1.0 + step

        assert not seats_identically((2.0 - step, clip), (2.0 - step**2, clip))

    def test_a_changed_clip_shape_is_not_even_at_the_same_sigma(self):
        step = np.linspace(0.0, 1.0, 32)
        sigma = 2.0 - step

        # The clip enters the seated schedule directly, not just through the ratio.
        assert not seats_identically((sigma, 1.0 + step), (sigma, 1.0 + 2.0 * step))

    def test_agrees_with_what_seat_on_budget_actually_does(self):
        """The screen replaces pushing every perturbation through the real seater
        (which needs jax, and a tolerance to decide 'did it move'). It has to give the
        same answer as the thing it replaces."""
        pp = GDPPrivacyParameters(eps=1.0, delta=1e-5, p=0.01, T=16)
        step = np.linspace(0.0, 1.0, 16)
        clip, sigma = 1.0 + step, 2.0 - step
        rescaled, reshaped = 7.5 * sigma, 2.0 - step**2

        base = _seated(pp, clip, sigma / clip)

        assert seats_identically((sigma, clip), (rescaled, clip))
        assert np.allclose(base, _seated(pp, clip, rescaled / clip), rtol=1e-5)
        assert not seats_identically((sigma, clip), (reshaped, clip))
        assert not np.allclose(base, _seated(pp, clip, reshaped / clip), rtol=1e-5)


class TestInvalidCandidatesAreRejectedBeforeSubmission:
    """``_TemplatePredictor`` evaluates under ``errstate(all="ignore")``, so a constant
    pushed outside the range the template was fitted over returns silent NaN/Inf rather
    than raising. Each such candidate that reaches the cluster costs a GPU-hour to
    discover, so they are screened CPU-side at generation time."""

    def test_rejects_non_finite_shapes(self):
        step = np.linspace(0.0, 1.0, 8)
        good = 1.0 + step

        assert is_evaluable((good, good))
        assert not is_evaluable((np.full(8, np.nan), good))
        assert not is_evaluable((good, np.full(8, np.inf)))

    def test_rejects_non_positive_shapes(self):
        # Seating computes exp(1/s**2) on s = sigma/clip, so finite is not enough:
        # a zero or negative sigma has no seated schedule at all.
        step = np.linspace(0.0, 1.0, 8)
        good = 1.0 + step

        assert not is_evaluable((step, good))  # sigma starts at exactly 0
        assert not is_evaluable((good, step - 0.5))  # clip goes negative


class TestConstantRangesComeFromTheFittedConditions:
    """Stage B's box is read off the fit, not guessed. Each constant's observed range
    across the trained conditions is the range over which the template is *known* to
    produce sensible shapes — literally the variation it was fitted to express. It also
    handles what a relative box cannot: the constants are heterogeneous in role
    (exponent vs rate vs offset), so a uniform 50% widening of each *value* would be
    meaningless across them, while per-constant empirical ranges are automatically on
    the right scale."""

    def test_each_constant_spans_the_values_its_conditions_carry(self):
        predictor = _predictor("#1 * #2 + #3", p1=[10.0, 30.0, 20.0], p2=[0.5, 0.25, -1.0])

        ranges = constant_ranges(predictor.constant_table(), widen=0.0)

        # Per-constant, and on each one's own scale — not one shared box.
        assert ranges == {"p1": (10.0, 30.0), "p2": (-1.0, 0.5)}

    def test_widening_extends_each_range_by_a_fraction_of_its_own_span(self):
        predictor = _predictor("#1 * #2 + #3", p1=[10.0, 30.0, 20.0], p2=[0.5, 0.25, -1.0])

        ranges = constant_ranges(predictor.constant_table(), widen=0.5)

        # The target is a different dataset and may need a constant no source
        # condition happened to need; the validity screen catches an overreach.
        assert np.allclose(ranges["p1"], (0.0, 40.0))  # span 20 -> +/- 10
        assert np.allclose(ranges["p2"], (-1.75, 1.25))  # span 1.5 -> +/- 0.75


class TestShapeConstantsAreSweptOneAtATime:
    """Stage B varies one constant at a time rather than sampling the joint box. At
    ~12 points and a per-score noise at or above the 0.36pp cell-level sigma_eval,
    nothing could fit an interaction model anyway — and OAT yields a per-constant
    sensitivity curve, which is the more useful result. Stage B runs nested inside
    stage A: the winning scale is fixed, not re-searched."""

    def test_every_candidate_perturbs_exactly_one_constant_of_one_equation(self):
        candidates = constant_candidates(
            sigma_ranges={"p1": (0.0, 1.0), "p2": (-2.0, 2.0)},
            clip_ranges={"p1": (5.0, 6.0)},
            scale=0.25,
            points=3,
        )

        perturbed = [len(k.sigma_constants) + len(k.clip_constants) for k in candidates]
        assert set(perturbed) <= {0, 1}
        # Everything not named is inherited from the borrowed condition's vector, so
        # an override list of length one is exactly a one-at-a-time perturbation.
        assert sum(n == 1 for n in perturbed) == 3 * 3  # 3 constants x 3 points

    def test_carries_the_winning_stage_a_scale_into_every_candidate(self):
        candidates = constant_candidates(
            sigma_ranges={"p1": (0.0, 1.0)}, clip_ranges={}, scale=0.25, points=3
        )

        assert {knobs.scale for knobs in candidates} == {0.25}

    def test_includes_the_unperturbed_candidate_so_shape_tuning_is_falsifiable(self):
        candidates = constant_candidates(
            sigma_ranges={"p1": (0.0, 1.0)}, clip_ranges={}, scale=0.25, points=3
        )

        # The stage-A winner itself is in stage B's pool, scored under the same
        # protocol, so "the shape constants helped" is separable from "the scale did".
        assert Knobs(scale=0.25) in candidates

    def test_sweeps_each_constant_across_its_own_range(self):
        candidates = constant_candidates(
            sigma_ranges={"p1": (-2.0, 2.0)}, clip_ranges={}, scale=1.0, points=5
        )

        swept = sorted(value for knobs in candidates for _, value in knobs.sigma_constants)
        assert np.allclose(swept, [-2.0, -1.0, 0.0, 1.0, 2.0])

    def test_names_which_equation_a_constant_belongs_to(self):
        # sigma's p1 and clip's p1 are different numbers in different equations; a
        # candidate that could not say which it moved would be unreproducible.
        candidates = constant_candidates(
            sigma_ranges={"p1": (0.0, 1.0)}, clip_ranges={"p1": (5.0, 6.0)}, points=2
        )

        assert ("p1", 0.0) in {c for knobs in candidates for c in knobs.sigma_constants}
        assert ("p1", 5.0) in {c for knobs in candidates for c in knobs.clip_constants}


class TestScreeningDropsWhatNeedNotBeScored:
    """Both filters run CPU-side at generation time, before anything is submitted: an
    unscreened candidate costs a GPU-hour to discover it was NaN or a duplicate."""

    def test_drops_a_constant_that_never_moves_the_seated_schedule(self):
        step = np.linspace(0.0, 1.0, 16)
        base = Knobs()
        # p1 enters the sigma equation only as a prefactor, so every value of it
        # seats onto the identical schedule — the constant is unidentifiable here.
        prefactor = [Knobs(sigma_constants=(("p1", v),)) for v in (2.0, 3.0, 4.0)]
        real = Knobs(sigma_constants=(("p2", 1.0),))

        def evaluate(knobs: Knobs):
            p1 = dict(knobs.sigma_constants).get("p1", 1.0)
            p2 = dict(knobs.sigma_constants).get("p2", 0.0)
            return (p1 * (2.0 - step + p2 * step**2), 1.0 + step)

        kept = screen_candidates([base, *prefactor, real], evaluate)

        assert kept == [base, real]

    def test_drops_candidates_that_evaluate_to_nothing_trainable(self):
        step = np.linspace(0.0, 1.0, 16)
        base, broken, good = (
            Knobs(),
            Knobs(sigma_constants=(("p1", -1.0),)),
            Knobs(sigma_constants=(("p1", 2.0),)),
        )

        def evaluate(knobs: Knobs):
            # p1 tilts the sigma curve. At p1=-1 it reaches zero: finite, but there is
            # no seated schedule for it, since seating computes exp(1/s**2).
            return (1.0 + dict(knobs.sigma_constants).get("p1", 1.0) * step, 1.0 + step)

        assert screen_candidates([base, broken, good], evaluate) == [base, good]

    def test_the_scale_is_never_screened_out_as_degenerate(self):
        step = np.linspace(0.0, 1.0, 16)
        candidates = scale_candidates()

        # apply_scale moves the clip curve too, so a scaled schedule is a genuinely
        # different one — the degeneracy filter must not mistake it for the base.
        def evaluate(knobs: Knobs):
            return apply_scale(2.0 - step, 1.0 + step, knobs.scale)

        assert screen_candidates(candidates, evaluate) == candidates
