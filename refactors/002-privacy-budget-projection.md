# RFC 002: Separate Privacy Accounting from Constraint Projection

**Status:** Proposed
**Date:** 2026-06-10
**Scope:** `src/privacy/gdp_privacy.py`, all nine schedule classes in `src/policy/schedules/` and `src/policy/stateful_schedules/`, `src/util/baselines.py`, `src/test/test_privacy.py`, `src/test/test_property_schedules.py`

## Problem

`GDPPrivacyParameters` (`src/privacy/gdp_privacy.py`, ~600 lines) mixes three distinct responsibilities behind one class:

1. **Budget accounting** — (ε, δ) → GDP μ via scipy Brent at construction, the per-step
   μ₀, expenditure computation.
2. **Differentiable representation conversions** — weights ↔ μ-schedule ↔ σ ↔ clip,
   called inside the gradient path by schedules' `get_private_noise_scales()`.
3. **Three pure-JAX projection solvers** — `project_weights` (nested bisection +
   Newton), `project_inverse_sigmas`, and `project_sigma_and_clip` (dual bisection +
   overflow-hardened 2D Newton, the module-level `_sc_*` helpers). All three enforce
   the *same* constraint `Σᵢ exp(μᵢ²) ≤ (μ/p)² + T` in different parametrizations,
   post-gradient-update, outside the grad path.

The shallowness shows up on the **caller side**: nine schedule classes hold a
`privacy_params` field and each picks methods à la carte, then recites the same
boilerplate — call a projector, rebuild the schedule via a full constructor that
enumerates every field, `squeeze()` the logging weights. `dynamic_dpsgd.py` even
re-implements its own μ₀ root-find in scipy because the built-in one doesn't fit its
geometric-decay parametrization. `SingletonConfig` is read *inside* the privacy module
(`max_sigma` in `sigma_schedule_to_weights` and `compute_eps`), so the module cannot be
unit-tested without global config setup.

Integration risk in the seams: `test_privacy.py` (791 lines) tests the solvers in
isolation, but nothing asserts the invariant that actually matters — *any schedule's
`project()` output is on-budget* — and nothing tests the schedule-`project()` →
solver → constructor-rebuild round trip. The constraint constant `(μ/p)² + T` is
re-derived independently inside each solver.

## Proposed Interface

The hybrid design: Design C's two-layer structure as the backbone, plus Design A's
ray-calibration, plus regression-pinning during migration.

### Layer 1: `src/privacy/projections.py` — stateless solvers

Pure JAX, `while_loop`/`fori_loop`-based, parametrized only by the explicit bound. No
object state, no config, no equinox modules — trivially property-testable.

```python
def project_sigma_and_clip(sigmas: Array, clips: Array, bound: Array) -> tuple[Array, Array]:
    """L2-project onto sum_i exp((clip_i/sigma_i)^2) <= bound. Feasible inputs pass through."""

def project_weights(weights: Array, bound: Array, tol: float | Array = 1e-6) -> Array:
    """Project onto sum_i exp(w_i^2) <= bound (nested bisection + Newton)."""

def project_inverse_sigmas(sigmas: Array, bound: Array, tol: float | Array = 1e-6) -> Array:
    """Project onto sum_i exp(1/sigma_i) <= bound (exotic one-off, kept as-is)."""

def calibrate_shape(shape: Array, bound: Array) -> Array:
    """Scale a fixed schedule shape so it exactly exhausts the budget:
    find scalar s s.t. sum_i exp((s * shape_i)^2) == bound; return s * shape.
    JAX bisection — jit-safe. Replaces dynamic_dpsgd's private scipy __find_mu_0."""
```

The `_sc_*` overflow-hardened helpers move here verbatim. The existing
`project_sigma_and_clip_given_bound` *is* this layer's signature already — this
generalizes that precedent to all solvers.

### Layer 2: `src/privacy/budget.py` — the module schedules hold

```python
class PrivacyBudget(eqx.Module):
    """GDP budget: accounting + differentiable conversions + projection entry points.

    Budget scalars are exposed only through stop_gradient properties.
    Valid eqx.Module field on any schedule (pytree/checkpoint/shard_map safe).
    """

    # ---- construction (scipy allowed here ONLY; config injected, never read) ----
    @classmethod
    def from_approx_dp(cls, eps: float, delta: float, p: float, T: int,
                       max_sigma: float) -> "PrivacyBudget": ...

    # ---- accounting (read-only, stop_gradient'd) ----
    mu: Array; p: Array; mu_0: Array; max_sigma: Array; T: int
    @property
    def bound(self) -> Array: ...          # (mu/p)^2 + T — computed in ONE place
    def expenditure(self, sigmas: Array, clips: Array) -> Array: ...
    def is_within_budget(self, sigmas: Array, clips: Array, *, rtol: float = 1e-4) -> Array:
        """Scalar bool — the one-call test invariant."""

    # ---- THE common projection ----
    def project(self, sigmas: Array, clips: Array) -> tuple[Array, Array]: ...
    def project_pair(self, noise_sched: AbstractSchedule, clip_sched: AbstractSchedule
                     ) -> tuple[AbstractSchedule, AbstractSchedule]:
        """project() + from_projection() rebuild of both base schedules, absorbed."""

    # ---- differentiable conversions (grad-path safe) ----
    def weights_to_mus(self, weights: Array) -> Array: ...
    def mus_to_weights(self, mus: Array) -> Array: ...
    def weights_to_sigmas(self, C: Array, weights: Array) -> Array: ...
    def weights_to_clips(self, sigmas: Array, weights: Array) -> Array: ...
    def sigmas_to_weights(self, C: Array, sigmas: Array) -> Array: ...   # uses self.max_sigma

    # ---- exotic projections + calibration (thin delegates) ----
    def project_weights(self, weights: Array, tol=1e-6) -> Array: ...
    def project_inverse_sigmas(self, sigmas: Array, tol=1e-6) -> Array: ...
    def calibrate_shape(self, shape: Array) -> Array:
        """Borrowed from Design A: dynamic_dpsgd's mu-schedule becomes
        budget.calibrate_shape(rho_mu ** (iters / T)) — jit-safe, no scipy."""

    # ---- logging convenience ----
    def realized_weights(self, sigmas: Array, clips: Array) -> Array:
        """project_weights(clips / sigmas) — the triple every _get_log_arrays repeats."""
```

### Layer 3: `SigmaClipPairMixin` in `src/policy/schedules/abstract.py`

Captures the dominant caller (sigma_and_clip, parallel, warmup tails): learnable
`noise_schedule` + `clip_schedule` fields.

```python
class SigmaClipPairMixin:
    """Methods-only mixin for schedules with fields: noise_schedule, clip_schedule, privacy_params."""

    def get_private_noise_scales(self) -> Array: ...
    def get_private_clips(self) -> Array: ...
    def get_private_weights(self) -> Array:        # realized_weights(...)
    def _get_log_arrays(self) -> dict[str, Array]: ...

    @eqx.filter_jit
    def project(self) -> Self:
        new_noise, new_clip = self.privacy_params.project_pair(
            self.noise_schedule, self.clip_schedule)
        return eqx.tree_at(
            lambda s: (s.noise_schedule, s.clip_schedule), self, (new_noise, new_clip))
```

The `eqx.tree_at` rebuild preserves every other field (FISTA state, step counters) of
*whatever class self is* — deleting the per-class constructor recitation.

### Usage after the refactor

```python
@register(SigmaAndClipScheduleConfig)
class SigmaAndClipSchedule(SigmaClipPairMixin, AbstractNoiseAndClipSchedule):
    noise_schedule: AbstractSchedule
    clip_schedule: AbstractSchedule
    privacy_params: PrivacyBudget

    @classmethod
    def from_config(cls, conf, privacy_params): ...
    def apply_updates(self, updates) -> Self:
        return eqx.apply_updates(self, updates)
    # project(), get_private_*, _get_log_arrays: inherited
```

`DynamicDPSGDSchedule` drops `__find_mu_0` and its scipy import; its μ-schedule is
`stop_gradient(self.privacy_params.calibrate_shape(shape))` and `project()` becomes
jit-safe. Warmup schedules with differently-named fields (`noise_tail`/`clip_tail`)
call `project_pair` + `tree_at` manually.

## Dependency Strategy

**In-process** — pure computation, merged directly. Two injection changes:

- `max_sigma` becomes a `PrivacyBudget.from_approx_dp` constructor argument. The only
  remaining `SingletonConfig` read in the privacy package lives in the composition-root
  factory `get_privacy_params(dataset_length)`. `privacy/projections.py` and
  `privacy/budget.py` never import `conf/`.
- scipy (`approx_to_gdp` Brent) is confined to `from_approx_dp`, at construction,
  outside any JIT region — now enforced by the classmethod boundary rather than by
  convention.

## Testing Strategy

**New boundary tests to write:**

- *The* invariant, parametrized over all nine schedule types: construct → perturb
  learnable leaves → `project()` → assert
  `budget.is_within_budget(s.get_private_noise_scales(), s.get_private_clips())`.
  This is the test the current architecture cannot express in one place.
- Round-trip: `project()` is idempotent (projecting a projected schedule is a no-op
  within tolerance) and preserves non-projected fields (FISTA state, step counters)
  exactly.
- Property tests on `projections.py` free functions over random `(arrays, bound)`
  pairs: feasibility of output, pass-through of feasible inputs, L2 optimality spot
  checks. These need no `PrivacyBudget`, no config, no schedules.
- `calibrate_shape`: output exhausts the budget to tolerance; validated against
  `dynamic_dpsgd`'s old scipy root-find (Dynamic DP-SGD paper Eq. 10) on a grid of
  `(rho_mu, T)` before the old code is deleted. Note float32 bisection caps tolerance
  near ~1e-6 vs scipy's 1e-12 — assert at the achievable tolerance.
- **Regression pins (borrowed from Design B's analysis):** before deleting the old
  solvers, snapshot old-vs-new projected outputs for representative inputs and pin
  them. Even pure code motion can shift values at float32 tolerance, which compounds
  over long training runs; the pin makes any drift a deliberate decision.

**Old tests to delete (replace, don't layer):**

- The portions of `test_privacy.py` that exercise `GDPPrivacyParameters.project_*`
  through the god-object with manual `SingletonConfig.config` fixture setup — replaced
  by the config-free `projections.py` property tests.
- Per-schedule `project()` unit tests in `test_property_schedules.py` that re-assert
  budget feasibility schedule-by-schedule — replaced by the single parametrized
  boundary test.
- Any test of `compute_eps` / `sigma_schedule_to_weights` that exists only to exercise
  the `SingletonConfig` fallback path.

**Test environment needs:** none — everything is in-process pure JAX; the new solver
tests run without datasets, config, or W&B.

## Implementation Recommendations

Durable guidance, not coupled to current file paths:

**The budget module should own:** the meaning of the privacy constraint (the bound,
expenditure, feasibility), all conversions between schedule representations, and the
entry points to projection and calibration. It is the *only* thing schedules know
about privacy.

**It should hide:** every solver (duals, tolerances, Newton damping, overflow guards),
the equivalence of the three parametrizations, the weights↔μ algebra, the scipy
construction step, and the stop_gradient discipline on budget scalars.

**It should expose:** `bound`, `is_within_budget`, `project`/`project_pair`, the named
differentiable conversions, `calibrate_shape`, and `realized_weights`. Names say what
the caller has and gets — no kwarg dispatch, no à-la-carte numerical internals.

**Migration order (each step leaves the tree green):**

1. Create `projections.py` by moving the three solvers + `_sc_*` helpers, taking
   `bound` as an argument. Keep `GDPPrivacyParameters.project_*` as thin delegates.
   Add the regression pins here.
2. Add `PrivacyBudget` with injected `max_sigma`; make `get_privacy_params` construct
   it. Keep the field name `privacy_params` on schedules so checkpoints keep loading.
3. Add `SigmaClipPairMixin`; migrate `SigmaAndClipSchedule` first (smallest), then
   `ParallelSigmaAndClipSchedule` (verifies FISTA state survives `tree_at`), then the
   warmup variants via `project_pair`.
4. Replace `dynamic_dpsgd.__find_mu_0` with `calibrate_shape`, validated against the
   pinned scipy values.
5. Write the parametrized on-budget boundary test; delete the superseded unit tests;
   delete `GDPPrivacyParameters` once no caller remains (alternating/legacy schedules
   may be deleted rather than migrated — `AlternatingSigmaAndClipSchedule` is already
   out of active use).

**Known risks to watch:** mixin/`eqx.Module` MRO (declare fields in concrete classes,
methods only in the mixin); `tree_at`'s blanket field preservation silently skipping
per-project state mutations a new schedule needs (document the override pattern);
`project_inverse_sigmas` uses `exp(1/σ)` not `exp(1/σ²)` — preserve, don't "fix",
without a deliberate decision.
