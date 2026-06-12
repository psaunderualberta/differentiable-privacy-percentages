# RFC 003 — Deepen the schedule-update step into a `ScheduleUpdater`

**Status:** Proposed
**Candidate category:** Co-ownership of a concept (In-process, pure computation)
**Related friction:** Item #4 in the architecture audit. Composes with RFC 004 (run lifecycle) — see "Interaction with RFC 004".

---

## Problem

Every outer-loop iteration in `src/main.py` open-codes the same multi-step ritual that takes `(grads, opt_state, schedule)` to the next `schedule`. It appears in two places that must stay in sync.

**Per-iteration (main.py:281–291):**

```python
updates, opt_state = optimizer.update(grads, opt_state, schedule)
schedule = schedule.apply_updates(updates)
schedule = ensure_valid_pytree(schedule, "schedule in main after updates")
x_new = schedule.project()
x_new = ensure_valid_pytree(x_new, "schedule in main after project")
if getattr(schedule, "use_fista", False):
    schedule = schedule.fista_advance(x_new)
    schedule = schedule.fista_extrapolate()
else:
    schedule = x_new
schedule = ensure_valid_pytree(schedule, "schedule in main after fista")
```

**Startup seed (main.py:74–76)** — a second copy of the same branch:

```python
schedule = schedule.project()
if getattr(schedule, "use_fista", False):
    schedule = schedule.fista_extrapolate()
```

**Optimizer construction (main.py:121–129)** — the robustness chain `clip_by_global_norm → zero_nans → sgd`, whose ordering is a *correctness* property (the load-bearing comment at L112–120 explains `zero_nans` must precede `sgd` so the momentum trace never ingests a NaN).

### Why this is a problem

1. **A 5-field state machine's ordering invariant lives in the caller.** FISTA exists only on `ParallelSigmaAndClipSchedule` (and its warmup variant), as a 5-field `eqx.Module` state machine (`_x_curr/_x_prev_sigmas/clips`, `_fista_t`). Its contract — *`fista_advance(x_new)` must be called on the gradient-updated `schedule`, not on `x_new`, then `fista_extrapolate()`* — is enforced only by reader vigilance in `main.py`. This is the single most error-prone line in the outer loop, and it is duplicated (seed + step).

2. **FISTA is duck-typed at the call site.** `getattr(schedule, "use_fista", False)` is the branch key. Non-FISTA schedules (`AbstractNoiseAndClipSchedule` in `policy/schedules/abstract.py`) have no `use_fista` attribute at all; the branch silently falls through to `x_new`. The discriminator is a string-named attribute probed at runtime, repeated in two locations.

3. **Dead code is unreachable by construction.** `ParallelSigmaAndClipSchedule.fista_advance_with_restart(x_new, grads)` implements the O'Donoghue–Candès gradient restart but is **never called** — the caller's branch never threads `grads` into the advance step, so the only way to reach it is to rewrite the loop.

4. **It is untestable except by running a full training loop.** There is no boundary at which you can assert "FISTA path advances state correctly" or "a NaN gradient produces a no-op step" without standing up the entire outer loop, dataset, and DP-SGD inner loop.

---

## Proposed Interface

**Design A (minimal)** from the design exploration, with the `optimizer=` injection escape hatch borrowed from the common-case design for testability.

A single `eqx.Module` with three entry points. The caller never touches `optimizer.update`, `apply_updates`, `project`, `fista_*`, or `ensure_valid_pytree` again.

```python
# src/policy/update/updater.py
import enum
from typing import NamedTuple
import equinox as eqx
import optax
from jaxtyping import PyTree

from policy.schedules.abstract import AbstractNoiseAndClipSchedule


class AccelMode(enum.Enum):
    PLAIN = "plain"                       # schedule = x_new
    FISTA = "fista"                       # advance(x_new) -> extrapolate
    FISTA_GRAD_RESTART = "fista_restart"  # advance_with_restart(x_new, grads) -> extrapolate


class StepResult(NamedTuple):
    schedule: AbstractNoiseAndClipSchedule
    opt_state: optax.OptState


class ScheduleUpdater(eqx.Module):
    """Owns one outer-loop schedule-update step: optax update -> apply ->
    project -> (optional FISTA accel) -> validate. Hides the FISTA branch,
    the advance-then-extrapolate ordering, the startup seed, and the
    NaN/Inf guards. Stateless w.r.t. schedule/opt_state (threaded through);
    holds only the optimizer and acceleration policy, fixed at build.
    """

    optimizer: optax.GradientTransformation = eqx.field(static=True)
    accel: AccelMode = eqx.field(static=True)
    validate: bool = eqx.field(static=True)

    @classmethod
    def from_config(
        cls,
        schedule_optimizer_conf,                       # sweep.schedule_optimizer
        schedule: AbstractNoiseAndClipSchedule,
        *,
        optimizer: optax.GradientTransformation | None = None,  # test override
    ) -> "ScheduleUpdater":
        """Build the optax robustness chain (clip_by_global_norm -> zero_nans ->
        sgd) unless an `optimizer` is injected, and pick the acceleration mode
        from the schedule's `use_fista` flag + config. Optimizer is OWNED here."""
        ...

    def init_opt_state(self, schedule) -> optax.OptState:
        return self.optimizer.init(eqx.filter(schedule, eqx.is_array))

    def seed(self, schedule):
        """Startup: project, then seed FISTA lookahead iff accelerated.
        Replaces main.py 74-76."""
        ...

    def step(self, schedule, opt_state, grads) -> StepResult:
        """One full outer-loop update. Replaces main.py 281-291."""
        ...
```

### Usage

**Build + startup seed** (replaces L121–129 and L74–76):

```python
schedule = make_schedule(schedule_conf, gdp_params)
updater = ScheduleUpdater.from_config(sweep_config.schedule_optimizer, schedule)
schedule = updater.seed(schedule)
opt_state = updater.init_opt_state(schedule)
```

**Loop body** (replaces L281–291):

```python
loss = ensure_valid_pytree(loss, "loss in main")   # stays — it's the loss, not the schedule
schedule, opt_state = updater.step(schedule, opt_state, grads)
```

### What complexity it hides internally

```python
def step(self, schedule, opt_state, grads):
    updates, opt_state = self.optimizer.update(grads, opt_state, schedule)
    schedule = schedule.apply_updates(updates)
    schedule = self._guard(schedule, "after updates")
    x_new = self._guard(schedule.project(), "after project")
    match self.accel:
        case AccelMode.PLAIN:
            schedule = x_new
        case AccelMode.FISTA:
            schedule = schedule.fista_advance(x_new).fista_extrapolate()
        case AccelMode.FISTA_GRAD_RESTART:
            schedule = schedule.fista_advance_with_restart(x_new, grads).fista_extrapolate()
    schedule = self._guard(schedule, "after accel")
    return StepResult(schedule, opt_state)
```

- **The FISTA-vs-plain branch** — encoded once as `AccelMode`, resolved at build from `getattr(schedule, "use_fista", False)`; the `getattr` smell vanishes from both call sites.
- **The ordering contract** — `fista_advance` is *always* called on `schedule` (carrying `_x_curr = x_k`), never on `x_new`; `fista_extrapolate` always follows. Enforced structurally, not by comment.
- **The startup seed** — `seed()` shares the same `AccelMode` switch, so seed and step can never disagree on whether FISTA is on.
- **`ensure_valid_pytree` placement** — `_guard` wraps it and is a no-op when `validate=False`.
- **Optimizer construction** — the `clip_by_global_norm → zero_nans → sgd` chain and its rationale comment move into `from_config`, next to the only thing that consumes it.
- **Gradient restart becomes reachable** — `FISTA_GRAD_RESTART` routes to the dead `fista_advance_with_restart(x_new, grads)`. Since `step` already has `grads` in scope, enabling it is a one-line config switch with no loop edits.

---

## Dependency Strategy

**In-process — merged directly.** Pure computation, no I/O, JIT-adjacent.

- **Optimizer: OWNED.** `from_config` builds the optax chain internally. The chain order is a correctness property consumed *only* here; leaving its assembly to the caller re-exposes the coupling we are hiding. An optional `optimizer=` parameter lets tests inject a toy `optax.sgd(...)` and never touch `SingletonConfig`.
- **`opt_state`: threaded, not owned.** It stays a plain caller-owned value because it must survive checkpoint save/restore (round-tripped through `jnp2np2jnp` + Orbax). The updater owns the *transformation*, not the *state*. **This is the deliberate divergence from the "common-case" design that owned `opt_state` internally — owning it would entangle the updater with RFC 004's checkpoint surface.**
- **`schedule`: threaded.** It is checkpointed, logged, and passed to `get_training_loss`, so it flows through `step` as a value.
- **`ensure_valid_pytree`: owned internal detail**, gated by `validate` so tests can exercise the pure transform.

---

## Testing Strategy

**New boundary tests to write** (at the `ScheduleUpdater` interface, no training loop required):

- `step` on a plain (non-FISTA) schedule returns a projected schedule and advances `opt_state`; output passes the privacy constraint check.
- `step` on a FISTA schedule advances the 5-field state correctly: `_x_prev ← x_k`, `_fista_t` increments, lookahead `≠` projected iterate after the first step.
- `seed` on a FISTA schedule projects then extrapolates (lookahead seeded); on a plain schedule projects only.
- `seed`/`step` agree on the FISTA branch — a plain schedule never calls `fista_*`, a FISTA schedule always does.
- `FISTA_GRAD_RESTART` mode routes through `fista_advance_with_restart` and restarts when the gradient opposes momentum (`dot > 0`).
- With `validate=True`, a NaN-injected gradient is neutralised by the optax chain into a no-op step rather than raising.
- `StepResult` ordering — `.schedule`/`.opt_state` are not swappable silently.

**Old tests to delete:** none directly — this logic currently has *no* tests (it is only exercised end-to-end). The new boundary tests are net-new coverage.

**Test environment needs:** none beyond a constructed schedule + the injected toy optimizer. No dataset, no W&B, no SingletonConfig if `optimizer=` is passed.

---

## Implementation Recommendations

Durable guidance, independent of current file paths:

- **The module should own:** the optax robustness-chain construction, the FISTA-vs-plain dispatch, the FISTA advance-then-extrapolate ordering, the startup seed, and the placement of Python-level NaN/Inf guards.
- **The module should hide:** which schedules have FISTA (the `use_fista` discriminator), the `x_new` temporary, the rule that non-FISTA discards `x_new` after no-op while FISTA advances on the gradient-updated object.
- **The module should expose:** `from_config` (build + pick accel mode), `init_opt_state`, `seed` (once at startup), `step(schedule, opt_state, grads) -> StepResult` (once per iteration).
- **Acceleration is a closed set.** Add new schemes as `AccelMode` members handled in the one `match` block, *not* via a strategy registry — the flexible (Protocol + registry) design is over-built for a 2-to-3 implementation problem and was rejected. Revisit only if a fourth scheme that re-queries the objective (e.g. line search) actually lands.
- **`use_fista` stays a field on the FISTA schedules** (the state fields are meaningless without it); the refactor removes it only from the *hot loop*, not from existence.
- **Migration:** replace L74–76, L121–129, and L281–291 in `main.py` with the four-call usage above. Keep `ensure_valid_pytree(loss, ...)` in the caller — it guards the loss scalar, not the schedule.

### Interaction with RFC 004

RFC 004 (`run lifecycle`) bundles `schedule` + `opt_state` into a typed `TrainingState` for checkpointing. Because this RFC keeps `opt_state` a caller-owned value (rather than hiding it inside the updater, as the rejected common-case design did), the two refactors **compose cleanly**: `ScheduleUpdater` is a stateless transform that never appears in a checkpoint, and `TrainingState` carries `schedule`/`opt_state` as plain leaves. Land them in either order.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
