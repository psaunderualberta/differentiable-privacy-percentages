# RFC 001 — Replace `SingletonConfig` with a scoped `RunContext`

**Status:** Proposed
**Candidate category:** Shared-types coupling (In-process)
**Related friction:** Originally item #1 in the architecture audit; fixes #6 (schedule logging config leak) as a side effect.

---

## Problem

`SingletonConfig` (`src/conf/singleton_conf.py`) is a module-global, lazily-initialized `Config` dataclass accessed via classmethods. It is imported in 20+ files and read deep inside:

- **Hot paths under JIT** — `environments/dp.py` (`batch_size` at L122, L276), `environments/losses.py` (`loss_type` at L24), `util/util.py` reshape helpers (L77, L130).
- **`eqx.Module` methods and constructors** — `privacy/gdp_privacy.py` (`max_sigma` at L230, L316; `num_training_steps` at L488, L492), `policy/schedules/abstract.py` `get_logging_schemas` (L44), `policy/schedules/warmup_alternating.py` / `warmup_sigma_and_clip.py` / `warmup_parallel_sigma_and_clip.py` constructors (`num_outer_steps`), `policy/stateful_schedules/median_gradient.py` (L97).
- **Cross-boundary utilities** — `util/logger.py` (L58), `util/dataloaders.py` (L357–358), `util/job_chain.py` (L73), `networks/net_factory.py` (L71–72), `environments/dp_params.py` (L40).

### Why this is a problem

1. **Tests must mutate global state.** Every test that touches a schedule, the privacy module, or the inner training loop sets `SingletonConfig.config = Config(...)` and unsets it in a teardown. Fixtures in `test/test_privacy.py:38`, `test/test_property_based.py:109`, `test/property-tests/conftest.py:26`, and `test/test_dynamic_dpsgd.py:39` all implement this same pattern. Forgotten teardown = cross-test pollution; no mechanism enforces cleanup.

2. **`eqx.Module`s are not self-contained.** `GDPPrivacyParameters.compute_eps` reaches out to the singleton when no `max_sigma` is passed. Schedules read `plotting_interval` inside `get_logging_schemas()`. This means a schedule Module cannot be pickled and restored in a fresh process without re-initializing the singleton — blocking checkpointing, in-process comparison runs, and any form of serialization-based testing.

3. **Implicit config reads hide the true dependency surface of each module.** A reader of `train_with_noise` cannot see that it depends on `batch_size`; it is buried inside the function body. The *actual* contract of each module is wider than its signature suggests.

4. **No scope boundary.** The singleton is process-global. Tests run sequentially-only unless they each restore state; `pytest-xdist` is unsafe; running two experiments in one process is impossible.

---

## Proposed Interface

### The core abstraction — `conf/scope.py` (~25 lines)

```python
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from conf.config import Config

@dataclass(frozen=True)
class RunContext:
    config: Config
    # Future seam: rng_root_key, run_dir, logger handle, etc. can be added
    # here without touching any caller.

_CTX: ContextVar[RunContext] = ContextVar("run_ctx")

def current() -> RunContext:
    """Read the active RunContext. Call only at trace time, not inside JIT."""
    return _CTX.get()

@contextmanager
def using(ctx: RunContext):
    """Scope a RunContext for the duration of a block. Nests safely."""
    tok = _CTX.set(ctx)
    try:
        yield ctx
    finally:
        _CTX.reset(tok)
```

Three public names: `RunContext`, `current()`, `using(...)`. Nothing else.

### Transitional shim on `SingletonConfig` (removed in the cutover PR)

```python
# conf/singleton_conf.py — appended to the SingletonConfig class
@classmethod
@contextmanager
def override(cls, cfg: Config):
    prev = cls.config
    cls.config = cfg
    try:
        yield
    finally:
        cls.config = prev
```

This exists only to let tests move off the unsafe `SingletonConfig.config = X` idiom in PR 1, before the deeper call-site migration lands.

### Usage

**Composition root (`main.py`, `sweep.py`):**

```python
_get_config()  # existing tyro + W&B restore, populates SingletonConfig
with using(RunContext(SingletonConfig.get_instance())):
    _train_inner()
```

**Deep call site (`environments/dp.py`):**

```python
# Before: batch_size = SingletonConfig.get_environment_config_instance().batch_size
# After:
batch_size = current().config.sweep.env.batch_size
```

**`eqx.Module` factory (`privacy/gdp_privacy.py`):**

```python
class GDPPrivacyParameters(eqx.Module):
    max_sigma: float = eqx.field(static=True)  # captured at construction
    T: int = eqx.field(static=True)
    ...
    def compute_eps(self, max_sigma=None):
        ms = max_sigma if max_sigma is not None else self.max_sigma
        return (jnp.exp(1/ms**2) - 1) / (jnp.exp(self.mu_0**2) - 1)

def get_privacy_params(n_train: int) -> GDPPrivacyParameters:
    cfg = current().config
    return GDPPrivacyParameters(
        ...,
        max_sigma=cfg.sweep.schedule_optimizer.max_sigma,
        T=cfg.sweep.env.num_training_steps,
    )
```

**Test (post-migration):**

```python
from conf.scope import RunContext, using

def test_compute_eps():
    cfg = Config(sweep=SweepConfig(env=EnvConfig(...), ...), ...)
    with using(RunContext(cfg)):
        params = get_privacy_params(n_train=1000)
    # Note: outside the `using` block, `params` still works — max_sigma and T
    # were captured at construction. This is the main testability win.
    assert params.compute_eps().shape == ()
```

### What this hides

- The mechanism by which implicit config propagates to deep call sites (`ContextVar` vs. module global vs. thread-local) is an implementation detail of `conf/scope.py`.
- `RunContext` is a seam: future cross-cutting run-level state (RNG root key, output directory, logger handle) can be added to it without touching any of the ~30 call sites.
- `eqx.Module`s capture the scalars they need at construction, hiding the fact that those values originated in a config tree.

### What this deliberately does NOT introduce

- No typed `ProviderKey` / per-key binding system (rejected design D2). The `Config` dataclass tree remains the single source of truth and `current().config.<path>` is the universal access pattern.
- No DI container, no resolver chaining, no `scope.child(overrides=...)`.
- No change to the existing `tyro.cli(Config, ...)` parsing or the W&B-run-restore merge logic in `_reconstruct_from_dict`.

---

## Dependency Strategy

**Category:** In-process (pure computation, in-memory state, no I/O at the scope boundary).

The scope machinery is a process-local `ContextVar`. No external service, no local stand-in, no port/adapter split needed. Tests instantiate a `Config`, wrap the code under test in `with using(RunContext(cfg)):`, and assert on the public return values of the deepened Modules.

One discipline rule, identical to today's implicit rule for `SingletonConfig`: **`current()` is called at trace time (Python level, outside `@eqx.filter_jit`), not inside traced code.** Factories and `__init__` methods are the right places; the body of a `jax.lax.scan` step is not. This is not a regression — today's code has the same constraint.

---

## Testing Strategy

### New boundary tests to write

1. **`test/test_scope.py` (new file)** — unit-test the scope machinery itself:
   - `using(ctx)` makes `current()` return `ctx`.
   - `using` nests correctly and restores the outer context on exit.
   - `current()` outside any `using` block raises a clear `LookupError`.
   - Exception inside `using` body still restores the prior context.

2. **Pickle round-trip tests for `eqx.Module`s that were previously non-picklable** — one per affected Module class:
   - Construct `GDPPrivacyParameters` / `WarmupAlternatingSigmaAndClipSchedule` / `MedianGradientSchedule` under a `using(...)` block.
   - Exit the block.
   - Pickle → unpickle in a fresh context (no active `using`).
   - Call representative methods (`compute_eps`, `get_private_sigmas`, `get_logging_schemas`) and assert they succeed and return the pre-pickle values. This is the test that does not exist today.

3. **`train_with_noise` boundary test** — invoke the inner training loop with an explicitly constructed `Config` and no pre-existing global state; assert the val loss drops on a 2-step synthetic scenario. Currently this path is only covered indirectly by integration tests.

### Old tests to delete / simplify

- The four existing fixtures that do `SingletonConfig.config = Config(...)` + teardown (`test/test_privacy.py:38`, `test/test_property_based.py:109`, `test/property-tests/conftest.py:26`, `test/test_dynamic_dpsgd.py:39`) are replaced with a single shared pytest fixture that wraps `using(...)`, yielding the config. Net: ~40 lines of fixture boilerplate deleted.

- Any test whose *only* purpose was to verify a schedule Module reads the right field from the singleton (grep for assertions on `SingletonConfig.config` inside tests) becomes redundant — the field is now a captured static on the Module and is directly inspectable.

### Test environment needs

None. No new dependencies, no stand-in services, no port adapters. Pure in-process refactor.

---

## Implementation Recommendations

### Rollout — four small PRs

| PR | Scope | Revertible? |
|---|---|---|
| **1** | Add `conf/scope.py`. Add `SingletonConfig.override()` shim. Migrate the 4 existing test fixtures to `override()`. Wrap `main()` and `sweep.py` entry bodies in `with using(RunContext(SingletonConfig.get_instance())):`. No call-site changes yet. | Yes — purely additive. |
| **2** | Convert `eqx.Module` reach-outs into captured static fields. Affected classes: `GDPPrivacyParameters`, `WarmupAlternatingSigmaAndClipSchedule`, `WarmupSigmaAndClipSchedule`, `WarmupParallelSigmaAndClipSchedule`, `MedianGradientSchedule`, and any other Module that calls `SingletonConfig` inside a method or `__init__`. Factories (`from_config` classmethods, `get_privacy_params`, `make_schedule`) do the `current()` read. Add the pickle round-trip tests here. | Yes — per-Module, independently. |
| **3** | Mechanical rewrite of the remaining ~25 non-Module call sites: `SingletonConfig.get_X_config_instance().Y` → `current().config.<path>.Y`. File-by-file; each file's diff is trivially reviewable. | Yes — per-file. |
| **4** | Cutover: delete the `SingletonConfig` class. Move `_get_config()` + `_reconstruct_from_dict()` into `conf/scope.py` (or a sibling `conf/startup.py`). Migrate tests from `override()` → `using()`. Remove the transitional shim. | This PR is the point of no return. |

PR 1 alone delivers the full testability win (safe test overrides). PR 2 alone delivers the full architectural win (`eqx.Module`s are self-contained and picklable). PRs 3 and 4 are hygiene.

### Module responsibilities after the refactor

- **`conf/scope.py`** owns the lifecycle of the active `RunContext`: entering, exiting, and reading. It does not own parsing or W&B restore.
- **`conf/startup.py`** (or `conf/scope.py` after PR 4) owns CLI parsing and W&B-run-restore merging — the one-shot "build a `Config` from external inputs" responsibility.
- **`eqx.Module` classes** own only the subset of config they actually use, as static fields captured at construction. They never reach out to read config during methods.
- **Factories** (`get_privacy_params`, `make_schedule`, `DPTrainingParams.create_direct_from_config`, `net_factory`) are the exclusive bridge between the ambient `RunContext` and the Modules they build.

### What the interface exposes

- `RunContext(config)` — construct with a `Config`.
- `using(ctx)` — context manager; enter to scope, exit to restore.
- `current()` — read the active context.

That is the entire public API. Downstream code accesses config via `current().config.<path>` — the same `Config` dataclass tree that exists today, touched through a scoped accessor instead of a class-level global.

### Migration rule for reviewers

In PR 3, when rewriting a call site, if the read happens **inside a function that is compiled into a `jax.lax.scan` body or the body of a `@eqx.filter_jit`'d function**, hoist it to the nearest enclosing factory / top-level trace-time function and pass the resulting scalar as an argument. Do not place `current()` calls inside traced code. This preserves the existing (correct) pattern where config-derived scalars become static constants in the JAX trace.

---

## Non-goals

- **In-process side-by-side runs of different configs.** A single `ContextVar` cannot hold two values simultaneously in the same async context. If this capability is ever needed, upgrading from `RunContext` to a per-key scoped binding (design D2) is a mechanical follow-up refactor — no call-site changes required beyond swapping the accessor.
- **Eliminating the `Config` dataclass tree.** This RFC preserves `Config` / `SweepConfig` / `EnvConfig` / `ScheduleOptimizerConfig` / `WandbConfig` unchanged. All CLI parsing, W&B sweep param generation, and run-restore logic keep working.
- **Refactoring the registry / `_get_config_classes` system.** Candidate #4 in the architecture audit is a separate concern, tracked in a future RFC.
