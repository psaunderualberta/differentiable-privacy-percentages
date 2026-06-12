# RFC 004 — Unify checkpoint + graceful shutdown + job-chain resubmit into `RunLifecycle`

**Status:** Proposed
**Candidate category:** Co-ownership of a concept (mixed: in-process decision logic + local-substitutable filesystem + true-external W&B/subprocess)
**Related friction:** Item #1 in the architecture audit. Composes with RFC 003 (schedule-update step) — see "Interaction with RFC 003".

---

## Problem

The decision "should this run stop now, persist its state, and re-launch a successor?" is split across three places with a fragile ordering contract enforced only by *where statements sit* in `src/main.py`. Two shallow modules (`util/job_chain.py`, `util/checkpointing.py`) plus inline glue co-own the concept.

**The pieces:**

1. **Stop decision** — a module-global `threading.Event` set by a SIGUSR1 handler (`job_chain.py:41`), OR `time_limit_approaching()` which lazily imports `SingletonConfig` and compares `SLURM_JOB_END_TIME` against `sweep_config.shutdown_buffer_secs`.
2. **State persistence** — `make_state(...)` (`checkpointing.py:83`) returns an **untyped 6-key dict** that `main.py` must reconstruct *exactly* as an Orbax restore template (L138), then un-stick from device-0 via `jnp2np2jnp` on four leaves (L153–160).
3. **Resubmit-vs-suppress** — the loop checks `shutdown_requested() or time_limit_approaching()` → checkpoint → `break` (L298–301); then *after* `run.finish()`, `resubmit_if_requested()` re-checks the same predicate and shells out to `run-starter.py` (L345–347) — but only if `interrupted` (KeyboardInterrupt) is False (L323–327).

### Why this is a problem

1. **Invariants are encoded by line ordering and cannot be tested.** Four invariants live implicitly in `main.py`: (a) checkpoint is written *before* the break; (b) resubmit fires *after* `run.finish()`; (c) KeyboardInterrupt finishes+syncs but does **not** resubmit; (d) the periodic checkpoint (every `checkpoint_every` steps) coexists with the shutdown checkpoint. None of these has a test, because every branch is welded to `os.environ`, a process-global `Event`, `wandb`, Orbax, and `subprocess`.

2. **The stop predicate is evaluated three times and can disagree.** `shutdown_requested() or time_limit_approaching()` is re-evaluated at the loop break (L298), the final-logging guard (L307), and again inside `resubmit_if_requested`. A SIGUSR1 or deadline crossing *after* the loop but *before* the resubmit can fire a resubmit on what should have been a non-graceful exit. This is a latent correctness bug — a wrongly-fired (or wrongly-suppressed) resubmit wastes a SLURM allocation.

3. **The state contract is untyped and duplicated.** The 6-key dict (`schedule`, `opt_state`, `key`, `init_key`, `step`, `es_state`) is built three times in `main.py` via `make_state`, unpacked by hand, and must structurally match the Orbax template or restoration silently corrupts. A field rename is a coordinated edit across two files with no type checking.

4. **A non-obvious workaround leaks into the caller.** The `jnp2np2jnp` device-uncommit (L153–160) — needed because Orbax restores arrays committed to device 0, conflicting with sharded `noise_keys` — sits in `main.py` rather than next to the restore that necessitates it.

`job_chain.py` and `checkpointing.py` are shallow: a lot of caller-side knowledge for a small surface. `job_chain` has **no tests**; only `checkpointing` is covered (`test/test_checkpointing.py`, real filesystem + mocked W&B).

---

## Proposed Interface

**Design F (common-case)** from the design exploration: one cohesive coordinator that *wraps* the existing leaf functions rather than reimplementing them, lifting only the ~40 lines of decision logic into a testable surface. The explicit-invariant test list is lifted from the ports-and-adapters design.

```python
# src/util/run_lifecycle.py
import dataclasses
from enum import Enum, auto
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from conf.singleton_conf import SingletonConfig
from util import job_chain
from util.checkpointing import save_checkpoint, load_checkpoint
from util.util import jnp2np2jnp


class StopReason(Enum):
    RUNNING     = auto()   # no stop latched
    JOB_CHAIN   = auto()   # SIGUSR1 / wall-clock -> checkpoint + resubmit
    INTERRUPTED = auto()   # KeyboardInterrupt -> finish, NO resubmit
    COMPLETED   = auto()   # loop ran to num_outer_steps (implicit; reason stays RUNNING)


@dataclasses.dataclass
class TrainingState:
    """Typed façade over make_state's untyped 6-key dict. Field names/order are
    the Orbax pytree contract. `as_orbax_dict`/`from_orbax_dict` bridge to the
    existing wire format so checkpointing.py and its tests stay unchanged.
    Use a shallow {f.name: getattr(...)} mapping — NOT stdlib asdict, which
    recurses into and corrupts eqx Modules.
    """
    schedule: Any
    opt_state: Any
    key: jax.Array
    init_key: jax.Array
    es_state: Any
    step: jax.Array            # jnp.int32, last completed outer step


class RunLifecycle:
    """One object the training loop drives. Owns restore, periodic + shutdown
    checkpointing, the latched stop decision, and finalize/resubmit. Reads
    SingletonConfig directly (house style). The W&B `run` is passed via attach()
    because it does not exist at restore() time (restore precedes wandb.init).
    """

    def __init__(self, *, now: Callable[[], float] | None = None):
        self._now = now                       # the ONE injected seam (see below)
        self._reason = StopReason.RUNNING
        self._run = None
        self._wandb = SingletonConfig.get_wandb_config_instance()
        self._sweep = SingletonConfig.get_sweep_config_instance()

    # ---- startup (before wandb.init) ----
    def restore(self, template: TrainingState) -> tuple[Optional[TrainingState], int]:
        """Restore-if-resuming. Returns (state_or_None, start_step). Owns the
        jnp2np2jnp round-trip on device-committed leaves. (None, 0) when
        checkpoint_run_id is unset or nothing is found."""

    def attach(self, run: Any) -> None:
        """Bind the W&B run and register the SIGUSR1 handler. Call after wandb.init."""

    # ---- per-step ----
    def should_stop(self) -> bool:
        """Latch + report whether a job-chain stop is in effect. Latches
        JOB_CHAIN the first time SIGUSR1/wall-clock fires, so the decision
        can't flip mid-finalize (removes the triple-eval race)."""

    def checkpoint(self, state: TrainingState, *, force: bool = False) -> None:
        """Save iff force or (step+1) % checkpoint_every == 0. Covers BOTH the
        periodic save and the shutdown save (force=True)."""

    # ---- shutdown ----
    def mark_interrupted(self) -> None:
        """Record a KeyboardInterrupt. Suppresses resubmit in finalize()."""

    @property
    def stopped_for_chain(self) -> bool:
        """True iff stopping for job-chain; used to skip final logging."""

    def finalize(self) -> None:
        """Resubmit the continuation job iff a job-chain stop was latched.
        Call AFTER run.finish(). No-op for INTERRUPTED / COMPLETED."""
```

### Usage — the new `main.py`

**Restore (replaces L136–160, ~22 lines → ~6):**

```python
lifecycle = RunLifecycle()
template = TrainingState(schedule, opt_state, key, init_key, es_state, jnp.array(0, jnp.int32))
restored, start_step = lifecycle.restore(template)
if restored is not None:
    schedule, opt_state = restored.schedule, restored.opt_state
    key, init_key, es_state = restored.key, restored.init_key, restored.es_state
```

**After wandb.init (replaces L167):**

```python
run = init_wandb_run(wandb_config, sweep_config)
lifecycle.attach(run)
```

**Inside the loop (replaces L298–304):**

```python
    state = TrainingState(schedule, opt_state, key, init_key, es_state, jnp.array(t, jnp.int32))
    if lifecycle.should_stop():
        lifecycle.checkpoint(state, force=True)   # checkpoint-before-break
        break
    lifecycle.checkpoint(state)                    # periodic, internally gated
```

**Final-logging guard (replaces L307):**

```python
if not lifecycle.stopped_for_chain:
    ...  # force loggables, line plots
```

**Except / finally / resubmit (replaces L323–347):**

```python
except KeyboardInterrupt:
    lifecycle.mark_interrupted()
    print("\nKeyboardInterrupt — finishing run, syncing, no resubmit...")
finally:
    if sync_daemon is not None: sync_daemon.stop()
    logger.finish()
    if sync_daemon is not None: sync_offline_run(wandb_config.mode, run_dir)
    run.finish()

lifecycle.finalize()   # resubmit iff job-chain stop; no `interrupted` flag needed
```

### What complexity it hides internally

| Hidden | Was | Now owned by |
|---|---|---|
| Global `threading.Event` + SIGUSR1 handler | `job_chain._shutdown_requested` | `attach()` / private predicate |
| SLURM env reads + buffer-from-config | `time_limit_approaching()` | private predicate, evaluated once |
| Triple-evaluated stop predicate | L298, L307, inside resubmit | latched in `should_stop()` / `stopped_for_chain` |
| Untyped 6-key dict + manual unpack | `make_state` + L148–152 | `TrainingState` façade |
| `jnp2np2jnp` device-uncommit | L153–160 | inside `restore()` |
| `start_step = saved_step + 1` | `load_checkpoint` return | surfaced once off `restore()` |
| periodic-vs-shutdown checkpoint gating | L298–304 | `checkpoint(force=...)` |
| resubmit-after-finish ordering | L343 vs L347 | `finalize()` called after teardown |
| KeyboardInterrupt suppresses resubmit | `interrupted` flag, L327/L346 | `StopReason` state machine |

**The latch is the key correctness fix:** the stop decision transitions into `JOB_CHAIN` once and stays there, so break, final-logging gating, and resubmit all read one source of truth — eliminating the post-loop predicate race.

---

## Dependency Strategy

Mixed categories, handled pragmatically — a **single coordinator over unchanged leaf functions**, not a full hexagon.

- **In-process (kept concrete):** the stop decision (`Event` + SLURM env + buffer) and state bundling (`TrainingState`). Driven in tests by setting the env var / calling the handler — cheap and reliable, no port needed.
- **Local-substitutable (Orbax/filesystem):** unchanged. `checkpointing.py` already round-trips through the real filesystem in `test_checkpointing.py`; `RunLifecycle` calls `save_checkpoint`/`load_checkpoint` as-is and does not reimplement them.
- **True external (W&B artifact + sbatch subprocess):** these stay inside the leaf functions (`save_checkpoint` no-ops offline; `resubmit_if_requested` shells out). Tests **mock these two functions** — `resubmit_if_requested` in particular is the sbatch boundary and must always be mocked so a test never spawns a real job.
- **The ONE injected seam — `now`.** The wall-clock time-limit branch depends on real `time.time()` vs `SLURM_JOB_END_TIME`. Injecting `now: Callable[[], float]` (default = real clock) lets a test drive the deadline crossing deterministically and assert the latch fires exactly once, without the flakiness of monkeypatching `time.time` globally.

**Rejected alternative — full ports & adapters** (Clock / ShutdownSignal / CheckpointStore / Resubmitter, each with a production adapter and an in-memory fake). It is the textbook-correct decomposition and makes every invariant testable with *zero* I/O, but it adds ~10 new types of "Protocol + prod + fake" boilerplate for a module with **exactly one production caller**. YAGNI: adopt it only if a second caller (non-SLURM scheduler, or GCS instead of W&B artifacts) actually appears. The coordinator above buys the same invariant tests via two monkeypatches + one `now` seam, at a fraction of the surface.

---

## Testing Strategy

**New boundary tests to write** (the four invariants the current code cannot assert; monkeypatch `save_checkpoint` and `resubmit_if_requested`, inject `now`):

- **checkpoint-before-break + periodic coexist:** with `save_checkpoint` recording calls, a forced shutdown writes a checkpoint at the shutdown step *and* periodic saves still fire at each `checkpoint_every` multiple, with no double-write of the same step.
- **resubmit-after-finish:** after a `JOB_CHAIN` latch, `finalize()` calls the (mocked) resubmit exactly once, with the run id.
- **KeyboardInterrupt suppresses resubmit:** `mark_interrupted()` after a graceful stop → `finalize()` is a no-op; resubmit recorder empty.
- **no resubmit on normal completion:** loop never stops → `finalize()` no-op.
- **latched stop / no mid-finalize flip:** inject `now` so the deadline crosses at a chosen step; `should_stop()` stays True afterward and `stopped_for_chain` gates final logging off.
- **restore round-trip:** mock `load_checkpoint` to return a dict with device-committed arrays; assert the returned `TrainingState` leaves are uncommitted and `start_step` propagates. (Real Orbax/fs path stays covered by `test_checkpointing.py`; this tests the façade + `jnp2np2jnp` ownership.)

**Old tests to delete:** none. `test_checkpointing.py` stays valid (the wire format is unchanged). This is net-new coverage for logic that currently has none — `job_chain.py` gains its first tests.

**Test environment needs:** a seeded `SingletonConfig` (already standard in this repo's fixtures), the injected `now`, and monkeypatches on the two leaf functions. No real filesystem, W&B, subprocess, or signals required for the decision-logic tests.

---

## Implementation Recommendations

Durable guidance, independent of current file paths:

- **The module should own:** the restore-or-start decision (+ `jnp2np2jnp`), the latched stop predicate, the periodic-vs-forced checkpoint cadence, and the resubmit-vs-suppress truth table (`JOB_CHAIN ∧ ¬INTERRUPTED ∧ ¬COMPLETED`).
- **The module should hide:** the global `Event`, the SLURM env reads, the untyped state dict, the device-uncommit workaround, and the ordering invariants that today live in `main.py` line placement.
- **The module should expose:** `restore(template)`, `attach(run)`, `should_stop()`, `checkpoint(state, force=)`, `mark_interrupted()`, `stopped_for_chain`, `finalize()`.
- **Coordinate, don't reimplement.** Keep `checkpointing.py` and `job_chain.py` as the leaf implementations (Orbax/W&B/subprocess); `RunLifecycle` owns only *sequencing and invariants*. This keeps the heavy, already-tested code in place and limits the new type cost to what buys tested invariants.
- **Latch the stop decision once.** Never re-evaluate the stop predicate after it first fires — read the latched `StopReason` everywhere downstream.
- **Restore precedes `attach`.** `restore()` runs before `wandb.init`; `attach()` after. Document this ordering — nothing enforces it beyond the contract, and a caller that checkpoints before `attach` hits `self._run is None`.
- **`RunLifecycle` is single-shot.** Once `JOB_CHAIN`/`INTERRUPTED` latches it is terminal; a future caller running multiple sequential phases in one process needs a fresh instance per phase.
- **Migration:** replace L136–160, L167, L298–307, and L323–347 in `main.py` with the method calls above; the caller keeps its own `try/finally` for W&B/logger/sync teardown (not a lifecycle concern), and `finalize()` runs after that block so resubmit-after-finish holds structurally.

### Interaction with RFC 003

RFC 003 keeps `opt_state` a plain caller-owned value (rather than hiding it inside the `ScheduleUpdater`). That makes `TrainingState` here straightforward: it bundles `schedule` + `opt_state` as plain leaves, and the updater is a stateless transform that never appears in a checkpoint. The two refactors compose; land them in either order.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
