"""RunLifecycle — one coordinator the training loop drives.

Owns the restore-or-start decision, the periodic-vs-forced checkpoint cadence,
the *latched* stop decision (SIGUSR1 / wall-clock), and the resubmit-vs-suppress
truth table.  It wraps the existing leaf functions (``checkpointing.py``,
``job_chain.py``) rather than reimplementing them, so the heavy Orbax / W&B /
subprocess code stays in place and tested.

See refactors/004-run-lifecycle.md for the design rationale.
"""

import dataclasses
import os
import time
from collections.abc import Callable
from enum import Enum, auto
from typing import Any

import jax

from conf.singleton_conf import SingletonConfig
from util import job_chain
from util.checkpointing import load_checkpoint, save_checkpoint
from util.job_chain import resubmit_if_requested
from util.util import jnp2np2jnp


class StopReason(Enum):
    RUNNING = auto()  # no stop latched
    JOB_CHAIN = auto()  # SIGUSR1 / wall-clock -> checkpoint + resubmit
    INTERRUPTED = auto()  # KeyboardInterrupt -> finish, NO resubmit


@dataclasses.dataclass
class TrainingState:
    """Typed façade over the untyped 6-key Orbax checkpoint dict.

    Field names are the Orbax pytree contract.  ``as_orbax_dict`` /
    ``from_orbax_dict`` bridge to the existing wire format so checkpointing.py
    and its tests stay unchanged.  ``as_orbax_dict`` uses a shallow
    ``{name: value}`` mapping — NOT ``dataclasses.asdict``, which would recurse
    into and corrupt eqx Modules.
    """

    schedule: Any
    opt_state: Any
    key: jax.Array
    init_key: jax.Array
    es_state: Any
    step: jax.Array  # jnp.int32, last completed outer step

    _WIRE_KEYS = ("schedule", "opt_state", "key", "init_key", "step", "es_state")

    def as_orbax_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self._WIRE_KEYS}

    @classmethod
    def from_orbax_dict(cls, d: dict[str, Any]) -> "TrainingState":
        return cls(**{k: d[k] for k in cls._WIRE_KEYS})


class RunLifecycle:
    """One object the training loop drives.  See module docstring.

    Reads SingletonConfig directly (house style).  The W&B ``run`` is bound via
    ``attach()`` because it does not exist at ``restore()`` time (restore
    precedes wandb.init).  Single-shot: once JOB_CHAIN/INTERRUPTED latches it is
    terminal.
    """

    def __init__(self, *, now: Callable[[], float] | None = None):
        self._now = now  # the ONE injected seam (wall-clock deadline)
        self._reason = StopReason.RUNNING
        self._run = None
        self._wandb = SingletonConfig.get_wandb_config_instance()
        self._sweep = SingletonConfig.get_sweep_config_instance()

    # ---- startup (before wandb.init) ----
    def restore(self, template: TrainingState) -> tuple[TrainingState | None, int]:
        """Restore-if-resuming.  Returns (state_or_None, start_step).  Owns the
        jnp2np2jnp round-trip on device-committed leaves.  (None, 0) when
        checkpoint_run_id is unset or nothing is found."""
        if self._wandb.checkpoint_run_id is None:
            return None, 0

        result = load_checkpoint(
            self._wandb.checkpoint_run_id,
            self._wandb.checkpoint_step,
            template.as_orbax_dict(),
            self._wandb.entity,
            self._wandb.project,
        )
        if result is None:
            return None, 0

        restored_dict, start_step = result
        state = TrainingState.from_orbax_dict(restored_dict)
        # Orbax restores arrays as device-committed (bound to device 0), which
        # conflicts with sharded noise_keys inside the JIT call.  Round-trip the
        # committed leaves through numpy to produce uncommitted arrays, matching
        # what a fresh (non-checkpoint) run provides.  es_state/step are left as
        # restored (es_state may be None; step is consumed only as start_step).
        state.schedule = jnp2np2jnp(state.schedule)
        state.opt_state = jnp2np2jnp(state.opt_state)
        state.key = jnp2np2jnp(state.key)
        state.init_key = jnp2np2jnp(state.init_key)
        return state, start_step

    def attach(self, run: Any) -> None:
        """Bind the W&B run and register the SIGUSR1 handler.  Call after
        wandb.init (the run does not exist at restore() time)."""
        self._run = run
        job_chain.register_signal_handler()

    # ---- per-step ----
    def checkpoint(self, state: TrainingState, *, force: bool = False) -> None:
        """Save iff ``force`` or ``(step+1) % checkpoint_every == 0``.  Covers
        BOTH the periodic save and the shutdown save (force=True), so a step is
        written at most once per call."""
        step = int(state.step)
        if force or (step + 1) % self._wandb.checkpoint_every == 0:
            save_checkpoint(state.as_orbax_dict(), step, self._run)

    def should_stop(self) -> bool:
        """Latch + report whether a job-chain stop is in effect.  Latches
        JOB_CHAIN the first time SIGUSR1 / wall-clock fires, so the decision
        can't flip mid-finalize (removes the triple-eval race)."""
        if self._reason is StopReason.RUNNING and (
            job_chain.shutdown_requested() or self._time_limit_approaching()
        ):
            self._reason = StopReason.JOB_CHAIN
        return self._reason is StopReason.JOB_CHAIN

    @property
    def stopped_for_chain(self) -> bool:
        """True iff stopping for job-chain; used to skip final logging."""
        return self._reason is StopReason.JOB_CHAIN

    # ---- shutdown ----
    def mark_interrupted(self) -> None:
        """Record a KeyboardInterrupt.  Suppresses resubmit in finalize()."""
        self._reason = StopReason.INTERRUPTED

    def finalize(self) -> None:
        """Resubmit the continuation job iff a job-chain stop was latched.
        Call AFTER run.finish().  No-op for INTERRUPTED / normal completion."""
        if self._reason is StopReason.JOB_CHAIN:
            resubmit_if_requested(self._run.id)

    def _time_limit_approaching(self) -> bool:
        """True iff SLURM wall time expires within ``shutdown_buffer_secs``.
        Uses the injected ``now`` seam (default = real clock).  False when
        ``SLURM_JOB_END_TIME`` is absent (local / non-SLURM runs)."""
        job_end = os.environ.get("SLURM_JOB_END_TIME", None)
        if job_end is None:
            return False
        try:
            now = self._now() if self._now is not None else time.time()
            return now >= int(job_end) - self._sweep.shutdown_buffer_secs
        except ValueError:
            return False
