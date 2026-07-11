"""Checkpoint saving and restoring via Orbax (local) + W&B artifacts (remote).

Design
------
Each checkpoint is a snapshot of the full outer-loop training state:
  - schedule    : the learned eqx.Module (σ/clip weights)
  - opt_state   : optax optimizer state
  - key         : JAX PRNG key at end of step (resumes the random stream)
  - init_key    : JAX PRNG key used for network init (fixed when
                  train_on_single_network=True; updated each step otherwise)
  - step        : the outer-loop step index that was just completed

Saving
------
`save_checkpoint` always writes locally first with Orbax's StandardCheckpointer,
then calls `run.log_artifact()` to upload to W&B.  When the run is in
``disabled`` mode, `log_artifact` is a no-op — so saving always succeeds and
local checkpoints are always available even when offline.

Restoring
---------
`load_checkpoint` first looks for the checkpoint on disk (local saves from
prior runs on this machine).  Only if nothing is found locally does it attempt
to download from W&B, so it works fully offline as long as the local checkpoint
directory exists.  Returns ``(restored_state, start_step)`` where
``start_step = saved_step + 1``, or ``None`` if neither source is available.

W&B branching
-------------
Handled in main.py: if ``checkpoint_step`` is not None (i.e. restoring from
a specific historical step rather than the latest), a new W&B run is created
in ``{project}-branched`` with the original run id and step recorded in the
run notes.
"""

import os
import pathlib
from typing import Any

import numpy as np
import orbax.checkpoint as ocp

import wandb

# Resolve the project root from the location of this file:
# src/util/checkpointing.py → src/util → src → project root
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _ckpt_dir(run_id: str) -> pathlib.Path:
    slurm_tmpdir = os.environ.get("SLURM_TMPDIR", "")
    if os.environ.get("SLURM_TMPDIR", ""):
        return pathlib.Path(slurm_tmpdir) / "checkpoints" / run_id
    return _PROJECT_ROOT / "checkpoints" / run_id


def _artifact_name(run_id: str) -> str:
    return f"checkpoint-{run_id}"


def _remote_checkpoint_exists(entity: str, project: str, run_id: str) -> bool:
    """True iff at least one checkpoint artifact version exists on W&B for ``run_id``.

    Used to tell apart two cases that both surface as "the W&B download failed":
    the *first job of a chain* (nothing has been saved yet — safe to start
    fresh) versus a *continuation whose checkpoint exists but couldn't be
    fetched* (must abort rather than silently restart from step 0 and clobber
    the in-progress run's history).  Any lookup error is treated as "does not
    exist" so a genuine first run is never blocked.
    """
    collection = f"{entity}/{project}/{_artifact_name(run_id)}"
    try:
        return any(True for _ in wandb.Api().artifacts("checkpoint", collection))
    except Exception:
        return False


def _find_local_checkpoint(run_id: str, step: int | None) -> pathlib.Path | None:
    """Return the local checkpoint directory for a run and step.

    If ``step`` is None, returns the directory for the highest-numbered
    (latest) step.  Returns ``None`` if no matching local checkpoint exists.
    """
    run_dir = _ckpt_dir(run_id)
    if not run_dir.exists():
        return None

    candidates = [d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not candidates:
        return None

    if step is None:
        return max(candidates, key=lambda d: int(d.name))

    target = run_dir / str(step)
    return target if target.exists() else None


def save_checkpoint(
    state: dict[str, Any],
    step: int,
    run: Any,
) -> None:
    """Save training state locally with Orbax and upload to W&B as an artifact.

    Local saving always succeeds.  The W&B upload is a no-op when the run is
    in ``disabled`` mode, so this function is safe to call in offline tests.

    The artifact name is ``checkpoint-{run_id}`` and two aliases are set:
    ``latest`` (always points to the most recent upload) and ``step-{step}``
    (permanent — allows restoring any historical step even after later
    checkpoints have been uploaded).
    """
    step_dir = _ckpt_dir(run.id) / str(step)
    # Create the run directory; Orbax creates the step directory itself via an
    # atomic rename and will raise if the step directory already exists.
    step_dir.parent.mkdir(parents=True, exist_ok=True)

    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(step_dir, state)
    # StandardCheckpointer uses async I/O internally; wait here so the step
    # directory exists on disk before we try to add it to the W&B artifact.
    checkpointer.wait_until_finished()

    # Save human-readable schedules alongside the Orbax checkpoint so that
    # dp_psac_ref/run.py can consume them without importing src/.
    schedule = state["schedule"]
    np.save(step_dir / "sigmas.npy", np.asarray(schedule.get_private_noise_scales()))
    np.save(step_dir / "clips.npy", np.asarray(schedule.get_private_clips()))

    artifact = wandb.Artifact(
        name=_artifact_name(run.id),
        type="checkpoint",
        metadata={"step": step, "run_id": run.id},
    )
    artifact.add_dir(str(step_dir))
    run.log_artifact(artifact, aliases=["latest", f"step-{step}"])
    print(f"Checkpoint saved: step {step} → {step_dir}")


def load_checkpoint(
    checkpoint_run_id: str,
    checkpoint_step: int | None,
    state_template: dict[str, Any],
    entity: str | None,
    project: str | None,
) -> tuple[dict[str, Any], int] | None:
    """Restore a checkpoint, checking local disk before attempting a W&B download.

    Parameters
    ----------
    checkpoint_run_id:
        The W&B run ID whose checkpoint to restore.
    checkpoint_step:
        Step to restore.  ``None`` means the latest available checkpoint.
    state_template:
        A freshly-built state dict with the same pytree structure and array
        shapes as the saved checkpoint.  Orbax restores leaf values into this
        template, so the structure must match exactly.
    entity:
        W&B entity for the remote artifact fallback.  May be ``None`` to
        disable remote lookup (e.g. in tests or offline runs).
    project:
        W&B project for the remote artifact fallback.  When branching to a
        new run, pass the *original* project where the checkpoint was saved.

    Returns
    -------
    ``(restored_state, start_step)`` where ``start_step = saved_step + 1``,
    or ``None`` if the checkpoint cannot be found or restored from either source.
    """
    checkpointer = ocp.StandardCheckpointer()

    # --- 1. Try local disk first (works offline / in tests) ---
    local_dir = _find_local_checkpoint(checkpoint_run_id, checkpoint_step)
    if local_dir is not None:
        print(f"Restoring local checkpoint: {local_dir}")
        restored = checkpointer.restore(local_dir, target=state_template)
        start_step = int(restored["step"]) + 1
        print(f"Restored step {start_step - 1}; resuming at step {start_step}")
        return restored, start_step

    # --- 2. Fall back to W&B artifact download ---
    if entity is None or project is None:
        print(
            f"Warning: no local checkpoint found for run '{checkpoint_run_id}' "
            f"and entity/project not provided — cannot download from W&B."
        )
        return None

    alias = "latest" if checkpoint_step is None else f"step-{checkpoint_step}"
    artifact_path = f"{entity}/{project}/{_artifact_name(checkpoint_run_id)}:{alias}"
    print(f"Attempting to download checkpoint: {artifact_path}")

    try:
        artifact = wandb.Api().artifact(artifact_path)
        local_path = pathlib.Path(artifact.download())

        restored = checkpointer.restore(local_path, target=state_template)
        start_step = int(restored["step"]) + 1
        print(f"Restored W&B checkpoint step {start_step - 1}; resuming at {start_step}")
        return restored, start_step

    except Exception as e:
        # A missing artifact means this is the first job of a chain (nothing
        # saved yet), so starting fresh — return None — is correct.  But if a
        # checkpoint DOES exist and we merely failed to fetch it, restarting
        # from step 0 would resume the SAME W&B run and clobber the in-progress
        # chain's history (steps re-logged from scratch, summary reset).  This
        # is exactly what corrupted the NoMomentumSweep runs.  Refuse: raise so
        # the duplicate / re-launched job dies here (before wandb.init) instead
        # of silently overwriting good data.
        if _remote_checkpoint_exists(entity, project, checkpoint_run_id):
            raise RuntimeError(
                f"Checkpoint '{artifact_path}' exists on W&B but could not be "
                f"restored ({e!r}). Refusing to restart from step 0, which would "
                f"clobber the in-progress run. Resolve the fetch failure and retry."
            ) from e
        print(
            f"No checkpoint found for run '{checkpoint_run_id}' "
            f"(first job of chain) — starting fresh: {e}"
        )
        return None
