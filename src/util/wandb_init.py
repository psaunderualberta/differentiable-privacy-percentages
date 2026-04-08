"""W&B run initialisation helper.

Encapsulates the three-case branching logic for starting a run:
  1. Branching from a specific historical checkpoint step → new run in
     ``{project}-branched`` with the source run/step recorded in notes.
  2. Fresh start (no restart_run_id) → new run in the configured project,
     with ``resume="allow"`` so an accidental re-launch won't duplicate runs.
  3. Resuming an existing run (restart_run_id set) → continue that run
     without overwriting its stored config.
"""

import os
from typing import Any

import wandb
from conf.config import SweepConfig, WandbConfig


def init_wandb_run(
    wandb_config: WandbConfig,
    sweep_config: SweepConfig,
) -> Any:
    """Initialise and return a W&B run.

    Parameters
    ----------
    wandb_config:
        Top-level W&B settings (project, entity, mode, restart/checkpoint IDs).
    sweep_config:
        Experiment config serialised into the W&B run config on fresh starts.
    """
    # When running under SLURM, redirect all W&B local storage to the fast
    # per-job temp directory so the shared filesystem doesn't fill up.
    wandb_dir = os.environ.get("SLURM_TMPDIR", None)

    is_branching = wandb_config.checkpoint_run_id is not None

    if is_branching:
        branch_project = (wandb_config.project or "runs") + "-branched"
        notes = (
            f"Branched from run {wandb_config.checkpoint_run_id} "
            f"at step {wandb_config.checkpoint_step} "
            f"(original project: {wandb_config.project})"
        )
        return wandb.init(
            project=branch_project,
            entity=wandb_config.entity,
            mode=wandb_config.mode,
            config=sweep_config.to_wandb_conf(),
            notes=notes,
            dir=wandb_dir,
        )

    if wandb_config.restart_run_id is None:
        # Fresh run — pass id=None explicitly so wandb auto-generates an ID.
        return wandb.init(
            project=wandb_config.project,
            entity=wandb_config.entity,
            id=wandb_config.restart_run_id,
            mode=wandb_config.mode,
            config=sweep_config.to_wandb_conf(),
            resume="allow",
            dir=wandb_dir,
        )

    # Resuming an existing run — omit config so it is not overwritten.
    return wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        id=wandb_config.restart_run_id,
        mode=wandb_config.mode,
        resume="allow",
        dir=wandb_dir,
    )
