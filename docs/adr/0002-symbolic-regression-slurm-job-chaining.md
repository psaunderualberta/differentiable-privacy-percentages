# Run each symbolic-regression synthesis as a self-chaining, single-launch multi-node SLURM job that resumes via PySR's native state

Symbolic regression for σ, clip, and μ runs on SLURM as **three independent syntheses** (one per target, never sharing an output directory). Each synthesis is a **job chain**: a sequence of ~2h55m wall-time jobs, each running PySR for `timeout_in_seconds=9900` (2h45m) then resubmitting its successor with a `-d` dependency on its own job id. A chain stops on **natural completion** (`fit()` returns before the timeout) or when **chain depth** reaches `--max-chain-jobs` (default 16). Resume uses PySR's native `warm_start` from a fixed **run directory** (`output_directory` + pinned `run_id`), which on PySR ≥1.x restores the full Julia search state, not just the hall-of-fame. The run directory lives on `/scratch` during a job and is rsynced to a **persistent mirror** under `cache_dir/pysr_eval/<target>/` every 15 min and on `fit()` return; a resuming job restores from the mirror if `/scratch` was purged. Each synthesis requests `--ntasks=N` with no `--nodes` pin; the launcher runs once (single task) and PySR's `cluster_manager="slurm"` + `procs=$SLURM_NTASKS` fans workers out across whatever nodes the allocation lands on.

## Why

A future reader will expect this to reuse `main.py`'s existing job-chaining (`util/job_chain.py`), which polls `SLURM_JOB_END_TIME`, checkpoints to a **W&B artifact**, and resumes by `checkpoint_run_id`. It deliberately does not, because PySR is a different execution model:

1. **PySR self-stops and self-checkpoints.** `timeout_in_seconds` already ends the search gracefully and writes `julia_state_stream_` + `hall_of_fame`, so the `SLURM_JOB_END_TIME` polling loop is redundant. We set the PySR timeout *below* wall time and let PySR be the clock.
2. **State is file-based, not a W&B artifact.** A synthesis has no W&B run; its state is a directory. Resuming means pointing PySR at that directory with `warm_start`, not restoring an artifact by run_id. Hence a separate `run_id` namespace and a separate `from_file` resume path (see the "run"/"checkpoint" ambiguity notes in `UBIQUITOUS_LANGUAGE.md`).
3. **`cluster_manager="slurm"` requires a single launch.** PySR's `addprocs_slurm` runs `srun` *internally* to spawn workers. Wrapping the launcher in a multi-task `srun` (as `run-starter.py` does for `main.py`) would start N copies of the script. The launcher must therefore run exactly once while the allocation holds N tasks.

Requesting `--ntasks=N` without a `--nodes` pin lets the scheduler place tasks across any node layout, which backfills at least as fast as a single-node-constrained request — the scheduling-latency win that motivated going multi-node in the first place.

## Considered and rejected

- **Reuse `main.py`'s W&B-artifact chaining.** Rejected: PySR has no W&B run, self-checkpoints to files, and self-times-out — the artifact/run_id/polling machinery is a mismatch and would have to be bent awkwardly around `fit()`.
- **One job running all three targets in a `for target` loop.** Rejected: 3× the wall clock per job makes the <3h budget far harder, and a timeout mid-loop forces tracking which target was in flight. Independent per-target chains also backfill separately and isolate failures.
- **Hall-of-fame-only restart** (reseed a fresh search from `equations.csv` each job). Rejected as the default: PySR ≥1.x persists `julia_state_stream_` through pickling, so full-state resume is available and preserves population diversity across handoffs. Kept only as a fallback if native state resume proves flaky on the cluster.
- **Single-node multicore** (`--nodes=1`, no cluster manager). Rejected: pins the allocation to one node, losing backfill flexibility.
- **Convergence / stagnation-based stopping.** Rejected for now: adds a tolerance to tune and cross-job hall-of-fame comparison. Termination is the simpler pair of natural completion + a hard chain-depth cap.
- **Run directory on `/project` only** (no `/scratch`). Rejected: PySR's frequent checkpoint writes hit project quota and slower storage; `/scratch` is the fast shared tier, with periodic mirroring for durability.

## Consequence

- **`niterations` is the primary cost knob.** With a depth cap of 16 and natural completion as the only other stop, total compute per target is bounded by `min(niterations-worth-of-search, 16 × ~2h55m)`. There is no convergence-based early exit.
- **A crash ends a chain silently.** Resubmission fires only after `fit()` returns from a clean timeout; an exception (OOM, node failure, Julia error) leaves no successor. This is fail-safe against runaway chains but means transient failures require manual restart. (Revisit with `afterany` crash-retry if this proves painful.)
- **Two parallel notions of "run" and "checkpoint" now exist** in the codebase (W&B vs PySR); they share no code path and are disambiguated in `UBIQUITOUS_LANGUAGE.md`.
- A new launcher (`cc/slurm/sr-run-starter.py`) is needed rather than extending `run-starter.py`, because the single-launch requirement and the file-based `CHAIN_*` context (target, depth, run directory) differ from the `main.py` launcher.
