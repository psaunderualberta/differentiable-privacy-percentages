# The reference sweep is one SLURM task per candidate, with a selector task for the final eval

`Baseline.baseline_sweep` evaluates 20 random candidates at `iterations=10` inner trainings
each, then re-evaluates the winner — 203 inner DP-SGD trainings in one blocking call, or
roughly 87 GPU-hours per reference per target column at T=5000, against an 11:55 wall clock.
The transfer reference stage is therefore split along a new **candidate** manifest
dimension: one task per (reference × target × candidate) at `iterations=3`, each writing its
candidate's mean accuracy, followed by a selector task that picks the best and runs the
final evaluation.

## Status

accepted

## Why

**The stage could not complete as written.** Every reference task would have been killed at
roughly 13% of its work, and `transfer_reference.py` never passes `resumable=True` — ADR
0003's per-candidate checkpointing is wired into `main.py`'s pre-training path only — so
nothing was saved and a relaunch restarted from zero. Without references the transfer matrix
has no comparison at all, making this the stage most likely to sink the chapter.

**Splitting is the only option that keeps the baselines honestly tuned.** The cheap
alternative is to expose `num_runs_in_sweep` and shrink the search, but a 6-point random
search over 3–4 hyperparameters is thin enough that an examiner can fairly call the
references a straw man — and the chapter's claim is precisely that transferred shapes beat
them. Splitting preserves the full 20-candidate search while bounding a task at ~1.3h, which
also moves the stage into the ≤3h queue where scheduling priority is best. Total compute is
unchanged; only its packaging is.

## Considered and rejected

- **Plumb `num_runs_in_sweep` / `iterations` and shrink** (6 candidates × 2 runs, ~6.5h per
  task). Half an hour of work against several hours for the split, but it buys the fit
  problem above, and with no checkpointing one timeout still loses a whole task.
- **Tune once at ε=10 and reuse the hyperparameters at ε=8.** Halves 18 sweeps to 9 but
  leaves each at 87h, so it only ever helped in combination — and σ magnitude moves with the
  budget, weakening the matched-tuning claim.
- **Reuse the baselines computed during source training.** Cheapest by far and rejected
  outright: those were tuned on mnist/fashion-mnist, so any margin measured on
  eyepacs/imagenet/chexpert would be confounded by target mismatch rather than schedule shape.
- **Wire ADR 0003's checkpointing into the transfer producer instead.** Makes an 87h task
  survivable rather than short, so the stage would still occupy the long queue for days;
  the split gets both the tuning and the fast queue.

## Consequences

- A new intermediate artifact exists: per-candidate result rows, which are **not** transfer
  cells and must not be read by the assembler. Only the selector's output carries
  `producer="reference"`.
- `drop_finished` gains a second granularity — a finished candidate is skipped
  independently of its selector — so a partial reference stage resumes at candidate level.
- Candidate selection now uses a 3-run mean rather than a 10-run mean. Selection is noisier
  but unbiased; the winner is re-evaluated cleanly for the reported number regardless.
- The reference stage gains a DAG edge (candidates → selector), making it the only
  two-phase producer.
