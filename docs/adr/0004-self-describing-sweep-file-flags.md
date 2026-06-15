# Sweep files carry per-run submission flags as named columns whose header is the run-starter flag name

`create_experiments.py` writes a sweep file of W&B run IDs that the SLURM
fan-out (`cc/slurm/run-starter.py`, driven by `parallel` or
`cc/slurm/cumulative_run_starter.py`) consumes one line at a time. Some
submission flags are *per-run*, not constant across the sweep — `--mem_per_gpu`
is the first: it depends on the run's dataset/architecture/T, which the SLURM
scripts do not otherwise know. Rather than recompute or look those up at submit
time, we make the sweep file **self-describing**: an optional tab-separated
header row whose column names are *exactly* the `run-starter.py` flag names
(`run_id\tmem_per_gpu`), one row per run. The fan-out driver parses the header
and appends `--<col>=<val>` for every non-`run_id` column when it invokes
`run-starter.py`. A future per-run flag is added by emitting one more column
named after its `SlurmConfig` field — no driver logic changes.

`create_experiments.py` seeds the `mem_per_gpu` column with the GPU-free
**heuristic** estimate (`predict_memory._peak_bytes_for(sweep,
heuristic_only=True)`, deduped by `_signature`, rounded by `_next_pow2_gib`),
computed in-process from the `SweepConfig`s it already holds. `predict_memory.py`
becomes a *refine-in-place* tool: on a GPU node it reads the same header file,
recomputes the compiled (exact) estimate, and rewrites the `mem_per_gpu` column
in place — preserving every other column and the row order. The separate
`.mem.txt` output is removed; there is one file and one format.

## Why

- **Per-run flags belong with the run, in the file.** Memory varies by run but
  is knowable at creation time, where the full `SweepConfig` exists. Writing it
  next to the run ID means the submit path stays a dumb forwarder and never
  reconstructs configs just to size a job.
- **Header == flag name is the whole generalisation.** It makes the column set
  open: the driver maps `column → --column` blindly, so "any new run-specific
  flag is added in a similar manner" (the stated goal) costs one column plus a
  matching `SlurmConfig` field, not a parser edit.
- **One source of truth, two estimators.** The cheap heuristic seeds the column
  at creation (no GPU, conservative by construction); the exact compiled path
  refines the same column later. Unifying on one file in place avoids two
  formats and the "which file do I submit from" ambiguity.

## Considered and rejected

- **Positional columns, no header** (`run_id\tmem`, the shape the old printed
  `parallel --colsep` one-liner and `predict_memory`'s `.mem.txt` used).
  Rejected: order-coupled and not self-describing — adding a third flag silently
  shifts column meaning, and a reader cannot tell which flag a column is.
- **Header present but driver maps known columns explicitly.** Rejected: every
  new per-run flag would require editing the driver, defeating the "added in a
  similar manner" goal.
- **`run-starter.py` reads its own row from the file** (`--sweep-file` +
  self-lookup). Rejected: heavier change to the per-run CLI; the driver already
  has the line in hand, so forwarding `--col=val` is simpler and keeps
  `run-starter.py` a pure per-run command.
- **Shell out to `predict_memory.py --heuristic-only` from
  `create_experiments.py`.** Rejected: it would re-download every run's config
  from W&B to recompute a number derivable from `SweepConfig`s already in
  memory.
- **Migrate `sweep.py` to write headers and rewrite existing files.** Rejected:
  unnecessary blast radius. Instead the driver **sniffs**: if line 1's first
  tab-field is literally `run_id` it is the header format; otherwise every line
  is a bare run ID (legacy `sweep.py` / pre-existing sweeps), forwarded with no
  extra flags.

## Consequences

- **The driver must detect format and skip the header.**
  `cumulative_run_starter.py`'s `_count_runs` (and any line-count) subtract the
  header row; the bare-`cat | parallel --run_id={}` path is replaced by a
  column-aware forward for header files and left as-is for legacy files.
- **`predict_memory.py` no longer emits `.mem.txt`** and instead rewrites the
  `mem_per_gpu` column in place; it must preserve unknown columns and row order
  so other per-run flags survive a re-estimate.
- **`create_experiments.py` touches the dataset loader at creation time.** The
  heuristic calls `get_dataset_shapes()`, which may cache/download a dataset the
  first time it runs on a login node. This also fires under `--dry-run`, which
  now previews the deduped signature→mem table.
- **`mem_per_gpu` is seeded with the heuristic, which over-estimates by design**
  (4 B/param upper bound on the scan carry + a conv-activation term). Runs that
  want the tighter compiled number run `predict_memory.py` on a GPU node to
  refine the column before submitting.
