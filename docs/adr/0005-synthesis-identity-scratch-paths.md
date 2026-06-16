# Key each symbolic-regression synthesis by a hashed identity, not by target alone

A **synthesis** is now keyed by `(synthesis identity, target)`, where the **synthesis
identity** is every problem-defining input — the sweep (`cache_dir` basename), the row
filters (`datasets`, `arch_labels`, `optimizers`, `run_ids`, the include flags,
`datapoint_frequency`, `keep_features`) and the search space (`maxsize`) — but **not**
the orchestration knobs (`scratch_dir`, `out_dir`, `procs`, `niterations`, the timeouts,
the chain controls, `mirror_sync_secs`) and **not** `targets` (the per-target dir sits
below the identity). The identity is canonicalised (`cache_dir` → basename; multi-valued
tuples sorted; serialised as sort-keyed JSON) and hashed `sha1[:8]`. Its directory-name
rendering — the **synthesis slug** — is an optional human selector prefix from a
hard-coded shortlist (today just `datasets`, e.g. `mnist+fashion-mnist`) joined to the
hash, or the bare hash when no shortlisted selector is set.

The slug becomes one path segment, inserted **above** the per-target directory, on both
tiers:

```
/scratch/$USER/pysr/<sweep>/<slug>/<target>/pysr_run     # live run directory
<cache_dir>/pysr_eval/<slug>/<target>/pysr_run           # persistent mirror
<cache_dir>/pysr_eval/<slug>/{manifest.json,features_full.parquet}
```

The slug derivation, hashing and CLI-flag emission live in one stdlib-only module,
`src/sr_identity.py`, imported by both `symbolic_regression.py` and the
`cc/slurm/sr-run-starter.py` launcher. The launcher folds the slug into the SLURM job
name (`sr-<sweep>-<slug>-<target>`) so dataset/arch variants are distinguishable in
`squeue`. Chains stay on-identity for free: a job already holds its `PySRConfig`, so
`_resubmit_chain` re-emits the identity flags from `conf` rather than threading them
through new `CHAIN_*` env vars.

## Why

Before this, **target was the sole disambiguator** — scratch was `…/pysr/<target>` and
the mirror `…/pysr_eval/<target>`. The moment you filter one sweep to different datasets
(or, later, architectures or T), you create several distinct syntheses that share a
target. They would collide on the same run directory and the same mirror, so a
mnist-filtered job would `warm_start` from — and overwrite — a fashion-mnist job's PySR
state, silently corrupting both fits. Encoding the full problem definition in the path is
what makes concurrent filtered syntheses safe to coexist and to resume.

A **hash** rather than a fully-readable path is what makes this robust to *future* filter
dimensions. Adding a T-filter or single-architecture filter changes the identity, hence
the hash, hence the directory — without changing the path *schema* (no new segment, no
sentinel for unset filters, no path-length blowup). The human selector prefix is kept
purely for greppability; correctness rests entirely on the hash.

Deriving the slug **in the script** (single source of truth) rather than threading a
prebuilt path through the chain means a resumed job lands on the exact same directory as
its predecessor by *recomputing* it from the same identity — there is no opaque path to
keep in sync across `CHAIN_*` vars, only the identity flags the resubmit forwards anyway.

## Considered and rejected

- **Fully human-readable path** (`ds=mnist/arch=all/T=all/…`). Rejected: the path schema
  changes every time a filter dimension is added, unset filters need sentinel segments,
  and deep filter sets risk path-length limits. The hash absorbs all of this.
- **Pure hash, no human prefix.** Rejected as the default: the selector prefix costs
  nothing and makes the common dataset-split case (`…-mnist-…`) scannable in `squeue` and
  on disk. The bare hash is still used when no shortlisted selector is set.
- **Computing the slug in the launcher and threading `scratch_dir`/`out_dir` through
  `CHAIN_*`.** Rejected: it splits the identity logic across two files, grows a new
  `CHAIN_*` env var per future filter, and makes a forgotten var a *silent fork* of the
  search. Re-emitting identity flags from `conf` keeps `conf` the single source of truth.
- **Literal "`asdict(conf)` minus an orchestration denylist" as the identity (auto-include
  every future `PySRConfig` field).** Rejected *for the shared implementation*: the
  login-node launcher must not import `PySRConfig`, because that pulls in `pysr`/Julia,
  which is heavy and may be absent on the submission host. Both sides therefore consult an
  explicit `IDENTITY_FIELDS` includelist in `sr_identity.py` — conceptually the denylist's
  complement, but enumerated so the launcher can compute a matching slug with zero heavy
  imports. The trade-off: a new identity-affecting field must be added to `IDENTITY_FIELDS`
  (one line, one canonical place) as well as to the configs.

## Consequence

- **`IDENTITY_FIELDS` is load-bearing for correctness.** A new `PySRConfig` field that
  changes the *fit* but is omitted from `IDENTITY_FIELDS` will not change the slug, so two
  different problems would share a `warm_start` directory and corrupt each other. Adding a
  filter/search field therefore *requires* adding its name to `IDENTITY_FIELDS`. This is
  the price of keeping the launcher `pysr`-free; the list carries a prominent comment
  saying so.
- **The slug directory is a self-contained synthesis-group artifact.** `manifest.json`,
  `features_full.parquet`, and the three `<target>/` subdirs all live under
  `pysr_eval/<slug>/`. `symbolic_regression_eval.py` needs no change — point its
  `--eval-dir` at `pysr_eval/<slug>` instead of `pysr_eval`.
- **`scratch_dir` and `out_dir` are now bases, not full paths.** The script appends
  `<sweep>/<slug>/<target>` (scratch) and `<slug>/<target>` (mirror). Anything that read
  the old `…/<target>` layout directly must add the `<slug>` level.
- **SLURM job names now embed the slug.** `squeue` distinguishes variants; `scancel` by
  name targets a single variant.
- **No migration needed** — there were no syntheses on disk under the old layout when this
  landed.
</content>
