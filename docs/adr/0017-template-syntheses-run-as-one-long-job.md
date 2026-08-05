# A template-mode synthesis runs as one long job, not a resubmitting chain

A synthesis fitting a `TemplateExpressionSpec` is submitted as a **single** SLURM job whose
wall time is the whole intended search budget (`--max-chain-jobs 1`, `timeout_in_seconds`
set to wall time minus the teardown pad). It does not use the self-resubmitting chain of
ADR 0002. That chain is correct only for pooled scalar fits, whose PySR state can be
reloaded; a template synthesis cannot resume, so chaining it discards work instead of
accumulating it.

## Status

accepted (amends ADR 0002 for template mode; ADR 0002 stands unchanged for pooled fits)

## Why

ADR 0002 rests on a premise that template mode voids: *"Resume uses PySR's native
`warm_start` … which on PySR ≥1.x restores the full Julia search state, not just the
hall-of-fame."* ADR 0006 records why this fails — the template's `combine` is an anonymous
Julia closure whose type cannot be reconstructed in a fresh Julia session, so neither the
pickled `equations_` nor the raw `julia_state_stream_` deserialises across processes — and
`symbolic_regression.py` implements the consequence directly: a synthesis is resumable only
when it has no expression spec. In template mode that condition is never true.

Neither ADR drew the conclusion, so the chain logic was left as-is, and the two documents
combine into a silent waste. A template job runs a fresh search, hits its timeout (the
iteration budget never completes), and therefore always resubmits. Its successor starts
**another** fresh search into the same pinned run directory, overwriting the predecessor's
`model.pkl` and `equations.csv`. At the default depth cap of 16 that is roughly
16 × 2h55m × 32 cores ≈ **1500 core-hours per target**, of which the surviving output is a
single 2h45m search. ADR 0002's stated bound — total compute per target is
`min(niterations-worth-of-search, 16 × ~2h55m)` — is false here: the search *achieved* is
one job's worth no matter how many run.

A single long job is the only option that makes the search budget real without new
machinery. PySR's `timeout_in_seconds` remains the clock, exactly as ADR 0002 intends; only
the number of jobs changes.

## Considered and rejected

- **Reseed each chained job from the previous `equations.csv`** (the hall-of-fame fallback
  ADR 0002 explicitly kept in reserve for this situation). Rejected for now: it would make
  chaining genuinely cumulative and preserve the backfill-friendly ~3h job size, but it
  needs a way to inject a starting population into a template-spec search, and PySR offers
  no supported path for that. Worth revisiting if queue times make long jobs impractical.
- **Switch to pooled scalar fits to regain native resume.** Rejected: it would trade the
  whole per-condition-constant design of ADR 0006 for a scheduling convenience.
- **Leave the chain at 16 and accept the result.** Rejected: it spends ~6000 core-hours
  across the four FirSweep syntheses to obtain four 2h45m searches, and does so invisibly —
  the chain looks like it is making progress.
- **Keep one successor as crash insurance** (`--max-chain-jobs 2`). Rejected as the default:
  the successor restarts fresh, so it does not extend the search; it only guarantees one
  completed long run if the first dies early, at double the worst-case cost. Available when
  a particular synthesis warrants it.

## Consequences

- **Longer queue waits.** A multi-day allocation backfills far worse than a ~3h one, which
  is the property ADR 0002 optimised for. This is the price of the search budget being
  real.
- **A crash loses the run outright**, since nothing resubmits. It does not lose the
  *results*: PySR writes `hall_of_fame.csv` into the run directory throughout the search, so
  the Pareto front reached so far survives on disk and can be read even though the search
  cannot be continued.
- **`timeout_in_seconds` must be set per submission**, no longer a fixed 9900s. It is
  wall time minus the pad, and its relationship to `pad_seconds` still decides whether
  `should_resubmit` fires — which with a cap of 1 job it cannot.
- **`max_chain_jobs` counts jobs, not chain depth.** It did not when this ADR was written:
  `should_resubmit` tested `chain_depth < max_chain_jobs` against a 0-based depth, so
  `--max-chain-jobs 1` still submitted one successor — a second full-length job that
  restarts the search from scratch and overwrites the first's `model.pkl` and
  `equations.csv`, which is this ADR's failure mode at 2× rather than 16×. The comparison
  is now `chain_depth + 1 < max_chain_jobs`, making both this ADR's cap of 1 and ADR 0002's
  16-job bound mean what they say.
- **The multi-target loop stays forbidden.** ADR 0002 rejected running several targets in
  one job; with a days-long per-target timeout that rejection becomes essential rather than
  merely tidy, since the timeout applies to each `fit()` in turn.
- ADR 0002 remains the design of record for pooled scalar syntheses, where full-state
  resume works and the chain accumulates as described.
