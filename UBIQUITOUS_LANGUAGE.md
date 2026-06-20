# Ubiquitous Language
Idea from https://github.com/mattpocock/skills/

## Differential Privacy

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Privacy budget** | The total allowed privacy cost for a training run, expressed as (ε, δ)-DP or equivalently as a GDP μ value | — |
| **ε (epsilon)** | The privacy loss parameter in (ε, δ)-DP; smaller = stronger privacy | privacy epsilon |
| **δ (delta)** | The failure probability in (ε, δ)-DP; probability of violating the ε bound | privacy delta |
| **GDP μ (mu)** | The scalar privacy parameter under Gaussian Differential Privacy; the composition of all inner steps must satisfy μ-GDP | Gaussian DP parameter |
| **μ₀ (mu-zero)** | The per-step GDP μ value when all T inner steps spend the privacy budget uniformly (the "flat" baseline) | uniform mu |
| **Poisson subsampling probability p** | The probability that any single training example is included in a given minibatch; equals batch_size / dataset_size | subsampling rate |
| **Privacy expenditure** | The actual GDP μ consumed by a given (σ, C) schedule; must not exceed the target μ | privacy cost, privacy usage |
| **Budget fidelity** | How well a **predicted schedule** stays within the privacy budget: the gap between its **privacy expenditure** and the target GDP μ, plus the **projection** distance needed to make it budget-valid. A pointwise-accurate equation can still have poor budget fidelity | privacy fidelity, constraint fidelity |

## DP-SGD Inner Loop

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Inner loop** | The T-step DP-SGD training run for a single network; produces a val loss that feeds back to the outer loop | private training, inner training |
| **T (num_training_steps)** | The total number of DP-SGD gradient steps in one inner loop run | total_timesteps (deprecated CLI flag), inner timesteps |
| **σ (sigma, noise scale)** | The standard deviation of the Gaussian noise added to clipped gradients at each inner step; larger = more privacy noise | sigma schedule value |
| **C (clip, clipping threshold)** | The per-sample gradient L2-norm bound used in Abadi clipping at each inner step | gradient clip, clip norm |
| **Noise vector** | The actual sampled spherical Gaussian with standard deviation σ·C added to the summed clipped gradients; distinct from the noise scale σ | noise |
| **Minibatch** | The Poisson-sampled subset of training examples used in a single inner step | batch |
| **Per-sample gradient** | The gradient of the loss with respect to model parameters for a single training example, computed before clipping | individual gradient |
| **Abadi clipping** | The global-norm clipping strategy that scales each per-sample gradient so its L2 norm ≤ C | gradient clipping |

## Schedule

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Schedule** | The sequence of (σₜ, Cₜ) pairs—one per inner step—that defines how noise and clipping vary over T steps | policy (when referring to the parametric object), noise plan |
| **Schedule weights** | The learnable, non-negative T-dimensional vector whose squared values sum to T; the internal parameterization from which σ and C are derived | weights (without qualifier—conflicts with model weights) |
| **σ-schedule** | The length-T array of noise scales derived from the schedule weights; returned by `get_private_sigmas()` | sigma array |
| **C-schedule (clip-schedule)** | The length-T array of clipping thresholds derived from the schedule weights; returned by `get_private_clips()` | clip array |
| **Project** | The operation that maps raw gradient-updated schedule weights back onto the valid privacy constraint set (privacy expenditure ≤ target μ) | normalize, clip schedule |
| **Warmup** | A variant of a schedule that holds σ and C fixed for the first K outer steps before beginning gradient-based optimization | burn-in |
| **Predicted schedule** | The σ-, C-, or μ-schedule obtained by evaluating a symbolic-regression equation across a run's inner steps. Compared against that run's *actual* learned schedule to assess how well the equation matches it | fitted schedule, equation schedule |
| **Template mode** | The default symbolic-regression mode (`template_mode`): a PySR `TemplateExpressionSpec` fits one **schedule shape** shared across runs plus per-**condition** free constants, instead of a single pooled equation over the existing features | category mode, parameterized SR |
| **Schedule shape (f)** | The universal functional form `f(step_norm)`, shared across all conditions in a template-mode fit, that captures how σ/C vary over training progress before per-condition modulation | shape function, base curve |
| **Schedule hyperparameter (pₖ)** | One of the K free per-condition constants (`p1, p2, …`, default 3) that modulate the **schedule shape** for a given **condition**; interpretable knobs and the (future) stage-2 regression target | template parameter, fitted constant (ambiguous with PySR equation constants) |
| **Condition** | The grouping key `(dataset, eps, T, arch_label)` that indexes the per-condition constants in template mode; the 3 replicate **seeds** of a condition share one constant vector | category (use only for the PySR feature column), group |

## Outer Loop (Meta-optimization)

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Outer loop** | The gradient-based loop that updates schedule weights to minimize mean val loss across sampled inner loop runs | RL loop, training loop (ambiguous) |
| **Outer step (num_outer_steps)** | One iteration of the outer loop: sample networks, run inner loops, backprop into schedule, update, project | total_timesteps (deprecated), outer timestep |
| **Schedule optimizer** | The optax optimizer (SGD with momentum) that updates schedule weights in the outer loop | policy optimizer |
| **Schedule batch size** | The number of independent inner loop networks trained in parallel per outer step to estimate the policy loss | policy_batch_size |
| **Policy loss** | The mean validation loss across all inner loop runs in one outer step; the quantity whose gradient flows back into the schedule | training loss (ambiguous with inner loop loss) |
| **val loss** | The loss on the held-out test set at the end of an inner loop run; what the outer loop minimizes | validation loss |

## Model & Data

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Model weights** | The neural network parameters being trained inside the inner loop; distinct from schedule weights | weights (without qualifier) |
| **Network** | A single `eqx.Module` instance (MLP or CNN) trained in one inner loop run | model |
| **Reinit** | The operation that randomizes model weights while preserving network architecture, used to sample a fresh network for each inner loop | reset |

## Symbolic Regression & Job Orchestration

| Term | Definition | Aliases to avoid |
| --- | --- | --- |
| **Synthesis** | One PySR symbolic-regression search fitting an equation for a *single* target (σ, clip, or μ) over a *single* **synthesis identity** (data selection + search space). A synthesis is keyed by `(identity, target)`, not by target alone: filtering the same sweep to different datasets/architectures yields *distinct* syntheses that share a target. Each has its own output directory and its own SLURM job chain | PySR run (ambiguous with W&B run), regression job |
| **Synthesis identity** | The canonical set of inputs that determines *which problem* a synthesis fits: every `PySRConfig` field **except** the orchestration denylist (`scratch_dir`, `out_dir`, `procs`, `niterations`, `timeout_in_seconds`, `pad_seconds`, `max_chain_jobs`, `chain_depth`, `mirror_sync_secs`) and `target`. `cache_dir` enters by **basename** only (host-independence). Multi-valued tuples are sorted; the lot is serialized as canonical JSON and hashed `sha1[:8]`. Two invocations sharing an identity are the *same* problem and may safely `warm_start` from each other's checkpoint | filter config, fingerprint |
| **Synthesis slug** | The directory-name rendering of a **synthesis identity**: an optional human-readable selector prefix (e.g. `mnist+fashion-mnist`) from a hard-coded shortlist of salient fields, joined to the `sha1[:8]` identity hash — `mnist-a1b2c3d4`, or the bare hash when no shortlisted selector is set. Inserted as one path segment *above* the per-target dir on both `/scratch` and the mirror | identity dir, hash dir |
| **Synthesis group** | The up-to-three syntheses (one per target) that share one **synthesis slug**. Their shared slug directory under `cache_dir/pysr_eval/<slug>/` is a self-contained artifact — `manifest.json`, `features_full.parquet`, and one `<target>/` subdir each — and is exactly what `symbolic_regression_eval.py --eval-dir` consumes | — |
| **Target** | The schedule quantity a synthesis fits: one of `sigma`, `clip`, `mu`. Within a **synthesis group** the three targets never share an output directory; they sit side by side under the group's **synthesis slug** | — |
| **Run directory** | The PySR output location for one synthesis, `<output_directory>/<run_id>/`, holding the live checkpoint (`julia_state_stream_`) and `hall_of_fame`. Pinned via a fixed `run_id` so chained jobs reuse it; lives on `/scratch/$USER/pysr/<sweep>/<slug>/<target>/` during a job | output dir (unqualified) |
| **Persistent mirror** | The copy of a synthesis's **run directory** kept on durable shared storage (`cache_dir/pysr_eval/<slug>/<target>/pysr_run/`), refreshed by a background rsync every 15 min and on `fit()` return. The fallback a resuming job restores from if `/scratch` was purged | backup, checkpoint copy |
| **Job chain** | The sequence of SLURM jobs that together complete one synthesis: each job runs PySR for ~2h45m, then resubmits its successor (`-d` on its own job id) to continue from the checkpoint, until termination | job chaining, resubmission chain |
| **Chain depth** | How many jobs into a chain the current job is, carried in the `CHAIN_DEPTH` env var and incremented on each resubmit. The chain stops resubmitting at `--max-chain-jobs` (default 16) | — |
| **Natural completion** | A synthesis finishing its `niterations` *before* the per-job `timeout_in_seconds` elapses — detected by timing `fit()`. The non-cap termination condition: a naturally completed job does not resubmit | convergence (distinct: no stagnation test is used) |

## Flagged ambiguities

- **"weights"** appears for two unrelated concepts: **model weights** (neural network parameters, updated by DP-SGD inside the inner loop) and **schedule weights** (the learnable T-vector parameterizing the σ/C schedule, updated by the outer loop). Always qualify: say "schedule weights" or "model weights". The code uses `get_private_weights()` for schedule weights and `eqx.partition` to separate model arrays.

- **"timesteps"** was overloaded. The CLI flag `--sweep.total_timesteps` was the outer loop iteration count; it has been renamed `num_outer_steps` in the config. The inner DP-SGD step count is `num_training_steps` (config field) / `T` (mathematical notation). Always qualify: **outer step** vs **inner step** (or **T**).

- **"batch_size"** appears in two configs: `EnvConfig.batch_size` is the DP-SGD minibatch size (inner loop) and `ScheduleOptimizerConfig.batch_size` is the number of networks sampled per outer step. Use **minibatch size** for the former and **schedule batch size** for the latter.

- **"noise"** can refer to either the **noise scale σ** (a scalar per step, part of the schedule) or the **noise vector** (the actual Gaussian sample added to gradients at runtime). Prefer "noise scale" and "noise vector" rather than bare "noise".

- **"run"** is overloaded across subsystems: a **W&B run** (one outer-loop training run, keyed by a W&B run_id) versus a PySR **synthesis** (keyed by a fixed PySR `run_id` naming its **run directory**). The job-chaining for the two is also separate: `main.py` resumes a W&B run from a **W&B artifact checkpoint** keyed by `checkpoint_run_id`; a symbolic-regression **synthesis** resumes from its **run directory** / **persistent mirror** via PySR `warm_start`. Say "W&B run" or "synthesis", never bare "run".

- **"checkpoint"** likewise differs: in the outer loop it is a W&B artifact (schedule weights, optimizer state, PRNG keys) saved every `checkpoint_every` steps; in a synthesis it is PySR's `julia_state_stream_` + `hall_of_fame` in the **run directory**. They share no code path.

- **"policy"** is used loosely to mean the schedule (inherited from RL framing). The code has `PolicyAndClipScheduleConfig` (a specific schedule type where a neural network generates weights) and `ScheduleOptimizerConfig` (the outer-loop optimizer). Use **schedule** for the (σ, C) sequence object; reserve **policy** only for the `policy-and-clip` schedule variant where a network generates weights.

## Example dialogue

> **Dev:** "What exactly does the outer loop optimize?"
>
> **Domain expert:** "It optimizes the **schedule** — the sequence of noise scales σ and clipping thresholds C across all T **inner steps**. Each **outer step**, we run a **schedule batch** of fresh networks through the full **inner loop** and take the gradient of the mean **val loss** with respect to the **schedule weights**."
>
> **Dev:** "So the **schedule weights** are what get updated by SGD, not the **model weights**?"
>
> **Domain expert:** "Exactly. The **model weights** are thrown away after each inner loop — we always **reinit** a fresh network. Only the **schedule weights** persist across **outer steps**. After each SGD update to the schedule, we call **project** to make sure the resulting σ/C sequence stays within the **privacy budget**."
>
> **Dev:** "What does 'within the privacy budget' mean concretely?"
>
> **Domain expert:** "The **privacy expenditure** — `p * sqrt(sum(exp((C/σ)²) - 1))` over all T steps — must not exceed the target **GDP μ** derived from (ε, δ). The **project** operation uses bisection to push the **schedule weights** back onto that constraint surface."
>
> **Dev:** "And μ₀ is just the uniform baseline — what you'd spend per step if you spread the budget evenly?"
>
> **Domain expert:** "Right. **μ₀** is the per-step GDP μ when all T steps are identical. The schedule learns to allocate the budget non-uniformly — spending more noise in early steps and less later, or vice versa."
