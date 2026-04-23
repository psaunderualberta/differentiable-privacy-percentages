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

## Flagged ambiguities

- **"weights"** appears for two unrelated concepts: **model weights** (neural network parameters, updated by DP-SGD inside the inner loop) and **schedule weights** (the learnable T-vector parameterizing the σ/C schedule, updated by the outer loop). Always qualify: say "schedule weights" or "model weights". The code uses `get_private_weights()` for schedule weights and `eqx.partition` to separate model arrays.

- **"timesteps"** was overloaded. The CLI flag `--sweep.total_timesteps` was the outer loop iteration count; it has been renamed `num_outer_steps` in the config. The inner DP-SGD step count is `num_training_steps` (config field) / `T` (mathematical notation). Always qualify: **outer step** vs **inner step** (or **T**).

- **"batch_size"** appears in two configs: `EnvConfig.batch_size` is the DP-SGD minibatch size (inner loop) and `ScheduleOptimizerConfig.batch_size` is the number of networks sampled per outer step. Use **minibatch size** for the former and **schedule batch size** for the latter.

- **"noise"** can refer to either the **noise scale σ** (a scalar per step, part of the schedule) or the **noise vector** (the actual Gaussian sample added to gradients at runtime). Prefer "noise scale" and "noise vector" rather than bare "noise".

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
