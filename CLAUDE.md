# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands must be run from `src/` as the working directory (it is not a Python package root; imports are relative to `src/`).

**Run the main training script:**
```bash
cd src && uv run main.py --sweep.env.eps 0.5 --sweep.env.delta 1e-7 --sweep.dataset mnist --sweep.total_timesteps 100
```

**Run tests:**
```bash
cd src && uv run pytest
```

**Run a single test file:**
```bash
cd src && uv run pytest test/test_privacy.py
```

**Run a single test:**
```bash
cd src && uv run pytest test/test_privacy.py::test_approx_to_gdp
```

**Launch a W&B sweep** (creates sweep, then starts an agent that logs run IDs to `cc/sweeps/<sweep_id>.txt`):
```bash
cd src && uv run sweep.py --wandb_conf.project "MyProject" --wandb-conf.entity <entity> --wandb-conf.mode online --sweep.total_timesteps 2000 ...
```

**Submit a sweep run to SLURM** (reads config from a prior W&B run):
```bash
# Submit one run by ID:
cd src && uv run ../cc/slurm/run-starter.py --run_id <wandb_run_id> --runtime.days 1

# Batch submit from a sweep file:
cat cc/sweeps/<sweep_id>.txt | parallel -q uv run cc/slurm/run-starter.py --run_id={} --jobname='"<name>"'
```

**Print parsed config (useful for debugging CLI args):**
```bash
cd src && uv run conf/singleton_conf.py <same args as main.py>
```

## Architecture

### Entry Points

- **`src/main.py`** — The main training loop. Reads config via `SingletonConfig`, builds the schedule/policy, runs the gradient-based RL outer loop, and logs to W&B.
- **`src/sweep.py`** — Creates a W&B sweep from the current config, then starts a W&B agent that records run IDs for later SLURM submission.
- **`cc/slurm/run-starter.py`** — Generates and submits a SLURM `sbatch` script. Takes a `--run_id` (W&B run ID) and downloads that run's config to populate `main.py` args.

### Configuration System

All config is a single `Config` dataclass tree parsed by `tyro` CLI at startup. The singleton `SingletonConfig` (in `conf/singleton_conf.py`) holds the parsed config and is accessed globally throughout the codebase — no config is passed as function arguments at the module level.

- `Config.sweep` → `SweepConfig` (top-level experiment settings: dataset, total timesteps, policy, env)
  - `SweepConfig.env` → `EnvConfig` (private network: architecture, optimizer, ε, δ, batch size, T)
  - `SweepConfig.policy` → `PolicyConfig` (schedule type, LR, momentum, batch size)
- `Config.wandb_conf` → `WandbConfig` (project, entity, mode, optional restart run ID)

`DistributionConfig` is used for hyperparameters that can be either constants or sampled distributions (for W&B sweeps). `.sample()` returns a float.

When `wandb_conf.restart_run_id` is set, `singleton_conf.py` downloads and merges that run's config on top of CLI defaults — this is how SLURM reruns a sweep agent's config.

### The Outer Training Loop (main.py)

```
for t in total_timesteps:
    1. Log current schedule (sigmas, clips, weights) to W&B
    2. Compute policy loss:
       - shard_map over GPUs × policy_batch_size networks
       - vmap over noise_keys within each shard
       - Each network: train_with_noise() → runs T DP-SGD steps via jax.lax.scan
       - Loss = mean val_loss across all networks
    3. Backprop through the loss into the schedule's learnable parameters
    4. SGD update on schedule
    5. schedule.project() → enforce privacy constraints
```

The `get_policy_loss` function is decorated with `@eqx.filter_jit`, `@eqx.filter_value_and_grad`, and `@shard_map` — the gradient flows back through the JAX scan and into the schedule weights.

### Privacy Accounting (`privacy/gdp_privacy.py`)

The core abstraction is `GDPPrivacyParameters`, an `eqx.Module` holding (ε, δ, p, T) and derived GDP parameters. Key conversions:

- **Weights** → arrays of non-negative values that sum to T; the learnable representation of the schedule.
- **`weights_to_mu_schedule(weights)`** → per-step GDP μ values.
- **`weights_to_sigma_schedule(C, weights)`** → per-step noise σ = C/μ.
- **`weights_to_clip_schedule(sigmas, weights)`** → per-step clip = σ × μ.
- **`project_weights(weights)`** → project raw gradient-updated weights back onto the valid constraint set (∑e^(wᵢ²) = (μ/p)² + T) using nested bisection + Newton's method, fully implemented in JAX `while_loop` so it is JIT-compatible.

`approx_to_gdp(eps, delta)` converts (ε, δ)-DP to GDP μ via Brent's method (scipy, runs once at startup outside JAX).

### Schedule / Policy (`policy/`)

All schedules implement `AbstractNoiseAndClipSchedule` (an `eqx.Module`):
- `get_private_sigmas()` / `get_private_clips()` — return length-T arrays used directly in DP-SGD.
- `apply_updates(updates)` — apply optax gradient updates; handles `stop_gradient` logic for alternating optimization.
- `project()` — project back to valid privacy space after each gradient step.
- `get_loggables()` / `get_logging_schemas()` — W&B logging hooks.

**Schedule types** (selected by `--sweep.policy.schedule_type`):
| Type | Description |
|---|---|
| `alternating_schedule` | Optimizes σ and clip alternately; each `project()` call flips which is differentiated. Default. |
| `sigma_and_clip_schedule` | Jointly optimizes σ and clip. |
| `policy_and_clip_schedule` | A policy network generates weights → σ. |
| `dynamic_dpsgd_schedule` | Dynamic DP-SGD variant. |
| `stateful_median_schedule` | Stateful schedule that updates based on gradient medians at runtime. |

**Base schedules** (`policy/base_schedules/`) are the parametric forms underlying σ and clip:
- `ConstantSchedule` — single learnable scalar broadcast to T steps.
- `InterpolatedExponentialSchedule` — exponential curve between learned min/max.
- `ClippedSchedule` — polynomial/linear interpolation with bounds.

### DP-SGD Training (`environments/dp.py`)

`train_with_noise(schedule, params, mb_key, init_key, noise_key)`:
1. Extracts σ and clip arrays from the schedule.
2. Reinitializes a fresh network.
3. Runs `jax.lax.scan` over T steps (`scanned_training_step`), each step:
   - Poisson-samples a minibatch (`sample_batch_uniform`).
   - Computes per-sample gradients via `vmapped_loss`.
   - Clips gradients (`clip_grads_abadi` — Abadi global norm clipping).
   - Adds spherical Gaussian noise (`get_spherical_noise`).
   - Updates model via optax.
4. Returns final model, val loss, train losses, train accuracies, val accuracy.

`jax.checkpoint` with `dots_with_no_batch_dims_saveable` policy is used inside the scan to control memory during backprop through T steps.

`train_with_stateful_noise` is the equivalent for stateful schedules, where σ/clip are computed dynamically per step from gradient statistics.

### Networks (`networks/`)

- `MLPConfig` / `CNNConfig` select architecture. Default is MLP.
- Networks are `eqx.Module`s; `reinit_model(network, key)` re-randomizes weights while preserving architecture, used to sample fresh networks each RL timestep.
- `eqx.partition` / `eqx.combine` separate array leaves from static structure for shard_map and scan compatibility.

### Datasets (`util/dataloaders.py`)

Datasets are downloaded from Hugging Face on first use and cached as `.npy` files in `src/data/<dataset-name>/`. Supported: `mnist`, `fashion-mnist`, `cifar-10`, `california` (tabular regression treated as binary classification).

### Logging (`util/logger.py`)

`WandbTableLogger` accumulates arrays into W&B tables (one row per timestep) and produces line plots at the end. Schedules declare their logging schema via `get_logging_schemas()`. Direct `wandb.log()` calls in `main.py` handle scalar metrics (val-loss, val-accuracy, etc.).

## JAX Patterns

- All hot-path code is `@eqx.filter_jit` compiled. Avoid Python-level conditionals on arrays inside JIT regions; use `jax.lax.select` / `jax.lax.cond`.
- `eqx.error_if` is used for runtime NaN/Inf checks that survive JIT compilation.
- `lax.pvary` is called on network params within `shard_map` to give each device a different random init.
- `ensure_valid_pytree` (in `util/util.py`) checks for NaN/Inf at Python level (outside JIT) after each outer loop step.
