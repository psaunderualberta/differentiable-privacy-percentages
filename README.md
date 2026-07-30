# Differentiable Privacy Percentages

**License:** MIT — Copyright (c) 2025 Paul Saunders

---

## Overview

This project learns optimal noise (σ) and gradient clipping (C) *schedules* for DP-SGD via **gradient-based outer-loop optimization**. Rather than using a fixed noise level throughout training, the schedule allocates the privacy budget (ε, δ) non-uniformly across the T training steps — spending more noise early or late where it helps most.

The outer loop treats the schedule as a policy and differentiates through the entire DP-SGD inner loop (via `jax.lax.scan` + `jax.checkpoint`) to update the schedule's learnable parameters. Privacy constraints are enforced after each outer update by projecting the weights back onto the valid Gaussian DP (GDP) constraint set.

**Key ideas:**
- Privacy budget is expressed in GDP (μ) and converted from (ε, δ)-DP at startup.
- Weights parameterize a discrete distribution over T steps; their exponentiated squares must sum to a value determined by (ε, δ, T).
- Projection onto the constraint set is implemented in JAX `while_loop` (fully JIT-compatible) using nested bisection + Newton's method.
- The outer loop is parallelised across GPUs via `shard_map`, training a batch of randomly-initialised networks per step to reduce variance.
- Experiment tracking is done with Weights & Biases (W&B); hyperparameter sweeps and SLURM job submission are supported.

---

## Call Graph

Standard execution flow through `main.py`:

```
main()
├── SingletonConfig.get_*_config_instance()        config singleton (tyro CLI)
├── get_dataset_shapes()                            load/cache dataset from HuggingFace
├── get_privacy_params(N)                           approx_to_gdp(ε,δ) → GDPPrivacyParameters
├── policy_factory(schedule_conf, gdp_params)       build schedule (eqx.Module)
│   ├── schedule_factory(conf, privacy_params)      non-stateful schedule variants
│   └── stateful_schedule_factory(conf, ...)        stateful schedule variants
├── DP_RL_Params.create_direct_from_config()        frozen DP-SGD env params
├── WandbTableLogger()                              accumulates per-step W&B tables
├── get_optimal_mesh(devices, batch_size)           JAX device mesh over GPUs
├── make_policy_loss_fn(mesh, env_params)           JIT+shard_map compiled loss fn
│   └── [returns get_policy_loss — called each step]
│       ├── [shard_map: parallel across GPU devices]
│       └── vmapped_train_with_noise(...)           vmap over noise_keys
│           └── train_with_noise(schedule, env_params, mb_key, init_key, noise_key)
│               ├── reinit_model(network, key)      fresh random network weights
│               └── jax.lax.scan(scanned_training_step, T steps)
│                   └── training_step(model, opt_state, batch, σ, clip)
│                       ├── vmapped_loss(model, batch_x, batch_y)   per-sample grads
│                       ├── clip_grads_abadi(grads, clip)           Abadi norm clip
│                       ├── get_spherical_noise(clipped, σ, clip)   Gaussian noise
│                       └── optimizer.update(noised_grads, ...)     inner model step
├── optax.sgd(lr, momentum) + optimizer.init(schedule)
├── [load_checkpoint() — if checkpoint_run_id set]
├── init_wandb_run(wandb_config, sweep_config)
│
└── for t in range(total_timesteps):               outer RL loop
    ├── schedule.get_loggables()                   log σ/clip/weights to W&B table
    ├── get_policy_loss(schedule, mb_key, init_key, noise_keys)
    │   [eqx.filter_value_and_grad — returns loss and ∂loss/∂schedule]
    │   └── → (loss, (train_losses, train_accs, val_accs)), grads
    ├── wandb.log(val_loss, val_accuracy, ...)
    ├── ensure_valid_pytree(loss, grads)            NaN/Inf guard (outside JIT)
    ├── optimizer.update(grads, opt_state)          outer SGD step on schedule
    ├── schedule.apply_updates(updates)             apply outer gradients
    ├── schedule.project()                          enforce GDP privacy constraint
    ├── [save_checkpoint() — every checkpoint_every steps]
    └── [baseline.log_comparison() — if with_baselines]
```

---

## Repository Structure

```
.
├── src/                        # All Python source code
│   ├── main.py                 # Entry point: outer training loop
│   ├── sweep.py                # W&B sweep creation + agent
│   ├── conf/                   # Config dataclasses (tyro CLI) + singleton
│   ├── privacy/                # GDP privacy accounting + weight projection
│   ├── policy/                 # Schedule implementations (eqx.Module)
│   │   ├── schedules/          # Differentiable schedule variants
│   │   └── stateful_schedules/ # Schedules that update from runtime stats
│   ├── environments/           # DP-SGD inner loop (dp.py)
│   ├── networks/               # MLP / CNN definitions + registry
│   ├── util/                   # Dataloaders, logging, misc utilities
│   └── test/                   # pytest test suite
├── cc/
│   ├── slurm/                  # SLURM job submission scripts
│   └── sweeps/                 # Saved sweep run-ID files
├── pyproject.toml
└── LICENSE
```

---

## Installation

[uv](https://github.com/astral-sh/uv) is used for dependency management.

```bash
# Install uv if not already available
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies (creates a virtual environment automatically)
cd differentiable-privacy-percentages
uv sync
```

A CUDA 12-compatible GPU is expected. CPU-only execution is possible but very slow.

---

## Usage

All commands must be run from the `src/` directory.

### Training a single run

```bash
cd src && uv run main.py \
    --sweep.env.eps 1.0 \
    --sweep.env.delta 1e-5 \
    --sweep.dataset mnist \
    --sweep.total_timesteps 200
```

Common arguments:

| Argument | Default | Description |
|---|---|---|
| `--sweep.env.eps` | — | Privacy budget ε |
| `--sweep.env.delta` | — | Privacy budget δ |
| `--sweep.dataset` | `mnist` | Dataset (`mnist`, `fashion-mnist`, `cifar-10`, `california`, `eyepacs`) |
| `--sweep.total_timesteps` | — | Outer-loop iterations |
| `--sweep.policy.schedule_type` | `alternating_schedule` | Schedule variant |
| `--wandb_conf.mode` | `online` | W&B mode (`online`, `offline`, `disabled`) |

Debug config parsing without running:

```bash
cd src && uv run conf/singleton_conf.py --sweep.env.eps 1.0 --sweep.env.delta 1e-5 --sweep.dataset mnist --sweep.total_timesteps 100
```

### Running tests

```bash
# All tests
cd src && uv run pytest

# Single test file
cd src && uv run pytest test/test_privacy.py

# Single test
cd src && uv run pytest test/test_privacy.py::test_approx_to_gdp
```

### W&B hyperparameter sweeps

```bash
cd src && uv run sweep.py \
    --wandb_conf.project "MyProject" \
    --wandb_conf.entity <entity> \
    --wandb_conf.mode online \
    --sweep.total_timesteps 2000 \
    --sweep.env.eps 1.0 \
    --sweep.env.delta 1e-5 \
    --sweep.dataset mnist
```

This creates a W&B sweep and starts a local agent. Run IDs are logged to `cc/sweeps/<sweep_id>.txt` for later SLURM submission.

### SLURM job submission

```bash
# Submit a single run by W&B run ID
cd src && uv run ../cc/slurm/run-starter.py --run_id <wandb_run_id> --runtime.days 1

# Batch-submit all runs from a sweep file
cat cc/sweeps/<sweep_id>.txt | parallel -q uv run cc/slurm/run-starter.py --run_id={} --jobname='"<name>"'
```

---

## Schedule Types

Select with `--sweep.policy.schedule`:

| Config class | Description |
|---|---|
| `AlternatingSigmaAndClipScheduleConfig` | Alternately optimises σ and clip; recommended default |
| `WarmupAlternatingSigmaAndClipScheduleConfig` | Same, with a fixed-noise warmup phase |
| `SigmaAndClipScheduleConfig` | Jointly optimises σ and clip |
| `WarmupSigmaAndClipScheduleConfig` | Same, with a fixed-noise warmup phase |
| `ParallelSigmaAndClipScheduleConfig` | Parallel optimisation of σ and clip |
| `WarmupParallelSigmaAndClipScheduleConfig` | Same, with a fixed-noise warmup phase |
| `PolicyAndClipScheduleConfig` | A small policy network generates the σ weights |
| `DynamicDPSGDScheduleConfig` | Dynamic DP-SGD baseline |
| `MedianGradientStatefulScheduleConfig` | Stateful schedule updated from per-step gradient medians |

---

## License

MIT License. See [LICENSE](LICENSE) for full text.
