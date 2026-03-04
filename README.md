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
| `--sweep.dataset` | `mnist` | Dataset (`mnist`, `fashion-mnist`, `cifar-10`, `california`) |
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

Select with `--sweep.policy.schedule_type`:

| Type | Description |
|---|---|
| `alternating_schedule` | Alternately optimises σ and clip; recommended default |
| `sigma_and_clip_schedule` | Jointly optimises σ and clip |
| `policy_and_clip_schedule` | A small policy network generates the σ weights |
| `dynamic_dpsgd_schedule` | Dynamic DP-SGD baseline |
| `stateful_median_schedule` | Updates schedule based on per-step gradient medians |

---

## License

MIT License. See [LICENSE](LICENSE) for full text.
