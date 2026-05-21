# dp_psac_ref

Standalone reference implementation of **DP-PSAC** (Xia et al., "Differentially
Private Learning with Per-Sample Adaptive Clipping", AAAI 2023, arXiv:2212.00328)
for independently validating results from the main `src/` codebase.

Shares no code with `src/`. Own `pyproject.toml`, own dataset loader, own
accountant. Accepts arbitrary per-step schedules for noise (σ) and clip (C),
not just constants.

## Clip rule

```
g̃_{t,i} = C_t · g_{t,i} / ( ||g_{t,i}|| + r / (||g_{t,i}|| + r) )
ĝ_t     = Σ_i g̃_{t,i} + 𝒩(0, C_t² · σ_t² · I)
θ_{t+1} = θ_t − (η / B) · ĝ_t
```

## Install

```bash
cd dp_psac_ref
uv sync
```

## Run

Schedules are supplied as two `.npy` files of equal length `T`:

```bash
# Constant schedule sanity: DP-PSAC @ MNIST CNN (paper Table 2 row)
python -c "import numpy as np; np.save('sigmas.npy', np.full(1000, 1.23)); np.save('clips.npy', np.full(1000, 0.1))"

uv run run.py --sigmas sigmas.npy --clips clips.npy \
              --dataset mnist --arch cnn \
              --batch-size 512 --lr 1.0 --r 0.1 --delta 1e-5 --seed 0
```

Per-step schedules (e.g. exported from the main repo) work transparently —
just ensure `len(sigmas) == len(clips) == T`.

## Tests

```bash
uv run pytest test_dp_psac.py -v
```

Covers: clip norm bound (`||g̃|| ≤ C`), small-grad limit (`g̃ → C·g`),
large-grad limit (`g̃ → C·g/||g||`), exact formula, per-step schedule indexing,
accountant monotonicity.

## Output

`run.py` prints JSON to stdout (and optionally writes `--out results.json`):

```json
{
  "test_accuracy": 0.9823,
  "final_train_loss": 0.112,
  "epsilon_spent": 3.01,
  "delta": 1e-5,
  "T": 1000, "B": 512, "q": 0.00853,
  "r": 0.1, "lr": 1.0,
  "dataset": "mnist", "arch": "cnn", "seed": 0
}
```

## Cross-checking the main repo

Export a learned (σ, C) schedule from a `src/main.py` run as two `.npy` arrays,
feed them here, and compare `test_accuracy` and `epsilon_spent` against the
in-tree run. Divergence isolates bugs to (a) the clip rule, (b) the accountant,
or (c) inner-loop wiring.
