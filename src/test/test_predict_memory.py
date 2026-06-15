"""Regression tests for the conv-activation heuristic in ``predict_memory.py``.

The heuristic is a safety fallback used only when the exact AOT-compile path
OOMs on the prediction machine. For CNNs the param-count model alone is
structurally wrong (tiny weight count, but peak memory dominated by per-step
conv activations stacked across the T-step scan), so a conv activation term is
added. These tests pin that term against XLA-compiled peaks measured on a GPU
backend for the default / width / depth CNN ladders at T=1000 and T=5000.

The contract is one-directional: the heuristic must never request a *smaller*
power-of-two memory tier than the run actually needs (under-provisioning would
OOM the real SLURM run). Over-provisioning by one tier is allowed.
"""

import math

import pytest

import predict_memory as pm
from networks.cnn.CNN import CNN
from networks.cnn.config import CNNConfig
from networks.mlp.config import MLPConfig

# (N, C, H, W) / (N, nclasses) for the 28x28 single-channel datasets the CNN
# ladders run on (mnist / fashion-mnist).
_INPUT_SHAPE = (1, 1, 28, 28)
_OUTPUT_SHAPE = (1, 10)
_BATCH = 250  # create_experiments.BATCH_SIZE

# Architectures from create_experiments / experiments.architectures.
_CASES = {
    "default(16,32)": CNNConfig(
        channels=(16, 32),
        kernel_sizes=(8, 4),
        paddings=(2, 0),
        strides=(2, 2),
        pool_kernel_size=2,
        mlp=MLPConfig(hidden_sizes=(32,)),
    ),
    "width(64,128)": CNNConfig(
        channels=(64, 128),
        kernel_sizes=(8, 4),
        paddings=(2, 0),
        strides=(2, 2),
        pool_kernel_size=2,
        mlp=MLPConfig(hidden_sizes=(64,)),
    ),
    "depth(32x3)": CNNConfig(
        channels=(32, 32, 32),
        kernel_sizes=(3, 3, 3),
        paddings=(1, 1, 1),
        strides=(1, 1, 1),
        pool_kernel_size=2,
        mlp=MLPConfig(hidden_sizes=(64,)),
    ),
}

# Exact XLA-compiled peaks (GiB) measured on the GPU backend — the ground truth.
_COMPILED_GIB = {
    "default(16,32)": {1000: 1.34, 5000: 6.71},
    "width(64,128)": {1000: 3.68, 5000: 10.62},
    "depth(32x3)": {1000: 14.57, 5000: 36.90},
}


def _tier(gib: float) -> int:
    return 1 << math.ceil(math.log2(max(gib, 1.0)))


def _heuristic_mem(cfg: CNNConfig, T: int) -> int:
    net = CNN.build(cfg, _INPUT_SHAPE, _OUTPUT_SHAPE, key=0)
    P = pm._param_count(net)
    fm = pm._conv_feature_map_elements(cfg, _INPUT_SHAPE[-2], _INPUT_SHAPE[-1])
    peak = pm._heuristic_bytes(1, T, _BATCH, P, fm)
    return pm._next_pow2_gib(peak, 1)


@pytest.mark.parametrize("name", list(_CASES))
@pytest.mark.parametrize("T", [1000, 5000])
def test_conv_heuristic_never_under_provisions(name, T):
    """The heuristic's memory tier must be >= the compiled run's tier."""
    mem = _heuristic_mem(_CASES[name], T)
    needed = _tier(_COMPILED_GIB[name][T])
    assert mem >= needed, (
        f"{name} T={T}: heuristic requests {mem}G but compiled run needs {needed}G "
        f"(compiled peak {_COMPILED_GIB[name][T]} GiB) — under-provision would OOM."
    )


def test_default_cnn_at_t5000_is_the_calibration_floor():
    """The default CNN at T=5000 is the case the param-only model under-provisioned;
    pin it to the 8G tier so a regression in alpha is caught."""
    assert _heuristic_mem(_CASES["default(16,32)"], 5000) == 8


def test_mlp_has_no_conv_activation_term():
    """Non-conv configs contribute zero conv feature-map elements."""
    assert pm._conv_feature_map_elements(MLPConfig(hidden_sizes=(128,)), 28, 28) == 0
