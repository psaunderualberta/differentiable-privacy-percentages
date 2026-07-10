"""experiments/architectures.py — Generative definitions of the architecture
*ladders* used by ``create_experiments.py``.

A *ladder* is an ordered family of network configs that holds every shape
property constant except one, so changes in the learned schedule can be
attributed to that property (width, depth, …). Ladders are produced by
generator functions keyed by their *knobs* (widths, depths, channels) rather
than enumerated by hand — in particular the param-matched depth ladder computes
its widths from the shared anchor instead of hard-coding them.

The single source of truth is :data:`LADDERS`, mapping a ladder name to its list
of configs. ``create_experiments.py`` inverts this into ``arch -> {ladder tags}``
(deduplicating architectures shared across ladders, e.g. the anchor), and tags
each W&B run ``ladder:<name>`` for every ladder it belongs to. Downstream tooling
(``compile_results_fetch.py``) discovers membership generically from that prefix,
so adding a ladder here requires no changes elsewhere.
"""

from __future__ import annotations

from networks.cnn.config import CNNConfig
from networks.mlp.config import MLPConfig

# MLP ladders run on the 28x28, 10-class datasets (mnist / fashion-mnist), which
# share these dimensions; param-matching is computed against them.
_MLP_DIN: int = 28 * 28
_MLP_NCLASSES: int = 10

# Prefix marking a W&B tag as ladder membership. Producer and consumer agree on
# this string and nothing else, so the set of ladders stays data-driven.
LADDER_TAG_PREFIX: str = "ladder:"


# ---------------------------------------------------------------------------
# Parameter counting (mirrors compile_results_fetch._mlp_param_count exactly,
# including the extra 2*b affine params per hidden layer)
# ---------------------------------------------------------------------------


def _mlp_param_count(din: int, hidden_sizes: tuple[int, ...], nclasses: int) -> int:
    sizes = [din, *hidden_sizes, nclasses]
    total = 0
    for i in range(len(sizes) - 1):
        a, b = sizes[i], sizes[i + 1]
        total += a * b + b
        if i < len(sizes) - 2:
            total += 2 * b
    return total


def _solve_width(target_params: int, depth: int, din: int, nclasses: int) -> int:
    """Smallest-error constant width for a ``depth``-layer MLP near ``target_params``.

    Searches integer widths and returns the one whose param count is closest to
    the target, so the param-matched ladder tracks the anchor even if the anchor
    or param formula changes.
    """
    best_w, best_err = 1, None
    for w in range(1, target_params):
        err = abs(_mlp_param_count(din, (w,) * depth, nclasses) - target_params)
        if best_err is None or err < best_err:
            best_w, best_err = w, err
        elif err > best_err:
            # Param count is monotonic in width, so once the error grows we are past the optimum.
            break
    return best_w


# ---------------------------------------------------------------------------
# Ladder generators
# ---------------------------------------------------------------------------


def width_ladder(widths: list[int], depth: int = 1) -> list[MLPConfig]:
    """Fixed depth, varying per-layer width."""
    return [MLPConfig(hidden_sizes=(w,) * depth) for w in widths]


def depth_ladder(width: int, depths: list[int]) -> list[MLPConfig]:
    """Fixed per-layer width, varying depth (total params grow with depth)."""
    return [MLPConfig(hidden_sizes=(width,) * d) for d in depths]


def param_matched_depth_ladder(
    target_params: int,
    depths: list[int],
    din: int = _MLP_DIN,
    nclasses: int = _MLP_NCLASSES,
) -> list[MLPConfig]:
    """Varying depth with per-layer width shrunk so total params stay ~target."""
    return [
        MLPConfig(hidden_sizes=(_solve_width(target_params, d, din, nclasses),) * d) for d in depths
    ]


def cnn_depth_ladder(
    channels: int,
    depths: list[int],
    head: tuple[int, ...] = (64,),
) -> list[CNNConfig]:
    """Fixed channels per layer, varying conv depth using the same-conv block.

    The same-conv block (3x3, pad 1, stride 1, pool 2) is spatially gentle so
    conv layers stack to depth 4 on 28x28 inputs, unlike the default
    aggressive-downsampling block used by ``cnn-width``.
    """
    return [
        CNNConfig(
            channels=(channels,) * d,
            kernel_sizes=(3,) * d,
            paddings=(1,) * d,
            strides=(1,) * d,
            pool_kernel_size=2,
            mlp=MLPConfig(hidden_sizes=head),
        )
        for d in depths
    ]


# ---------------------------------------------------------------------------
# The ladder registry — single source of truth
# ---------------------------------------------------------------------------

# Anchor: width-128 / depth-1 MLP, shared by all three MLP ladders. The
# param-matched ladder holds total params equal to this point.
_ANCHOR_WIDTH: int = 128
_ANCHOR_PARAMS: int = _mlp_param_count(_MLP_DIN, (_ANCHOR_WIDTH,), _MLP_NCLASSES)

_MLP_DEPTHS: list[int] = [1, 2, 3, 4]
_CNN_DEPTHS: list[int] = [1, 2, 3, 4]

LADDERS: dict[str, list[MLPConfig | CNNConfig]] = {
    "mlp-width": width_ladder([64, 128, 256, 512]),
    "mlp-depth": depth_ladder(_ANCHOR_WIDTH, _MLP_DEPTHS),
    # cnn-width: existing aggressive-downsampling block, fixed (64,) head.
    "cnn-width": [
        CNNConfig(
            channels=ch,
            kernel_sizes=(8, 4),
            paddings=(2, 0),
            strides=(2, 2),
            pool_kernel_size=2,
            mlp=MLPConfig(hidden_sizes=(64,)),
        )
        for ch in [(8, 16), (16, 32), (32, 64), (64, 128)]
    ],
    "cnn-depth": cnn_depth_ladder(16, _CNN_DEPTHS),
}
