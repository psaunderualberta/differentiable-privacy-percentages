"""experiments/architectures.py — Generative definitions of the architecture
*ladders* used by ``create_experiments.py``.

A *ladder* is an ordered family of network configs that holds every shape
property constant except one, so changes in the learned schedule can be
attributed to that property (width, depth, …). Ladders are produced by generator
functions keyed by their *knobs* (widths, channels, depths) rather than
enumerated by hand.

No ladder holds parameter count constant: the overlay answers a *robustness*
question ("does Learned beat Constant at every architecture"), not a scaling one,
so a rung's difference is attributable to architecture without needing to be
attributable to the knob alone (see CONTEXT.md, "Overlay", and ADR 0002).

The single source of truth is :data:`LADDERS`, mapping a ladder name to its list
of configs. ``create_experiments.py`` inverts this into ``arch -> {ladder tags}``
(deduplicating architectures shared across ladders), and tags each W&B run
``ladder:<name>`` for every ladder it belongs to. Downstream tooling
(``compile_results_fetch.py``) discovers membership generically from that prefix,
so adding a ladder here requires no changes elsewhere.
"""

from __future__ import annotations

from networks.cnn.config import CNNConfig
from networks.mlp.config import MLPConfig

# Prefix marking a W&B tag as ladder membership. Producer and consumer agree on
# this string and nothing else, so the set of ladders stays data-driven.
LADDER_TAG_PREFIX: str = "ladder:"


# ---------------------------------------------------------------------------
# Ladder generators
# ---------------------------------------------------------------------------


def width_ladder(widths: list[int], depth: int = 1) -> list[MLPConfig]:
    """Fixed depth, varying per-layer width."""
    return [MLPConfig(hidden_sizes=(w,) * depth) for w in widths]


def cnn_depth_ladder(
    channels: int,
    depths: list[int],
    head: tuple[int, ...] = (64,),
) -> list[CNNConfig]:
    """Fixed channels per layer, varying conv depth using the same-conv block.

    The same-conv block (3x3, pad 1, stride 1) is spatially size-preserving, so
    all downsampling is carried by the halving pool (ADR 0010) and each block
    halves both spatial dimensions. That caps the ladder at depth 4 on 28x28
    inputs; depth 5 is available on 32x32 but deliberately unused, so the ladder
    is identical across all three datasets.
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

_CNN_DEPTHS: list[int] = [1, 2, 3, 4]

LADDERS: dict[str, list[MLPConfig | CNNConfig]] = {
    "mlp-width": width_ladder([64, 128, 512]),
    # cnn-width: aggressive-downsampling block, fixed (64,) head.
    "cnn-width": [
        CNNConfig(
            channels=ch,
            kernel_sizes=(8, 4),
            paddings=(2, 0),
            strides=(2, 2),
            pool_kernel_size=2,
            mlp=MLPConfig(hidden_sizes=(64,)),
        )
        for ch in [(8, 16), (16, 32), (32, 64)]
    ],
    "cnn-depth": cnn_depth_ladder(16, _CNN_DEPTHS),
}
