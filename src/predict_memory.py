#!/usr/bin/env python3
"""predict_memory.py — Predict per-run GPU memory for the experiments created by
``create_experiments.py`` and emit a ``run_id -> --mem-per-gpu`` map.

Memory is obtained *near-exactly* and *without running* by ahead-of-time (AOT)
lowering the outer-loop compiled region (``make_training_loss_fn``) for each
config and reading XLA's compiled-memory analysis.  For the configs this repo
produces, peak memory depends only on a handful of fields — dataset, network
architecture, T (= ``num_training_steps``), batch size, ``scan_segments``, the
analytic vmap width (``schedule_optimizer.batch_size``) and the schedule type —
so runs that share that *signature* are compiled once and the result is mapped
back to every matching ``run_id`` (ε and seed do not affect memory).

The emitted memory is the smallest power-of-two gibibytes >= the measured peak.

The sweep file is the self-describing header format written by
``create_experiments.py`` (``run_id<TAB>mem_per_gpu`` ...); this tool *refines*
the ``mem_per_gpu`` column in place, preserving all other columns and row order.
Legacy bare run-ID lists are accepted as input and gain a ``mem_per_gpu`` column.

Usage (from src/, ideally on a GPU node so the analysis targets the GPU backend):
    uv run predict_memory.py --entity <entity> --project <project> \\
        --sweep-file cc/sweeps/<file>.txt

Submit with (reads the header columns):
    parallel --colsep '\\t' --header : -q uv run cc/slurm/run-starter.py \\
        --run_id={run_id} --mem_per_gpu={mem_per_gpu} \\
        --jobname='"<name>"' :::: cc/sweeps/<file>.txt
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.random as jr
import tyro

from conf.config import SweepConfig

# Mirror the PROJECT_ROOT convention used by sweep.py / create_experiments.py.
os.environ.setdefault(
    "PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cc")),
)

from conf.config import Config, WandbConfig
from conf.scope import RunContext, using
from conf.singleton_conf import (
    SingletonConfig,
    _reconstruct_from_dict,
    get_wandb_run_conf,
)
from create_experiments import _make_sweep_config
from environments.dp_params import DPTrainingParams
from environments.outer_loop import make_initial_es_state, make_training_loss_fn
from networks.cnn.config import CNNConfig
from networks.mlp.config import MLPConfig
from networks.net_factory import resolve_network_config
from policy.factory import make_schedule
from privacy.gdp_privacy import get_privacy_params
from util.dataloaders import get_dataset_shapes
from util.sweep_file import read_sweep_file, write_sweep_file


@dataclass
class PredictConfig:
    entity: str
    project: str
    sweep_file: str
    """Path to a sweep .txt of run IDs (one per line), e.g. cc/sweeps/<file>.txt."""
    out: str | None = None
    """Where to write the updated header file. Defaults to rewriting --sweep-file
    in place (the mem_per_gpu column is replaced; all other columns and the row
    order are preserved)."""
    min_gib: int = 8
    """Floor for the emitted power-of-two memory request."""
    max_gib: int = 32
    """Ceil for the emitted power-of-two memory request."""
    safety_factor: float = 1.0
    """Multiply the measured peak before rounding up (headroom for runtime/host
    allocations XLA's static analysis does not capture)."""
    heuristic_only: bool = False
    """Skip AOT compilation entirely and use the param-count heuristic for every
    signature. Useful when the prediction machine is known to be smaller than the
    target (compilation would OOM) or for a fast, no-GPU estimate."""


def _next_pow2_gib(num_bytes: float, min_gib: int, max_gib: int) -> int:
    """Smallest power-of-two gibibytes >= num_bytes (>= min_gib)."""
    gib = max(num_bytes / 2**30, float(min_gib))
    return min(max(min_gib, 1 << math.ceil(math.log2(gib))), max_gib)


def _signature(sweep: SweepConfig) -> tuple:
    """Memory-relevant fingerprint of a SweepConfig.

    Excludes ε, δ, seed and learning rate, which do not change the compiled
    region's buffer sizes — only the data/network shapes, rollout length and
    parallel/segment structure do.
    """
    env = sweep.env
    return (
        sweep.dataset,
        repr(env.network),
        env.num_training_steps,
        env.batch_size,
        env.scan_segments_derived,
        env.microbatch_size_derived,
        repr(type(env.optimizer).__name__),
        sweep.schedule_optimizer.batch_size,
        repr(type(sweep.schedule_optimizer.schedule).__name__),
        sweep.schedule_optimizer.es.enabled,
    )


# --- Heuristic fallback model -------------------------------------------------
# Validated against AOT-compiled peaks for the dataset-default MLP (P≈408k params)
# on a GPU backend: peak ≈ FIXED_BASE + axis·(T·P + B·P)·4B, where the T·P term is
# the stacked (T,P) scan carry kept for backprop-through-T and B·P is one
# rematerialized step's per-sample gradients. 4 bytes/param (fp32) is the
# conservative upper bound on the stacked carry — the checkpoint policy elides
# some of it — so the heuristic over-estimates by design (it only fires when the
# exact path OOMs, where erring high beats crashing the real run). CNN activation
# feature maps are not modelled, so bump --safety_factor for conv-heavy runs.
_HEURISTIC_BYTES_PER_PARAM_STEP = 4  # fp32; also bytes per fp32 activation element
_HEURISTIC_FIXED_BASE_BYTES = 256 * 2**20  # runtime + dataset host->device staging
# Convs break the param-count model: their weight count is tiny but peak memory
# is dominated by per-sample conv-layer *activations*, which the scan stacks
# across all T steps for backprop-through-T — so a small-param CNN's peak grows
# strongly with T (the default CNN climbs 1.34->6.71 GiB over T=1000->5000)
# entirely from activations the param term cannot see. The conv fallback
# therefore adds an activation term ``alpha · axis · T · B · (Σ per-sample conv
# feature-map elements) · 4B`` on top of the param term.
#
# alpha is the per-element fraction of the activation stack XLA keeps live at the
# peak; it is not constant across architectures (aggressive-downsampling nets
# like the default need ~0.5, gentler width/depth ladders ~0.11-0.18 once their
# param term is accounted for), so alpha=0.30 is chosen as the smallest value
# that keeps the worst case (default CNN) safe. Verified against compiled peaks
# for the default / width / depth CNN ladders at T=1000 and T=5000: every case
# lands on its correct power-of-two tier or one tier high (width@5000: 16G->32G,
# the safe direction), none under-provisions. This is a coarse safety net that
# only fires on compile OOM — the exact compiled path is preferred for CNNs.
_HEURISTIC_CONV_ACT_ALPHA = 0.30


def _param_count(network) -> int:
    """Total number of array elements (parameters) in an equinox network."""
    return sum(int(x.size) for x in jax.tree.leaves(eqx.filter(network, eqx.is_array)))


def _conv_feature_map_elements(network_conf, height: int, width: int) -> int:
    """Sum over conv layers of per-sample post-conv feature-map element counts.

    Zero for non-conv networks. Mirrors ``CNN.from_config``'s spatial arithmetic:
    each ``Conv2d`` maps a ``D``-dim axis to ``(D + 2·pad − kernel)//stride + 1``,
    then ``MaxPool2d`` floor-divides both spatial axes by ``pool_kernel_size``.
    The post-conv (pre-pool) map is the largest activation per layer and the one
    stacked across the scan, so it is what the T-scaling activation term tracks.
    """
    if not isinstance(network_conf, CNNConfig):
        return 0
    total = 0
    h, w = height, width
    for ch, k, p, s in zip(
        network_conf.channels,
        network_conf.kernel_sizes,
        network_conf.paddings,
        network_conf.strides,
    ):
        h = (h + 2 * p - k) // s + 1
        w = (w + 2 * p - k) // s + 1
        total += ch * h * w
        h //= network_conf.pool_kernel_size
        w //= network_conf.pool_kernel_size
    return total


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "resource_exhausted" in msg or "out of memory" in msg or "out-of-memory" in msg


def _heuristic_bytes(
    parallel_axis_size: int, T: int, batch_size: int, P: int, conv_fm_elements: int
) -> int:
    per_run = _HEURISTIC_BYTES_PER_PARAM_STEP * (T * P + batch_size * P)
    # Conv activation stack: alpha-fraction of T steps' per-sample feature maps
    # kept live for backprop-through-T. Zero for non-conv nets (conv_fm=0).
    per_run += int(
        _HEURISTIC_CONV_ACT_ALPHA
        * T
        * batch_size
        * conv_fm_elements
        * _HEURISTIC_BYTES_PER_PARAM_STEP
    )
    return int(_HEURISTIC_FIXED_BASE_BYTES + parallel_axis_size * per_run)


def _peak_bytes_for(sweep, heuristic_only: bool = False) -> tuple[int, str]:
    """AOT-lower + compile the outer-loop region for ``sweep`` and return
    ``(peak_bytes, method)`` where ``method`` is ``"compiled"`` (XLA's exact
    argument + output + temp - alias) or ``"heuristic"`` (linear param-count
    model, used when ``heuristic_only`` is set or compilation hits a GPU
    out-of-memory condition)."""
    cfg = Config(wandb_conf=WandbConfig(mode="disabled"), sweep=sweep)
    # Two config systems coexist: dp.py reads SingletonConfig, the dataloaders
    # read the conf.scope RunContext. Both must be active during lowering.
    with SingletonConfig.override(cfg), using(RunContext(cfg)):
        # Mirror main.py's setup of the compiled region and its arguments.
        x_shape, *_ = get_dataset_shapes()
        gdp_params = get_privacy_params(x_shape[0])
        # Spatial dims (…, H, W) drive the conv activation term in the heuristic.
        # Resolve first so an AutoNetworkConfig that becomes a CNN is counted —
        # _param_count below runs on the resolved network, so conv_fm must too,
        # else the conv term silently drops to 0 and under-provisions.
        resolved_network = resolve_network_config(sweep.env.network, sweep.dataset)
        conv_fm = _conv_feature_map_elements(resolved_network, x_shape[-2], x_shape[-1])

        es_conf = sweep.schedule_optimizer.es
        if es_conf.enabled:
            parallel_axis_size = int(es_conf.population_size.sample()) // 2
        else:
            parallel_axis_size = sweep.schedule_optimizer.batch_size

        env_params = DPTrainingParams.create_direct_from_config()

        if heuristic_only:
            P = _param_count(env_params.network)
            peak = _heuristic_bytes(
                parallel_axis_size,
                sweep.env.num_training_steps,
                sweep.env.batch_size,
                P,
                conv_fm,
            )
            return peak, "heuristic"

        schedule = make_schedule(sweep.schedule_optimizer.schedule, gdp_params).project()
        if getattr(schedule, "use_fista", False):
            schedule = schedule.fista_extrapolate()

        key = jr.PRNGKey(0)
        key, mb_key, init_key, noise_key = jr.split(key, 4)
        noise_keys = jr.split(noise_key, parallel_axis_size)

        try:
            get_training_loss = make_training_loss_fn(env_params)
            es_state = make_initial_es_state()
            compiled = get_training_loss.lower(
                schedule, mb_key, init_key, noise_keys, es_state
            ).compile()
            # equinox wraps the executable; the JAX Compiled is under `.compiled`.
            stats = compiled.compiled.memory_analysis()
        except Exception as exc:
            if not _is_oom(exc):
                raise  # genuine error (config/shape bug) — don't mask it
            # The prediction machine can't even compile this config (e.g. a
            # smaller GPU than the intended target, or autotuning scratch
            # OOM). Fall back to the param-count heuristic so a memory request
            # is still produced rather than aborting the whole sweep.
            P = _param_count(env_params.network)
            peak = _heuristic_bytes(
                parallel_axis_size,
                sweep.env.num_training_steps,
                sweep.env.batch_size,
                P,
                conv_fm,
            )
            return peak, "heuristic"

    return (
        int(
            stats.argument_size_in_bytes
            + stats.output_size_in_bytes
            + stats.temp_size_in_bytes
            - stats.alias_size_in_bytes
        ),
        "compiled",
    )


def main() -> None:
    conf = tyro.cli(PredictConfig)
    out_path = conf.out or conf.sweep_file

    rows = read_sweep_file(conf.sweep_file)
    run_ids = [row["run_id"] for row in rows]
    print(f"{len(run_ids)} run IDs from {conf.sweep_file}")

    wandb_conf = WandbConfig(entity=conf.entity, project=conf.project)

    # run_id -> reconstructed sweep. The base sweep only supplies field *types*;
    # _reconstruct_from_dict overwrites values, including the Union network
    # variant selected via its stored _type.
    sweeps: dict[str, object] = {}
    for rid in run_ids:
        run_conf = get_wandb_run_conf(wandb_conf, rid)
        base = _make_sweep_config("mnist", 1.0, 1500, MLPConfig(), 0, _placeholder_opt())
        sweeps[rid] = _reconstruct_from_dict(base, run_conf)

    # Dedup by memory signature; compile once per unique signature.
    sig_to_bytes: dict[tuple, int] = {}
    n_heuristic = 0
    for _rid, sweep in sweeps.items():
        sig = _signature(sweep)
        if sig not in sig_to_bytes:
            peak, method = _peak_bytes_for(sweep, heuristic_only=conf.heuristic_only)
            sig_to_bytes[sig] = peak
            n_heuristic += method == "heuristic"
            label = (
                f"{sweep.dataset} / {sweep.env.network.__class__.__name__} / "
                f"T={sweep.env.num_training_steps}"
            )
            if method == "compiled":
                tag = "compiled"
            elif conf.heuristic_only:
                tag = "heuristic (--heuristic-only)"
            else:
                tag = "HEURISTIC (compile OOM)"
            print(f"  [{tag:>27}] {label:<48} peak={peak / 2**30:6.2f} GiB")
            # Release any device buffers autotuning held so they don't
            # accumulate across the per-signature compilations.
            jax.clear_caches()

    if n_heuristic and not conf.heuristic_only:
        print(
            f"\n  NOTE: {n_heuristic} signature(s) fell back to the param-count "
            f"heuristic because compilation OOM'd on this machine. These are "
            f"conservative over-estimates; raise --safety_factor for CNN-heavy "
            f"runs whose activation memory the heuristic does not model."
        )
    elif conf.heuristic_only:
        print(
            "\n  NOTE: --heuristic-only — all estimates use the param-count model "
            "(conservative; CNN activation memory not modelled)."
        )

    # Rewrite the mem_per_gpu column in place, preserving every other column and
    # the row order (read_sweep_file returned rows in file order).
    rid_to_mem: dict[str, str] = {}
    for rid, sweep in sweeps.items():
        sig = _signature(sweep)
        peak = sig_to_bytes[sig] * conf.safety_factor
        rid_to_mem[rid] = f"{_next_pow2_gib(peak, conf.min_gib, conf.max_gib)}G"
    for row in rows:
        row["mem_per_gpu"] = rid_to_mem[row["run_id"]]

    write_sweep_file(out_path, rows)

    print(f"\nUpdated mem_per_gpu for {len(rows)} runs in {out_path}")
    print("\nSubmit with (reads run_id + mem_per_gpu from the header):")
    print(
        f"  parallel --colsep '\\t' --header : -q uv run cc/slurm/run-starter.py"
        f" --run_id={{run_id}} --mem_per_gpu={{mem_per_gpu}}"
        f" --jobname='\"<name>\"' :::: {out_path}"
    )


def _placeholder_opt():
    """A minimal optimizer config for the reconstruction template (overwritten)."""
    from conf.config_util import dist_config_helper
    from conf.optimizer_config import SGDConfig

    return SGDConfig(
        learning_rate=dist_config_helper(value=1.0, distribution="constant"),
        momentum=dist_config_helper(value=0.0, distribution="constant"),
    )


if __name__ == "__main__":
    main()
