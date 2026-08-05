#!/usr/bin/env python3
"""compile_results_fetch.py — Pull per-run scalars and final-schedule arrays
from a W&B project produced by ``create_experiments.py``, and write four
artefacts under a cache dir:

    scalars.parquet    one row per (run_id, schedule)
    schedules.parquet  one row per (run_id, inner_step, var ∈ {sigma, clip})
    histories.parquet  one row per (run_id, outer_step) — Learned only
    missing.csv        runs that were skipped, with reason

Run once per project; ``compile_results_plot.py`` and ``symbolic_regression.py``
both read these caches.

Usage (from src/):
    uv run compile_results_fetch.py --project schedule-T-arch --entity <entity>
"""

from __future__ import annotations

import contextlib
import gc
import itertools
import json
import resource
import shutil
import tempfile
import threading
import zipfile
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tqdm
import tyro

import wandb

CACHE_ROOT = Path(__file__).parent / "cache" / "results"
ARTIFACT_ROOT = Path(__file__).parent / "cache" / "artifacts"


# ---------------------------------------------------------------------------
# Concurrent-download safety
# ---------------------------------------------------------------------------

# Runtime check for the assumption that no two artifact downloads ever target
# the same scratch directory at once. We expect every artifact to resolve to a
# unique dir (names embed the run id), so under threading this guard should
# never fire — but if it does, two threads were interleaving writes into one
# directory and the downloaded files may be corrupt. Single-threaded callers
# claim and release serially, so the guard is a no-op for them.
_dir_claims: dict[str, str] = {}
_dir_claims_lock = threading.Lock()

# wandb.Api wraps a requests session, which is not guaranteed thread-safe. Give
# each worker thread its own Api (and thus its own run objects) so no client
# state is shared across threads.
_thread_local = threading.local()


def _get_api() -> wandb.Api:
    api = getattr(_thread_local, "api", None)
    if api is None:
        api = wandb.Api()
        _thread_local.api = api
    return api


# ---------------------------------------------------------------------------
# File-descriptor pressure
# ---------------------------------------------------------------------------


def raise_descriptor_limit() -> None:
    """Lift the soft descriptor limit to the hard one, where the OS allows it.

    Downloading artifacts is descriptor-hungry: wandb fans each artifact out over
    an internal 64-thread pool, so peak usage scales with ``num_workers`` — around
    2800 descriptors at the default 8 workers, measured. That is comfortably under
    a 1048576 limit and comfortably over the 1024 that many shells, containers and
    schedulers still default to. Raising the soft limit costs nothing and removes
    the whole class of failure on the hosts that need it.

    Best-effort: a refused raise is not worth aborting a multi-hour fetch over.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= hard:
        return
    with contextlib.suppress(ValueError, OSError):
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))


@contextmanager
def _claim_download_dir(path: str, owner: str) -> Generator[None]:
    with _dir_claims_lock:
        holder = _dir_claims.get(path)
        if holder is not None and holder != owner:
            raise RuntimeError(
                f"scratch-dir collision: {path!r} is being downloaded by {holder!r} "
                f"while {owner!r} tried to write into it concurrently"
            )
        _dir_claims[path] = owner
    try:
        yield
    finally:
        with _dir_claims_lock:
            if _dir_claims.get(path) == owner:
                del _dir_claims[path]


# Mirrors symbolic_regression.DATASET_SHAPES / _AUTO_CNN / _AUTO_MLP. Kept local
# so this script is independent of the training code.
DATASET_SHAPES: dict[str, tuple[tuple[int, ...], int]] = {
    "mnist": ((1, 28, 28), 10),
    "fashion-mnist": ((1, 28, 28), 10),
    "cifar-10": ((3, 32, 32), 10),
    "california": ((8,), 2),
    "eyepacs": ((3, 256, 256), 5),
    # Surrogate transfer targets (targets only; see ADR 0007).
    "chexpert": ((1, 64, 64), 2),
    "imagenet": ((3, 32, 32), 100),
}


def assert_shapes_consistent() -> None:
    """Guard the fetch-side DATASET_SHAPES mirror against drifting from the dataloader.

    The eyepacs entry silently disagreed with the cache once (224x224/2-class here
    vs the dataloader's 256x256/5-class), so its shape is tied to the dataloader's
    own image-size / class-count constants rather than a hand-copied literal. The
    two surrogate targets are pinned the same way.
    """
    from util.dataloaders import (
        _CHEXPERT_IMG_SIZE,
        _EYEPACS_IMG_SIZE,
        _IMAGENET32_IMG_SIZE,
        _IMAGENET100_NAMES,
    )

    expected = {
        "eyepacs": ((3, _EYEPACS_IMG_SIZE, _EYEPACS_IMG_SIZE), 5),
        "chexpert": ((1, _CHEXPERT_IMG_SIZE, _CHEXPERT_IMG_SIZE), 2),
        "imagenet": (
            (3, _IMAGENET32_IMG_SIZE, _IMAGENET32_IMG_SIZE),
            len(_IMAGENET100_NAMES),
        ),
    }
    for dataset, want in expected.items():
        got = DATASET_SHAPES[dataset]
        if got != want:
            raise AssertionError(
                f"DATASET_SHAPES[{dataset!r}]={got} drifted from the dataloader cache layout {want}"
            )


_AUTO_CNN: dict[str, dict] = {
    "mnist": {
        "channels": [16, 32],
        "kernel_sizes": [8, 4],
        "paddings": [2, 0],
        "strides": [2, 2],
        "pool_kernel_size": 2,
        "mlp": {"hidden_sizes": [32]},
    },
    "fashion-mnist": {
        "channels": [16, 32],
        "kernel_sizes": [8, 4],
        "paddings": [2, 0],
        "strides": [2, 2],
        "pool_kernel_size": 2,
        "mlp": {"hidden_sizes": [32]},
    },
    "cifar-10": {
        "channels": [32, 64],
        "kernel_sizes": [3, 3],
        "paddings": [1, 1],
        "strides": [1, 1],
        "pool_kernel_size": 2,
        "mlp": {"hidden_sizes": [256]},
    },
    "eyepacs": {
        "channels": [16, 32],
        "kernel_sizes": [8, 4],
        "paddings": [2, 0],
        "strides": [2, 2],
        "pool_kernel_size": 2,
        "mlp": {"hidden_sizes": [32]},
    },
    "chexpert": {
        "channels": [16, 32],
        "kernel_sizes": [8, 4],
        "paddings": [2, 0],
        "strides": [2, 2],
        "pool_kernel_size": 2,
        "mlp": {"hidden_sizes": [32]},
    },
    "imagenet": {
        "channels": [32, 64],
        "kernel_sizes": [3, 3],
        "paddings": [1, 1],
        "strides": [1, 1],
        "pool_kernel_size": 2,
        "mlp": {"hidden_sizes": [256]},
    },
}
_AUTO_MLP: dict[str, dict] = {"california": {"hidden_sizes": [64, 32]}}

# Prefix marking a W&B tag as architecture-ladder membership (e.g. "ladder:mlp-depth").
# Kept in sync with experiments.architectures.LADDER_TAG_PREFIX but declared locally
# so this fetch script stays independent of the training code.
_LADDER_TAG_PREFIX: str = "ladder:"

_OPTIMIZER_TYPE_TO_NAME: dict[str, str] = {
    "SGDConfig": "sgd",
    "AdamConfig": "adam",
    "AdamWConfig": "adamw",
}

# The learned schedule's own name in the baseline artifact's ``type`` column; it
# is not a baseline, so it is excluded from _BASELINE_SCHEDULES below.
LEARNED_SCHEDULE: str = "Learned Schedule"

_BASELINE_SCHEDULES: tuple[str, ...] = (
    "Constant σ/clip",
    "Adaptive Clip (Andrew et al.)",
    "Dynamic-DPSGD",
)


# ---------------------------------------------------------------------------
# Param-count helpers (mirror symbolic_regression.py)
# ---------------------------------------------------------------------------


def _mlp_param_count(din: int, hidden_sizes: list[int], nclasses: int) -> int:
    """Weights + biases of a Linear/tanh stack — the network MLP.from_config builds.

    An earlier version added 2 affine parameters per hidden unit, for a
    normalisation layer the MLP does not have; ``assert_mlp_param_counts_consistent``
    is what now keeps this honest.
    """
    sizes = [din, *hidden_sizes, nclasses]
    return sum(a * b + b for a, b in itertools.pairwise(sizes))


def _cnn_param_count(input_shape: tuple[int, ...], net: dict, nclasses: int) -> int:
    channels = net.get("channels", [16, 32])
    kernels = net.get("kernel_sizes", [8, 4])
    paddings = net.get("paddings", [2, 0])
    strides = net.get("strides", [2, 2])
    pool_k = net.get("pool_kernel_size", 2)
    mlp_hidden = net.get("mlp", {}).get("hidden_sizes", [32])

    total = 0
    in_ch, h, w = input_shape
    for out_ch, k, p, s in zip(channels, kernels, paddings, strides):
        total += in_ch * out_ch * k * k + out_ch
        h = (h + 2 * p - k) // s + 1
        w = (w + 2 * p - k) // s + 1
        h //= pool_k
        w //= pool_k
        in_ch = out_ch

    total += _mlp_param_count(in_ch * h * w, mlp_hidden, nclasses)
    return total


def _built_param_count(conf: Any, input_shape: tuple[int, ...], nclasses: int) -> int:
    """Parameter count of the network the training code actually builds from ``conf``.

    JAX and the network package are imported lazily, so the fetch script keeps its
    no-JAX import path for every other use; only the guards below pay for it.
    """
    import equinox as eqx
    import jax

    from networks._registry import build
    from networks.cnn.CNN import CNN  # noqa: F401 — triggers @register(CNNConfig)
    from networks.mlp.MLP import MLP  # noqa: F401 — triggers @register(MLPConfig)

    model = build(conf, (1, *input_shape), (1, nclasses))
    leaves = jax.tree.leaves(eqx.filter(model, eqx.is_array))
    return int(sum(leaf.size for leaf in leaves))


def _built_cnn_param_count(input_shape: tuple[int, ...], net: dict, nclasses: int) -> int:
    from networks.cnn.config import CNNConfig
    from networks.mlp.config import MLPConfig

    conf = CNNConfig(
        channels=tuple(net["channels"]),
        kernel_sizes=tuple(net["kernel_sizes"]),
        paddings=tuple(net["paddings"]),
        strides=tuple(net["strides"]),
        pool_kernel_size=net["pool_kernel_size"],
        mlp=MLPConfig(hidden_sizes=tuple(net["mlp"]["hidden_sizes"])),
    )
    return _built_param_count(conf, input_shape, nclasses)


def _built_mlp_param_count(din: int, hidden_sizes: list[int], nclasses: int) -> int:
    from networks.mlp.config import MLPConfig

    return _built_param_count(MLPConfig(hidden_sizes=tuple(hidden_sizes)), (din,), nclasses)


def assert_mlp_param_counts_consistent() -> None:
    """Guard ``_mlp_param_count`` against the built network (see below)."""
    for dataset in ("mnist", "cifar-10"):
        input_shape, nclasses = DATASET_SHAPES[dataset]
        din = 1
        for d in input_shape:
            din *= d
        for hidden in ([], [64], [128], [512], [128, 128]):
            modelled = _mlp_param_count(din, hidden, nclasses)
            built = _built_mlp_param_count(din, hidden, nclasses)
            if modelled != built:
                raise AssertionError(
                    f"_mlp_param_count({dataset}, hidden={hidden}) = {modelled} "
                    f"disagrees with the built network's {built} parameters"
                )


def assert_cnn_param_counts_consistent() -> None:
    """Guard ``_cnn_param_count``'s geometry model against the built network.

    The modelled count assumes a *halving* pool. When the pool was silently
    stride-1 (ADR 0010) the two disagreed by ~5x and ``arch_param_count`` was
    wrong for every CNN run, with nothing to catch it — the same failure mode
    ``assert_shapes_consistent`` guards for ``DATASET_SHAPES``.
    """
    probes: list[dict] = [
        *_AUTO_CNN.values(),
        # The same-conv block of the cnn-depth ladder: all downsampling is the pool's.
        {
            "channels": [16, 16, 16],
            "kernel_sizes": [3, 3, 3],
            "paddings": [1, 1, 1],
            "strides": [1, 1, 1],
            "pool_kernel_size": 2,
            "mlp": {"hidden_sizes": [64]},
        },
    ]
    for dataset in ("mnist", "cifar-10"):
        input_shape, nclasses = DATASET_SHAPES[dataset]
        for net in probes:
            modelled = _cnn_param_count(input_shape, net, nclasses)
            built = _built_cnn_param_count(input_shape, net, nclasses)
            if modelled != built:
                raise AssertionError(
                    f"_cnn_param_count({dataset}, channels={net['channels']}) = {modelled} "
                    f"disagrees with the built network's {built} parameters"
                )


# ---------------------------------------------------------------------------
# Config interpretation
# ---------------------------------------------------------------------------


def resolve_optimizer(env_dict: dict) -> str:
    """Name the optimizer and, for SGD, the arm it belongs to (ADR 0011).

    Returns e.g. ``"sgd-m0.9"`` / ``"sgd-m0.0"`` — the private network's inner
    momentum is an arm, and pooling the arms into one ``"sgd"`` column would make
    the difference invisible in every figure. Adam/AdamW carry no momentum and
    stay bare, as do legacy runs that predate the arm (literal-string optimizers,
    or configs with no momentum recorded), so cached sweeps still re-fetch.

    ``create_experiments._opt_tag`` reproduces this scheme independently; the
    naming is the contract between them.
    """
    opt = env_dict.get("optimizer")
    if isinstance(opt, str):
        name = opt.lower()
        if name not in {"sgd", "adam", "adamw"}:
            raise ValueError(f"unknown optimizer string: {opt!r}")
        return name
    if isinstance(opt, dict):
        t = opt.get("_type")
        if t not in _OPTIMIZER_TYPE_TO_NAME:
            raise ValueError(f"unknown OptimizerConfig _type: {t!r}")
        name = _OPTIMIZER_TYPE_TO_NAME[t]
        # Serialised either as the DistributionConfig's scalar .value or as the
        # nested dict, depending on which writer produced the run config.
        momentum = opt.get("momentum")
        if isinstance(momentum, dict):
            momentum = momentum.get("value")
        if momentum is None:
            return name
        return f"{name}-m{float(momentum)}"
    raise ValueError(f"missing or unrecognised env.optimizer: {opt!r}")


def _arch_info(env_dict: dict, dataset: str) -> tuple[str, int | None]:
    """Return ``(label, num_params)``. Mirrors create_experiments._arch_label."""
    net = env_dict.get("network", {})
    net_type = net.get("_type", "AutoNetworkConfig")

    resolved_net = net
    if net_type == "AutoNetworkConfig":
        if dataset in _AUTO_CNN:
            net_type, resolved_net = "CNNConfig", _AUTO_CNN[dataset]
        elif dataset in _AUTO_MLP:
            net_type, resolved_net = "MLPConfig", _AUTO_MLP[dataset]

    if net_type == "MLPConfig":
        hs = list(resolved_net.get("hidden_sizes", []))
        label = "mlp-" + "x".join(str(h) for h in hs)
    elif net_type == "CNNConfig":
        ch = "x".join(str(c) for c in resolved_net.get("channels", []))
        head = "x".join(str(h) for h in resolved_net.get("mlp", {}).get("hidden_sizes", []))
        label = f"cnn-{ch}-head{head}"
    else:
        label = f"unknown-{net_type}"

    n_params: int | None = None
    if dataset in DATASET_SHAPES:
        input_shape, nclasses = DATASET_SHAPES[dataset]
        din = 1
        for d in input_shape:
            din *= d
        if net_type == "MLPConfig":
            n_params = _mlp_param_count(din, list(resolved_net.get("hidden_sizes", [])), nclasses)
        elif net_type == "CNNConfig":
            n_params = _cnn_param_count(input_shape, resolved_net, nclasses)
    return label, n_params


def _seed(cfg: dict) -> int | None:
    raw = cfg.get("prng_seed")
    if isinstance(raw, dict):
        v = raw.get("value")
        return int(v) if v is not None else None
    if raw is None:
        return None
    return int(raw)


def _axis(tags: list[str]) -> str:
    """Coarse axis from run tags: "T-sweep" or "arch".

    Read directly from the tags written by create_experiments.py. "arch-sweep" is
    accepted for back-compat with pre-ladder projects.
    """
    if "T-sweep" in tags:
        return "T-sweep"
    if "arch" in tags or "arch-sweep" in tags:
        return "arch"
    return "unknown"


def _ladder_memberships(tags: list[str]) -> dict[str, bool]:
    """One ``in_<ladder>`` boolean per ``ladder:<name>`` tag on the run.

    Discovered generically from the tag prefix, so new ladders need no change here.
    Runs with no ladder tags (e.g. the T-sweep) contribute no columns.
    """
    return {
        f"in_{t.removeprefix(_LADDER_TAG_PREFIX).replace('-', '_')}": True
        for t in tags
        if t.startswith(_LADDER_TAG_PREFIX)
    }


# ---------------------------------------------------------------------------
# Full-run dump (lossless archive)
# ---------------------------------------------------------------------------


def _jsonable(value: Any) -> Any:
    """Plain-JSON view of a W&B config/summary/history value.

    ``run.summary`` returns nested keys as ``SummarySubDict``, which is neither a
    ``dict`` nor a ``Mapping``, so ``json.dumps`` rejects it outright and a single
    run with a nested summary key aborted the entire archive. numpy scalars and
    arrays arrive from the same place and are rejected for the same reason.
    Anything still unrecognised degrades to its ``repr`` rather than killing the
    archive: the manifest is a best-effort record, and losing one exotic summary
    value is far cheaper than losing the run.
    """
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    # SummarySubDict and friends: mapping-shaped, but not a Mapping.
    if hasattr(value, "keys") and hasattr(value, "__getitem__"):
        # .keys() is load-bearing, not the redundant call SIM118 takes it for:
        # these objects define __getitem__ but no __iter__, so iterating one
        # directly falls back to the sequence protocol and yields 0, 1, 2, ...
        return {str(k): _jsonable(value[k]) for k in value.keys()}  # noqa: SIM118
    return repr(value)


def build_run_manifest(run: Any) -> dict:
    """Serialize the non-artifact state of a run into a JSON-able manifest.

    Captures the *complete* config, summary, run metadata, and the full history
    across every logged key — unlike ``_history``/``_fetch_one_run``, which keep
    only the handful of fields the plots need. This is the config/scalar half of
    a lossless archive; artifact files are captured separately.
    """
    return {
        "config": _jsonable(dict(run.config)),
        "summary": _jsonable(dict(run.summary)),
        "meta": {
            "id": run.id,
            "name": run.name,
            "tags": list(run.tags or []),
            "state": getattr(run, "state", None),
            "notes": getattr(run, "notes", None),
            "group": getattr(run, "group", None),
            "job_type": getattr(run, "job_type", None),
            "created_at": getattr(run, "created_at", None),
            "url": getattr(run, "url", None),
        },
        "history": [_jsonable(dict(r)) for r in run.scan_history(keys=None)],
    }


@dataclass
class _LocalTable:
    """Minimal wandb.Table stand-in: exposes ``.columns`` and ``.data``."""

    columns: list
    data: list


class LocalArtifact:
    """A wandb.Artifact stand-in backed by a directory of downloaded files.

    Serves the read surface the fetch code uses: ``.name``, ``.get(table)`` (for
    the sigmas/clips W&B tables) and ``.download(root)`` (for the baseline pkl).
    """

    def __init__(self, name: str, directory: Path, metadata: dict | None = None):
        self.name = name
        self._directory = Path(directory)
        # Carries save_checkpoint's recorded step, which is what the checkpoint
        # fallback orders on — without it a replay cannot use a checkpoint at all.
        self.metadata = dict(metadata or {})

    def get(self, table_name: str) -> _LocalTable:
        path = self._directory / f"{table_name}.table.json"
        payload = json.loads(path.read_text())
        return _LocalTable(columns=payload["columns"], data=payload["data"])

    def download(self, root: str | None = None) -> str:
        if root is None or Path(root) == self._directory:
            return str(self._directory)
        shutil.copytree(self._directory, root, dirs_exist_ok=True)
        return str(root)


class LocalRun:
    """A wandb.Run stand-in backed by a dumped manifest (+ artifact dir).

    Exposes the same read surface the fetch code uses against a live run, so an
    archived run can be replayed through ``_fetch_one_run`` after the original is
    deleted from W&B.
    """

    def __init__(self, manifest: dict, artifact_root: Path | None):
        self._manifest = manifest
        self._artifact_root = artifact_root
        meta = manifest["meta"]
        self.id = meta["id"]
        self.name = meta["name"]
        self.tags = list(meta.get("tags") or [])
        self.state = meta.get("state")
        self.notes = meta.get("notes")
        self.group = meta.get("group")
        self.job_type = meta.get("job_type")
        self.created_at = meta.get("created_at")
        self.url = meta.get("url")
        self.config = manifest["config"]
        self.summary = manifest["summary"]

    def scan_history(self, keys: list[str] | None = None) -> Generator[dict]:
        for row in self._manifest["history"]:
            yield dict(row) if keys is None else {k: row[k] for k in keys}

    def logged_artifacts(self) -> list[LocalArtifact]:
        out: list[LocalArtifact] = []
        for entry in self._manifest.get("artifacts", []):
            if entry.get("kind") != "logged":
                continue
            directory = Path(self._artifact_root) / entry["dir"]
            out.append(
                LocalArtifact(
                    name=entry["name"],
                    directory=directory,
                    metadata=entry.get("metadata"),
                )
            )
        return out


def _safe_dir_name(name: str) -> str:
    """Filesystem-safe subdir name for an artifact (``sigmas:v1`` → ``sigmas-v1``)."""
    return name.replace(":", "-").replace("/", "-")


# ---------------------------------------------------------------------------
# Archive writer / reader
# ---------------------------------------------------------------------------

_MANIFESTS_SUBDIR = "manifests"
_ARTIFACTS_SUBDIR = "artifacts"


def _artifacts_worth_archiving(artifacts: list[Any]) -> list[Any]:
    """Drop the checkpoints no reader will ever open.

    A run logs a ``checkpoint-<run_id>`` artifact per outer step — 40 of the 55
    artifacts on a FirSweep run, and the bulk of its bytes. Only ever *one* is
    read back: ``_schedule_arrays_from_checkpoint`` takes the newest, and only as
    the fallback for a run whose sigmas/clips tables never uploaded. Archiving
    the rest cost hours of downloading and gigabytes of staging for bytes nothing
    reads.

    A checkpoint that records no ``step`` goes too. That is not a judgement call
    about size: ``_schedule_arrays_from_checkpoint`` refuses such an artifact
    outright, because without the step it cannot show the checkpoint is the run's
    last rather than wherever a dead job happened to stop.
    """
    newest = _newest_checkpoint(artifacts)
    keep = newest[0].name if newest is not None else None
    return [a for a in artifacts if "checkpoint-" not in a.name or a.name == keep]


def _dump_run_to_dir(run: Any, api: Any, entity: str, project: str, run_dir: Path) -> dict:
    """Download a run's manifest + every artifact a reader can use, under ``run_dir``.

    Returns the manifest, augmented with an ``artifacts`` index recording each
    downloaded artifact's name, on-disk subdir, kind (``logged`` for the
    sigmas/clips tables and the newest checkpoint, ``referenced`` for the
    baseline pulled by run id) and metadata.

    Superseded checkpoints are skipped — see ``_artifacts_worth_archiving``.
    """
    manifest = build_run_manifest(run)
    artifacts_root = run_dir / _ARTIFACTS_SUBDIR
    index: list[dict] = []

    for art in _artifacts_worth_archiving(list(run.logged_artifacts())):
        sub = _safe_dir_name(art.name)
        art.download(root=str(artifacts_root / sub))
        index.append(
            {
                "name": art.name,
                "dir": sub,
                "kind": "logged",
                # The checkpoint fallback selects on the recorded step, so the
                # replay needs it as much as the live fetch does.
                "metadata": _jsonable(getattr(art, "metadata", None) or {}),
            }
        )

    baseline_name = f"baseline-v2-{run.id}:latest"
    baseline = api.artifact(f"{entity}/{project}/{baseline_name}")
    sub = _safe_dir_name(baseline_name)
    baseline.download(root=str(artifacts_root / sub))
    index.append({"name": baseline_name, "dir": sub, "kind": "referenced"})

    manifest["artifacts"] = index
    return manifest


def _dump_one_run(
    entity: str, project: str, run_id: str, staging_dir: Path, api: Any = None
) -> dict:
    """Worker: fetch one run and dump its manifest + artifacts under staging.

    Builds its own per-thread ``wandb.Api`` (via ``_get_api``) so no client state
    is shared across worker threads — exactly like ``_fetch_one_run``. Tests and
    the archive round-trip inject their own api instead. Each run's files land in
    ``staging/<run_id>/`` (unique per run), so concurrent workers never collide.
    """
    if api is None:
        api = _get_api()
    run = api.run(f"{entity}/{project}/{run_id}")
    run_dir = Path(staging_dir) / run_id
    return _dump_run_to_dir(run, api, entity, project, run_dir)


def write_full_archive(
    run_ids: list[str],
    api: Any,
    entity: str,
    project: str,
    zip_path: str | Path,
    num_workers: int = 8,
) -> list[dict]:
    """Write a lossless, zipped archive of the given runs.

    Layout inside the zip:

        manifests/<run_id>.json                     full config/summary/meta/history
        artifacts/<run_id>/<artifact>/<files...>    every logged + referenced artifact

    The archive is self-describing and can be reopened with ``open_full_archive``
    to replay each run through ``_fetch_one_run`` after the originals are deleted.

    Runs are downloaded concurrently across ``num_workers`` threads (each worker
    uses its own ``wandb.Api``; pass ``api=None`` for that). Downloads stage to a
    temp dir, then the archive is zipped once serially (``zipfile`` is not
    thread-safe). A run that fails to download is collected into the returned
    skip-list rather than aborting the whole archive.
    """
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    missing: list[dict] = []
    workers = max(1, num_workers)
    with tempfile.TemporaryDirectory() as staging:
        staging_dir = Path(staging)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_dump_one_run, entity, project, run_id, staging_dir, api): run_id
                for run_id in run_ids
            }
            for fut in as_completed(futures):
                run_id = futures[fut]
                try:
                    manifest = fut.result()
                except Exception as exc:
                    missing.append({"run_id": run_id, "reason": str(exc)})
                    continue
                # Manifests are written from this single thread (after the worker
                # returns), so no concurrent writers touch the manifests dir.
                manifest_path = staging_dir / _MANIFESTS_SUBDIR / f"{run_id}.json"
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text(json.dumps(manifest))

        # Reading a run's artifacts leaves ~500 never-closed requests.Sessions
        # behind — roughly six per Artifact, most of them from the InternalApi
        # that WandbStoragePolicy.from_config builds for every artifact and again
        # for every manifest. Each owns a connection pool holding a keep-alive
        # socket, and they are reachable only through reference cycles, so
        # shutting the pool down does not release them: refcounting cannot break
        # a cycle. The descriptors therefore survive into the zip below, whose
        # very first os.scandir then dies with EMFILE — which is why this failed
        # hours in, at the amalgamation, having downloaded everything happily.
        # Collecting here, where those sessions are finally unreachable, returns
        # the descriptors before anything else asks for one.
        gc.collect()

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(staging_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(staging_dir).as_posix())
    return missing


class LocalApi:
    """A wandb.Api stand-in that replays runs from a full-config archive."""

    def __init__(self, root: Path):
        self._root = Path(root)
        # Map every referenced artifact's qualified name → its extracted dir, so
        # api.artifact("e/p/baseline-<id>:latest") resolves without a network.
        self._referenced: dict[str, Path] = {}
        for manifest_path in (self._root / _MANIFESTS_SUBDIR).glob("*.json"):
            manifest = json.loads(manifest_path.read_text())
            run_id = manifest["meta"]["id"]
            arts_root = self._root / run_id / _ARTIFACTS_SUBDIR
            for entry in manifest.get("artifacts", []):
                if entry.get("kind") == "referenced":
                    self._referenced[entry["name"]] = arts_root / entry["dir"]

    def run(self, path: str) -> LocalRun:
        run_id = path.split("/")[-1]
        manifest = json.loads((self._root / _MANIFESTS_SUBDIR / f"{run_id}.json").read_text())
        return LocalRun(manifest, artifact_root=self._root / run_id / _ARTIFACTS_SUBDIR)

    def artifact(self, path: str) -> LocalArtifact:
        name = path.split("/")[-1]
        directory = self._referenced[name]
        return LocalArtifact(name=name, directory=directory)


def open_full_archive(zip_path: str | Path) -> LocalApi:
    """Extract a full-config archive and return a LocalApi over its contents."""
    extract_dir = Path(tempfile.mkdtemp(prefix="full-config-"))
    root = extract_dir.resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            dest = (extract_dir / member.filename).resolve()
            if not dest.is_relative_to(root):
                raise RuntimeError(f"unsafe path in archive: {member.filename!r}")
        zf.extractall(extract_dir)
    return LocalApi(extract_dir)


# ---------------------------------------------------------------------------
# Per-run fetch
# ---------------------------------------------------------------------------


def _history(run: Any) -> list[dict]:
    """Return the full per-outer-step history as a list of dicts.

    Each dict has keys: outer_step, test_acc, test_loss. NaN/Inf values are kept
    so downstream plotting can show divergence as a break in the curve.
    """
    rows = list(run.scan_history(keys=["test-accuracy", "test-loss"]))
    if not rows:
        # A config seed created by create_experiments.py that no training job ever
        # wrote to is left in state "finished", so it arrives here looking exactly
        # like a run that trained and failed to log. `_runtime` separates them:
        # it is 0 (or absent) until a job actually starts writing to the run.
        if not (dict(run.summary or {}).get("_runtime") or 0):
            raise RuntimeError("run never started (config seed; no training job wrote to it)")
        raise RuntimeError("no test-accuracy / test-loss rows in run history")
    return [
        {
            "outer_step": i,
            "test_acc": float(r["test-accuracy"]),
            "test_loss": float(r["test-loss"]),
        }
        for i, r in enumerate(rows)
    ]


def _baseline_means(api: wandb.Api, entity: str, project: str, run_id: str) -> pd.DataFrame:
    name = f"baseline-v2-{run_id}:latest"
    artifact = api.artifact(f"{entity}/{project}/{name}")
    # Explicit per-artifact root: known before the download starts so the claim
    # guard covers the actual write window (see _claim_download_dir).
    root = str(ARTIFACT_ROOT / _safe_dir_name(name))
    with _claim_download_dir(root, owner=name):
        local = Path(artifact.download(root=root))
    pkls = list(local.glob("*.pkl"))
    if not pkls:
        raise RuntimeError("baseline artifact has no .pkl file")
    df = pd.read_pickle(str(pkls[0]))
    if not {"type", "accuracy", "loss"}.issubset(df.columns):
        raise RuntimeError(f"baseline df missing required cols: {df.columns}")
    return df


def _newest_checkpoint(artifacts: list[Any]) -> tuple[Any, int] | None:
    """The newest ``checkpoint-<run_id>`` artifact and the step it holds, or None.

    Ordered by the ``step`` that ``save_checkpoint`` records in the artifact
    metadata.  An artifact without that metadata is skipped rather than ordered
    by its ``:vNN`` suffix: the step is needed anyway to prove the checkpoint is
    the run's last one (see ``_schedule_arrays_from_checkpoint``), so a stand-in
    that cannot supply it cannot be recovered from either.
    """
    best: tuple[Any, int] | None = None
    for art in artifacts:
        if "checkpoint-" not in art.name:
            continue
        step = (getattr(art, "metadata", None) or {}).get("step")
        if step is None:
            continue
        if best is None or int(step) > best[1]:
            best = (art, int(step))
    return best


def _latest_checkpoint_artifact(run: Any) -> tuple[Any, int] | None:
    """``_newest_checkpoint`` over everything a run logged."""
    return _newest_checkpoint(list(run.logged_artifacts()))


def _schedule_arrays_from_checkpoint(run: Any) -> tuple[list[float], list[float]]:
    """Final σ/clip recovered from the newest checkpoint artifact.

    ``util.checkpointing.save_checkpoint`` writes ``sigmas.npy``/``clips.npy``
    beside every Orbax checkpoint, holding exactly ``get_private_noise_scales()``
    and ``get_private_clips()`` — the same two arrays the sigmas/clips W&B tables
    hold. The last checkpoint is taken after the final update+``project()``, so it
    reproduces the tables' final row (verified equal to 1e-16 on a run that has
    both).

    This is the recovery path for runs whose *table* artifacts never uploaded:
    the offline ``wandb sync`` used to run before ``run.finish()`` had flushed
    them (see ``util.wandb_init.finish_and_sync``), so the tables were lost while
    the checkpoints — logged throughout training — survived.

    Two conditions must BOTH hold, because either alone is passable by a run that
    never reached its final schedule:

    1. State ``finished``.  A run that died without teardown (SIGKILL / OOM /
       node failure) is ``crashed`` — and ``main`` fetches those too — but its
       newest checkpoint is wherever training happened to stop.
    2. The newest checkpoint is at the *last* outer step.  State is not enough:
       ``main.py`` runs its ``finally`` block (and so ``run.finish()``) on the
       job-chain shutdown path as well, so a run paused mid-chain — or whose
       continuation job never ran — also sits in W&B as ``finished`` with only
       part of its training done.  Three of the 61 FirSweep runs recovered here
       are exactly that, with their newest checkpoint at step 49 or 74 of 1000.

    Runs failing either check stay in missing.csv, where they are visible, rather
    than contributing an under-trained schedule that looks converged.
    """
    state = getattr(run, "state", None)
    if state != "finished":
        raise RuntimeError(
            f"missing 'sigmas'/'clips' artifact and run state is {state!r}, not 'finished' "
            "— its newest checkpoint is mid-training, so it is not the final schedule"
        )

    found = _latest_checkpoint_artifact(run)
    if found is None:
        raise RuntimeError(
            "missing 'sigmas'/'clips' artifact and no checkpoint (recording its step) "
            "to recover from"
        )
    art, step = found

    num_outer_steps = (getattr(run, "config", None) or {}).get("num_outer_steps")
    if num_outer_steps is None:
        raise RuntimeError(
            "missing 'sigmas'/'clips' artifact and run.config has no num_outer_steps, "
            f"so checkpoint {art.name} (step {step}) cannot be shown to be the final one"
        )
    if step + 1 != int(num_outer_steps):
        raise RuntimeError(
            f"missing 'sigmas'/'clips' artifact and the newest checkpoint is step {step} "
            f"of {num_outer_steps} outer steps — the run stopped early (job-chain hop, or "
            "a continuation that never ran), so this is not the final schedule"
        )

    root = str(ARTIFACT_ROOT / _safe_dir_name(art.name))
    with _claim_download_dir(root, owner=art.name):
        local = Path(art.download(root=root))

    arrays: list[list[float]] = []
    for tn in ("sigmas", "clips"):
        path = local / f"{tn}.npy"
        if not path.exists():
            raise RuntimeError(f"checkpoint artifact {art.name} has no {tn}.npy")
        arrays.append([float(v) for v in np.load(path)])
    return arrays[0], arrays[1]


def _final_schedule_arrays(run: Any) -> tuple[list[float], list[float], str]:
    """Final-outer-step σ/clip, plus which source they came from.

    Returns ``(sigmas, clips, source)`` where ``source`` is ``"table"`` (the
    sigmas/clips W&B tables) or ``"checkpoint"`` (recovered — see
    ``_schedule_arrays_from_checkpoint``). Both arrays always come from the same
    source: pairing a table σ with a checkpoint clip would mix two outer steps.
    """
    tables: dict[str, pd.DataFrame] = {}
    targets = ("sigmas", "clips")
    for art in run.logged_artifacts():
        for tn in targets:
            if tn in tables:
                continue
            if f"{tn}:v" in art.name:
                t = art.get(tn)
                tables[tn] = pd.DataFrame(data=t.data, columns=t.columns)
        if len(tables) == len(targets):
            break
    if len(tables) != len(targets):
        sigmas, clips = _schedule_arrays_from_checkpoint(run)
        return sigmas, clips, "checkpoint"

    def _final_row(df: pd.DataFrame) -> list[float]:
        cols = [c for c in df.columns if c != "step"]
        return [float(v) for v in df[cols].iloc[-1].tolist()]

    return _final_row(tables["sigmas"]), _final_row(tables["clips"]), "table"


def _fetch_one_run(
    entity: str, project: str, run_id: str, api: Any = None
) -> tuple[list[dict], list[dict], list[dict]]:
    # Use a per-thread Api so the run object and its client are never shared
    # across threads (see _get_api). Re-fetching by id is one cheap GraphQL call.
    # A caller may inject an api (e.g. a LocalApi replaying an archive).
    if api is None:
        api = _get_api()
    run = api.run(f"{entity}/{project}/{run_id}")
    cfg = run.config
    env = cfg.get("env", {}) or {}
    dataset = cfg.get("dataset")
    if dataset is None:
        raise RuntimeError("missing dataset in run.config")

    eps = float(env.get("eps")) if env.get("eps") is not None else None
    T = int(env.get("num_training_steps")) if env.get("num_training_steps") is not None else None
    seed = _seed(cfg)
    arch_label, n_params = _arch_info(env, dataset)
    optimizer = resolve_optimizer(env)
    tags = list(run.tags or [])
    axis = _axis(tags)

    common = {
        "run_id": run.id,
        "run_name": run.name,
        "dataset": dataset,
        "eps": eps,
        "T": T,
        "arch_label": arch_label,
        "arch_param_count": n_params,
        "seed": seed,
        "axis": axis,
        "optimizer": optimizer,
        **_ladder_memberships(tags),
    }

    history = _history(run)
    learned_acc = history[-1]["test_acc"]
    learned_loss = history[-1]["test_loss"]
    # Runs do not all reach the same step, so the step this run stopped at is part
    # of what its accuracy means (ADR 0014); the plot layer reads each cell back
    # at the minimum step common to its seeds.
    common["final_outer_step"] = history[-1]["outer_step"]
    bdf = _baseline_means(api, entity, project, run.id)
    means = bdf.groupby("type")[["accuracy", "loss"]].mean()
    counts = bdf.groupby("type").size()

    # The baseline artifact also holds a multi-rep evaluation of the learned
    # schedule, written only when the run did not stop for a chain hop. Kept
    # beside the 1-rep history read (never replacing it) so the two can be
    # checked against each other on the completed-run subset.
    learned_acc_8rep = (
        float(means.loc[LEARNED_SCHEDULE, "accuracy"]) if LEARNED_SCHEDULE in means.index else None
    )

    scalars: list[dict] = []
    scalars.append(
        {
            **common,
            "schedule": LEARNED_SCHEDULE,
            "mean_acc": learned_acc,
            "mean_loss": learned_loss,
            "n_reps": 1,
            "learned_acc_8rep": learned_acc_8rep,
        }
    )
    for sched in _BASELINE_SCHEDULES:
        if sched not in means.index:
            continue
        scalars.append(
            {
                **common,
                "schedule": sched,
                "mean_acc": float(means.loc[sched, "accuracy"]),
                "mean_loss": float(means.loc[sched, "loss"]),
                "n_reps": int(counts.loc[sched]),
            }
        )

    sigmas, clips, schedule_source = _final_schedule_arrays(run)
    if T is not None and (len(sigmas) != T or len(clips) != T):
        raise RuntimeError(
            f"final schedule length mismatch (sigmas={len(sigmas)}, clips={len(clips)}, T={T})"
        )

    schedule_rows: list[dict] = []
    for inner_step, (s_val, c_val) in enumerate(zip(sigmas, clips)):
        step_norm = inner_step / T if T else None
        schedule_rows.append(
            {
                **common,
                "inner_step": inner_step,
                "step_norm": step_norm,
                "sigma": s_val,
                "clip": c_val,
                # "table" or "checkpoint" — recovered rows are exact, but the
                # provenance is worth keeping auditable. See _final_schedule_arrays.
                "schedule_source": schedule_source,
            }
        )

    history_rows: list[dict] = [{**common, **h} for h in history]

    return scalars, schedule_rows, history_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class FetchConfig:
    project: str
    entity: str
    out_dir: str = ""
    """Cache directory. Defaults to src/cache/results/<entity>__<project>/."""
    limit: int = 0
    """If >0, fetch only this many runs (debugging)."""
    num_workers: int = 8
    """Number of parallel threads used to fetch runs. 1 = sequential."""
    full_config: bool = False
    """Also write full_config.zip — a lossless archive (full config/summary/meta/
    history + every artifact) so the W&B runs can be deleted and later replayed."""


def main(conf: FetchConfig) -> None:
    raise_descriptor_limit()
    out_dir = Path(conf.out_dir) if conf.out_dir else CACHE_ROOT / f"{conf.entity}__{conf.project}"
    out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    runs = list(
        api.runs(
            f"{conf.entity}/{conf.project}", filters={"state": {"$in": ["crashed", "finished"]}}
        )
    )
    if conf.limit > 0:
        runs = runs[: conf.limit]
    # Capture (id, name) up front: workers re-fetch by id on their own Api, and
    # the name is needed for the missing.csv fallback when a fetch fails.
    run_meta = [(run.id, run.name) for run in runs]
    print(f"{len(run_meta)} finished runs in {conf.entity}/{conf.project}")

    scalars: list[dict] = []
    schedules: list[dict] = []
    histories: list[dict] = []
    missing: list[dict] = []

    workers = max(1, conf.num_workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_one_run, conf.entity, conf.project, run_id): (run_id, name)
            for run_id, name in run_meta
        }
        for fut in tqdm.tqdm(as_completed(futures), total=len(futures), desc="runs"):
            run_id, name = futures[fut]
            try:
                s, sch, hist = fut.result()
                scalars.extend(s)
                schedules.extend(sch)
                histories.extend(hist)
            except Exception as exc:
                missing.append({"run_id": run_id, "run_name": name, "reason": str(exc)})
                tqdm.tqdm.write(f"  skipping {run_id} ({name}): {exc}")

    scalars_df = pd.DataFrame(scalars)
    schedules_df = pd.DataFrame(schedules)
    histories_df = pd.DataFrame(histories)
    missing_df = pd.DataFrame(missing)

    scalars_df.to_parquet(out_dir / "scalars.parquet", index=False)
    schedules_df.to_parquet(out_dir / "schedules.parquet", index=False)
    histories_df.to_parquet(out_dir / "histories.parquet", index=False)
    missing_df.to_csv(out_dir / "missing.csv", index=False)

    print(f"\n→ {out_dir}")
    print(f"  scalars.parquet:   {len(scalars_df)} rows")
    print(f"  schedules.parquet: {len(schedules_df)} rows")
    print(f"  histories.parquet: {len(histories_df)} rows")
    print(f"  missing.csv:       {len(missing_df)} runs")

    if conf.full_config:
        zip_path = out_dir / "full_config.zip"
        run_ids = [run_id for run_id, _ in run_meta]
        # Pass api=None: workers build their own per-thread Api (see _dump_one_run)
        # rather than sharing this main-thread client across threads.
        archive_missing = write_full_archive(
            run_ids, None, conf.entity, conf.project, zip_path, conf.num_workers
        )
        archived = len(run_ids) - len(archive_missing)
        print(f"  full_config.zip:   {archived} runs archived (lossless)")
        for m in archive_missing:
            print(f"    archive-skipped {m['run_id']}: {m['reason']}")


if __name__ == "__main__":
    main(tyro.cli(FetchConfig))
