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

import json
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
    own image-size / class-count constants rather than a hand-copied literal.
    """
    from util.dataloaders import _EYEPACS_IMG_SIZE

    expected_eyepacs = ((3, _EYEPACS_IMG_SIZE, _EYEPACS_IMG_SIZE), 5)
    got = DATASET_SHAPES["eyepacs"]
    if got != expected_eyepacs:
        raise AssertionError(
            f"DATASET_SHAPES['eyepacs']={got} drifted from the dataloader cache "
            f"layout {expected_eyepacs}"
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

_BASELINE_SCHEDULES: tuple[str, ...] = (
    "Constant σ/clip",
    "Clip to Median Gradient Norm",
    "Dynamic-DPSGD",
)


# ---------------------------------------------------------------------------
# Param-count helpers (mirror symbolic_regression.py)
# ---------------------------------------------------------------------------


def _mlp_param_count(din: int, hidden_sizes: list[int], nclasses: int) -> int:
    sizes = [din, *hidden_sizes, nclasses]
    total = 0
    for i in range(len(sizes) - 1):
        a, b = sizes[i], sizes[i + 1]
        total += a * b + b
        if i < len(sizes) - 2:
            total += 2 * b
    return total


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


# ---------------------------------------------------------------------------
# Config interpretation
# ---------------------------------------------------------------------------


def resolve_optimizer(env_dict: dict) -> str:
    """Map ``env.optimizer`` to {"sgd", "adam", "adamw"}.

    Handles both legacy literal-string runs and current OptimizerConfig dicts.
    """
    opt = env_dict.get("optimizer")
    if isinstance(opt, str):
        name = opt.lower()
        if name not in {"sgd", "adam", "adamw"}:
            raise ValueError(f"unknown optimizer string: {opt!r}")
        return name
    if isinstance(opt, dict):
        t = opt.get("_type")
        if t in _OPTIMIZER_TYPE_TO_NAME:
            return _OPTIMIZER_TYPE_TO_NAME[t]
        raise ValueError(f"unknown OptimizerConfig _type: {t!r}")
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


def build_run_manifest(run: Any) -> dict:
    """Serialize the non-artifact state of a run into a JSON-able manifest.

    Captures the *complete* config, summary, run metadata, and the full history
    across every logged key — unlike ``_history``/``_fetch_one_run``, which keep
    only the handful of fields the plots need. This is the config/scalar half of
    a lossless archive; artifact files are captured separately.
    """
    return {
        "config": dict(run.config),
        "summary": dict(run.summary),
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
        "history": [dict(r) for r in run.scan_history(keys=None)],
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

    def __init__(self, name: str, directory: Path):
        self.name = name
        self._directory = Path(directory)

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
            out.append(LocalArtifact(name=entry["name"], directory=directory))
        return out


def _safe_dir_name(name: str) -> str:
    """Filesystem-safe subdir name for an artifact (``sigmas:v1`` → ``sigmas-v1``)."""
    return name.replace(":", "-").replace("/", "-")


# ---------------------------------------------------------------------------
# Archive writer / reader
# ---------------------------------------------------------------------------

_MANIFESTS_SUBDIR = "manifests"
_ARTIFACTS_SUBDIR = "artifacts"


def _dump_run_to_dir(run: Any, api: Any, entity: str, project: str, run_dir: Path) -> dict:
    """Download a run's manifest + every artifact it touches under ``run_dir``.

    Returns the manifest, augmented with an ``artifacts`` index recording each
    downloaded artifact's name, on-disk subdir, and kind (``logged`` for the
    sigmas/clips tables, ``referenced`` for the baseline pulled by run id).
    """
    manifest = build_run_manifest(run)
    artifacts_root = run_dir / _ARTIFACTS_SUBDIR
    index: list[dict] = []

    for art in run.logged_artifacts():
        sub = _safe_dir_name(art.name)
        art.download(root=str(artifacts_root / sub))
        index.append({"name": art.name, "dir": sub, "kind": "logged"})

    baseline_name = f"baseline-{run.id}:latest"
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
    with zipfile.ZipFile(zip_path, "r") as zf:
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
    name = f"baseline-{run_id}:latest"
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


def _final_schedule_arrays(run: Any) -> tuple[list[float], list[float]]:
    """Pull the final-outer-step row from the sigmas/clips W&B tables."""
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
    for tn in targets:
        if tn not in tables:
            raise RuntimeError(f"missing '{tn}' artifact")

    def _final_row(df: pd.DataFrame) -> list[float]:
        cols = [c for c in df.columns if c != "step"]
        return [float(v) for v in df[cols].iloc[-1].tolist()]

    return _final_row(tables["sigmas"]), _final_row(tables["clips"])


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
    bdf = _baseline_means(api, entity, project, run.id)
    means = bdf.groupby("type")[["accuracy", "loss"]].mean()
    counts = bdf.groupby("type").size()

    scalars: list[dict] = []
    scalars.append(
        {
            **common,
            "schedule": "Learned Schedule",
            "mean_acc": learned_acc,
            "mean_loss": learned_loss,
            "n_reps": 1,
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

    sigmas, clips = _final_schedule_arrays(run)
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
