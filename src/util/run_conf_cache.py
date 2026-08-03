"""A persistent on-disk copy of run configs fetched from the W&B server.

Why this exists
---------------
Resuming a run (``restart_run_id`` / ``checkpoint_run_id``) requires the source
run's config, which ``conf/singleton_conf.get_wandb_run_conf`` fetches from the
W&B *server* before ``wandb.init`` — see `util/wandb_retry.py` for how that cost
30 FirSweep jobs.  Retrying covers a blip; it does not cover an outage longer
than the retry window.

But the config is immutable once the run has been created, and every job in a
chain after the first has already fetched it successfully at least once.  So
caching it makes continuations survivable regardless of how long W&B is down:
the network fetch stays authoritative, and the cache is consulted only after
retries are exhausted.

The cache lives on *persistent* storage (``src/cache/run-configs`` by default,
not ``SLURM_TMPDIR``) so it outlives the job that wrote it — the same mistake
that made the lost offline run dirs unrecoverable.  Every operation is
best-effort: a cache problem must never take down a run that fetched fine.
"""

import contextlib
import json
import os
import pathlib
import re

# src/util/run_conf_cache.py → src/util → src
_SRC_ROOT = pathlib.Path(__file__).resolve().parent.parent

CACHE_DIR_ENV_VAR = "WANDB_RUN_CONF_CACHE_DIR"

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


def cache_dir() -> pathlib.Path:
    """Where cached run configs live; overridable for tests and odd layouts."""
    override = os.environ.get(CACHE_DIR_ENV_VAR, "")
    if override:
        return pathlib.Path(override)
    return _SRC_ROOT / "cache" / "run-configs"


def cache_path(entity: str, project: str, run_id: str) -> pathlib.Path:
    """The cache file for one run.

    Keyed by all three components: the same run ID can exist under a different
    entity or project (e.g. the ``{project}-branched`` runs main.py creates).
    """
    slug = "__".join(_UNSAFE.sub("_", part) for part in (entity, project, run_id))
    return cache_dir() / f"{slug}.json"


def read_run_conf(entity: str, project: str, run_id: str) -> dict | None:
    """Return the cached config, or ``None`` if absent or unreadable.

    A corrupt entry (e.g. a write truncated by a job kill) reads as a miss
    rather than an error: the caller is already in a failure path.
    """
    path = cache_path(entity, project, run_id)
    try:
        with path.open() as f:
            conf = json.load(f)
    except (OSError, ValueError):
        return None
    return conf if isinstance(conf, dict) else None


def write_run_conf(entity: str, project: str, run_id: str, conf: dict) -> None:
    """Cache ``conf``, atomically. Never raises — caching is best-effort.

    Written to a temporary file and renamed so a concurrent reader (another job
    in the chain, or a duplicate chain) never observes a partial file.
    """
    path = cache_path(entity, project, run_id)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w") as f:
            json.dump(conf, f)
        tmp.replace(path)
    except (OSError, TypeError, ValueError) as e:
        print(f"Warning: could not cache the config for run '{run_id}': {e!r}")
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
