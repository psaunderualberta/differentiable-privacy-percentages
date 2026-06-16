"""Synthesis identity: the single source of truth for how a symbolic-regression
synthesis is keyed, hashed, and turned into a directory name.

A *synthesis* is keyed by ``(synthesis identity, target)``. The **identity** is every
problem-defining input — the sweep, the row filters, the search space — but NOT the
orchestration knobs (scratch/out dirs, proc count, iteration budget, timeouts, chain
controls) and NOT the target (which is a path segment *below* the identity). Two
invocations sharing an identity fit the same problem and may safely ``warm_start`` from
each other's PySR checkpoint; differing identities must never share a run directory.

This module is **stdlib-only on purpose**: it is imported both by
``symbolic_regression.py`` (which pulls in ``pysr``/Julia) and by the login-node launcher
``cc/slurm/sr-run-starter.py`` (which must NOT). Keep it free of heavy imports.

See docs/adr/0005-synthesis-identity-scratch-paths.md.
"""

import hashlib
import json
import os

# --- The canonical identity field set -------------------------------------------------
#
# CORRECTNESS-CRITICAL. A field that changes the *fit* but is missing here will not change
# the slug, so two different problems would share a warm_start directory and corrupt each
# other's PySR state. When you add a new filter/search field to PySRConfig (e.g. a
# T-filter or a single-architecture filter), ADD ITS NAME HERE TOO.
#
# Conceptually this is "every PySRConfig field except the orchestration denylist"
# (scratch_dir, out_dir, procs, niterations, timeout_in_seconds, pad_seconds,
# max_chain_jobs, chain_depth, mirror_sync_secs) and except `targets`. It is enumerated
# rather than reflected so the launcher can compute a matching slug without importing
# PySRConfig (which would pull in pysr). See ADR 0005.
IDENTITY_FIELDS: tuple[str, ...] = (
    "cache_dir",  # hashed by BASENAME only, for host-independence
    "datasets",
    "arch_labels",
    "optimizers",
    "run_ids",
    "datapoint_frequency",
    "keep_features",
    "include_nonfinite_schedules",
    "include_diverged_training",
    "maxsize",
)

# Tuple-valued identity fields are order-insensitive: sorted before hashing/emitting.
_TUPLE_FIELDS: frozenset[str] = frozenset(
    {"datasets", "arch_labels", "optimizers", "run_ids", "keep_features"}
)

# Identity fields rendered into the human-readable slug prefix, in order. Cosmetic only —
# the hash, not this list, guarantees uniqueness, so extending it is low-stakes.
_SLUG_FIELDS: tuple[str, ...] = ("datasets",)

# Defaults for the flags emitted by identity_flags(). `cache_dir` is excluded: the
# launcher/resubmit always pass it explicitly via --cache_dir. Must match PySRConfig (and
# the launcher's SRSlurmConfig) defaults exactly.
IDENTITY_FLAG_DEFAULTS: dict[str, object] = {
    "datasets": (),
    "arch_labels": (),
    "optimizers": (),
    "run_ids": (),
    "datapoint_frequency": 100,
    "keep_features": (),
    "include_nonfinite_schedules": False,
    "include_diverged_training": False,
    "maxsize": 25,
}


def canonical_identity(mapping: dict) -> dict:
    """Project a config mapping (e.g. ``dataclasses.asdict(conf)``) onto the canonical,
    hashable identity: only IDENTITY_FIELDS, cache_dir reduced to its basename, tuples
    sorted and coerced to lists for stable JSON."""
    ident: dict = {}
    for field in IDENTITY_FIELDS:
        if field not in mapping:
            continue
        value = mapping[field]
        if field == "cache_dir":
            value = os.path.basename(os.path.normpath(str(value)))
        elif field in _TUPLE_FIELDS:
            value = sorted(value)
        ident[field] = value
    return ident


def identity_hash(identity: dict) -> str:
    """8-char SHA-1 of the canonical identity (sort-keyed, compact JSON)."""
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]


def derive_slug(identity: dict) -> str:
    """Directory-name rendering of an identity: ``<selector-prefix>-<hash8>``, or the bare
    hash when no shortlisted selector is set (e.g. the no-filter default)."""
    prefix_parts: list[str] = []
    for field in _SLUG_FIELDS:
        values = identity.get(field) or []
        if values:
            prefix_parts.append("+".join(str(v) for v in values))
    digest = identity_hash(identity)
    if prefix_parts:
        return f"{'-'.join(prefix_parts)}-{digest}"
    return digest


def slug_for(mapping: dict) -> str:
    """Convenience: canonical_identity → derive_slug in one call."""
    return derive_slug(canonical_identity(mapping))


def identity_flags(mapping: dict) -> list[str]:
    """CLI flags that reproduce the non-default identity fields, for forwarding to the
    launcher / a chain resubmit. Emits only fields that differ from their default, and
    never `cache_dir` (passed separately as --cache_dir) or `targets`."""
    identity = canonical_identity(mapping)
    flags: list[str] = []
    for field, default in IDENTITY_FLAG_DEFAULTS.items():
        value = identity.get(field, default)
        if field in _TUPLE_FIELDS:
            items = list(value)
            if items:
                flags.append(f"--{field}")
                flags.extend(str(x) for x in items)
        elif isinstance(default, bool):
            if value and not default:
                flags.append(f"--{field}")
        elif value != default:
            flags.extend([f"--{field}", str(value)])
    return flags
