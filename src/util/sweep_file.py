"""Self-describing sweep-file format for per-run SLURM submission flags.

A sweep file is a tab-separated table whose header row names ``run-starter.py``
flags, one row per W&B run. The first column is always ``run_id``; any further
column (e.g. ``mem_per_gpu``) is a per-run flag forwarded to ``run-starter.py``
as ``--<column>=<value>``. Files without a ``run_id`` header are legacy bare
run-ID lists (one ID per line). See
docs/adr/0004-self-describing-sweep-file-flags.md.
"""

from __future__ import annotations

import os

RUN_ID_COL = "run_id"


def write_sweep_file(path: str | os.PathLike, rows: list[dict[str, str]]) -> None:
    """Write ``rows`` as a header sweep file. Column order follows the first row."""
    header = list(rows[0].keys())
    lines = ["\t".join(header)]
    lines += ["\t".join(row[col] for col in header) for row in rows]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def read_sweep_file(path: str | os.PathLike) -> list[dict[str, str]]:
    """Read a sweep file into ``[{column: value}, ...]``."""
    with open(path) as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    # Sniff: a leading ``run_id`` field marks the header format; anything else
    # is a legacy bare run-ID list (one ID per line, no per-run flags).
    if lines[0].split("\t")[0] != RUN_ID_COL:
        return [{RUN_ID_COL: ln} for ln in lines]
    header = lines[0].split("\t")
    return [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]


def row_to_run_args(row: dict[str, str]) -> list[str]:
    """Render a row as ``run-starter.py`` flags: ``--<column>=<value>`` each."""
    return [f"--{col}={val}" for col, val in row.items()]
