"""Remove the transfer cells ADR 0021 invalidated, so the relaunch actually re-runs them.

The arm fix does **not** change a curve cell's filename — it is keyed on the source's
W&B run id, which already determines the arm — which is exactly what lets the 766
matched ``sgd-m0.9`` cells survive the relaunch untouched. The same property makes the
mismatched ``sgd-m0.0`` cells invisible to the fix: their names are unchanged too, so
``transfer_launch.drop_finished`` reads them as finished and the relaunch would skip
the very cells it exists to redo.

So the invalidation is explicit and separate from the launcher: delete the mismatched
cells, then launch. Each cell records its own ``source_arm``, so which half a file
belongs to is read out of the file rather than re-derived from schedules.parquet.

    cd src && uv run transfer_invalidate.py --cell_root <cache>/transfer --dry_run
"""

import dataclasses
from pathlib import Path

import pandas as pd
import tyro

# Producers whose cells this touches. Reference cells are deliberately excluded: a
# reference's arm is its *target* momentum, the existing ones all ran at m=0.9 (the bug
# pinned every target there), and the arm fix renames them, so the skip filter re-runs
# them on filename alone. Deleting them would only make the same work happen twice.
_INVALIDATED_PRODUCERS = ("curve", "equation")


def invalid_cells(cell_root: Path | str, keep_arm: str) -> list[Path]:
    """The cell parquets under ``cell_root`` that ADR 0021 invalidates, sorted.

    A cell survives iff it is a curve/equation cell whose recorded ``source_arm`` is
    ``keep_arm`` — the arm whose target was matched by accident of the bug. Everything
    else in those producers goes, including cells written from a parquet predating
    ADR 0011 that carry no arm at all, and cells older than the ``source_arm`` column
    itself: nothing records which momentum they were learned under, so they cannot be
    salvaged. A cell is read for its arm alone, so a missing column is answered rather
    than raised — one prehistoric file must not strand the whole invalidation.
    """
    root = Path(cell_root)
    stale = []
    for producer in _INVALIDATED_PRODUCERS:
        for path in sorted((root / producer).glob("*.parquet")):
            if "source_arm" not in pd.read_parquet(path).columns:
                stale.append(path)
                continue
            arms = set(pd.read_parquet(path, columns=["source_arm"])["source_arm"].astype(str))
            if arms != {keep_arm}:
                stale.append(path)
    return stale


@dataclasses.dataclass
class InvalidateConfig:
    """Delete the ADR 0021-invalidated transfer cells from a cache."""

    cell_root: str
    """Directory holding the per-producer cell dirs (``curve/``, ``reference/``, ...)."""
    keep_arm: str = "sgd-m0.9"
    """The arm whose cells were matched by construction and stay valid."""
    dry_run: bool = True
    """Report what would be deleted and delete nothing. Pass --no-dry_run to act."""


def main(conf: InvalidateConfig) -> None:
    stale = invalid_cells(conf.cell_root, conf.keep_arm)
    total = sum(
        1 for p in _INVALIDATED_PRODUCERS for _ in (Path(conf.cell_root) / p).glob("*.parquet")
    )
    print(f"{len(stale)} of {total} curve/equation cells are invalid under ADR 0021")
    if conf.dry_run:
        for path in stale[:5]:
            print(f"  would delete {path.name}")
        if len(stale) > 5:
            print(f"  ... and {len(stale) - 5} more")
        print("dry run: nothing deleted (pass --no-dry_run to delete)")
        return
    for path in stale:
        path.unlink()
    print(f"deleted {len(stale)} cells; {total - len(stale)} matched cells left in place")


if __name__ == "__main__":
    main(tyro.cli(InvalidateConfig))
