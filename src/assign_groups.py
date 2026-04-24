#!/usr/bin/env python3
"""
assign_groups.py — Retroactively set run.group on existing config-seed runs
created by create_experiments.py, so that W&B's grouping UI aggregates them
by (dataset, eps, axis).

Usage (from src/):
    uv run assign_groups.py --project WarmupParallelSweepMomentum --entity psaunder --dry-run
    uv run assign_groups.py --project WarmupParallelSweepMomentum --entity psaunder
    uv run assign_groups.py --project WarmupParallelSweepMomentum --entity psaunder --force
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import tqdm
import tyro

import wandb
from create_experiments import _group_label

AXIS_TAGS = ("T-sweep", "arch-sweep")


@dataclass
class Args:
    project: str
    entity: str
    dry_run: bool = False
    """Print planned changes without writing."""
    force: bool = False
    """Overwrite group when already set to a different value."""


def _axis_from_tags(tags: list[str]) -> str | None:
    matches = [t for t in tags if t in AXIS_TAGS]
    if len(matches) != 1:
        return None
    return matches[0]


def main() -> None:
    args = tyro.cli(Args)
    api = wandb.Api()
    path = f"{args.entity}/{args.project}"
    runs = api.runs(path, filters={"tags": {"$in": ["config-seed"]}})

    planned: list[tuple[str, str, str, str | None, str]] = []  # (id, name, state, old, new)
    skipped_no_axis: list[str] = []
    skipped_missing_config: list[str] = []

    for run in runs:
        axis = _axis_from_tags(list(run.tags or []))
        if axis is None:
            skipped_no_axis.append(run.id)
            continue

        try:
            ds = run.config["dataset"]
            eps = run.config["env"]["eps"]
        except (KeyError, TypeError):
            skipped_missing_config.append(run.id)
            continue

        new_group = _group_label(ds, eps, axis)
        old_group = run.group or None
        planned.append((run.id, run.name, run.state, old_group, new_group))

    to_set = [p for p in planned if p[3] is None]
    already_ok = [p for p in planned if p[3] == p[4]]
    conflicts = [p for p in planned if p[3] is not None and p[3] != p[4]]

    print(f"Found {len(planned)} matching runs in {path}")
    print(f"  to set:       {len(to_set)}")
    print(f"  already ok:   {len(already_ok)}")
    print(f"  conflicts:    {len(conflicts)}")
    if skipped_no_axis:
        print(f"  skipped (no/ambiguous axis tag): {len(skipped_no_axis)}")
    if skipped_missing_config:
        print(f"  skipped (missing config fields): {len(skipped_missing_config)}")

    print("\nPlanned changes:")
    for rid, name, state, old, new in planned:
        marker = "=" if old == new else ("+" if old is None else "!")
        print(f"  [{marker}] {rid}  state={state:<8}  {old!r:<40} → {new!r}   {name}")

    if conflicts and not args.force:
        print(
            f"\nWARNING: {len(conflicts)} run(s) already have a different group set."
            " Re-run with --force to overwrite them."
        )

    if args.dry_run:
        print("\n--dry-run: no writes performed.")
    else:
        written = 0
        for rid, _name, _state, old, new in tqdm.tqdm(planned):
            if old == new:
                continue
            if old is not None and not args.force:
                continue
            run = api.run(f"{path}/{rid}")
            run.group = new
            run.update()
            written += 1
        print(f"\nUpdated {written} run(s).")

    target_counts = Counter(new for _rid, _name, _state, _old, new in planned)
    print("\nRuns per target group:")
    for group, count in sorted(target_counts.items()):
        print(f"  {count:>4}  {group}")


if __name__ == "__main__":
    main()
