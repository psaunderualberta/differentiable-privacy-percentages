import subprocess
from pathlib import Path

import tqdm
import tyro

import wandb


def _sweeps_dir() -> Path:
    return Path(__file__).parent.parent / "sweeps"


def _local_sweep_ids() -> list[str]:
    return [f.stem for f in _sweeps_dir().glob("*.txt")]


def _count_runs(sweep_file: Path) -> int:
    assert sweep_file.exists()
    with open(sweep_file) as f:
        return len(f.readlines())


def _most_recent_sweep_ids(n: int, project: str, api: wandb.Api) -> list[str]:
    local_ids = _local_sweep_ids()
    sweep_times: list[tuple[str, str]] = []
    for sweep_id in tqdm.tqdm(local_ids, desc="Getting Sweep Start Times"):
        try:
            sweep = api.sweep(f"psaunder/{project}/{sweep_id}")

            # sweep run timestamps serve as a good-enough ordering for sweep creation
            sweep_times.append((sweep_id, sweep.runs[0].created_at))
        except wandb.Error:
            pass
    sweep_times.sort(key=lambda x: x[1], reverse=True)
    return [sweep_id for sweep_id, _ in sweep_times[:n]]


def main(
    sweep_ids: list[str] | None = None,
    num_most_recent: int | None = None,
    project: str = "Testing Mu-gdp",
):
    """Submit SLURM jobs for all runs in the given W&B sweeps.

    Provide either sweep_ids (takes priority) or num_most_recent.
    """
    api = wandb.Api()

    if sweep_ids is not None and len(sweep_ids) > 0:
        ids = sweep_ids
    elif num_most_recent is not None:
        ids = _most_recent_sweep_ids(num_most_recent, project, api)
    else:
        raise ValueError("Provide either --sweep-ids or --num-most-recent")
    sweep_objects: dict[str, str] = {}
    print("The following sweeps will be submitted:")
    for sweep_id in ids:
        try:
            sweep = api.sweep(f"psaunder/{project}/{sweep_id}")
            sweep_objects[sweep_id] = sweep.name
            print(f"  {sweep_id}: {sweep.name}")
        except wandb.Error as e:
            print(f"\tWARNING - {e}. ")
            sweep_file = _sweeps_dir() / f"{sweep_id}.txt"
            if not sweep_file.exists():
                print(f"\t\tERROR: CANNOT FIND RUNS IN {sweep_file}. OMITTING")
            else:
                sweep_objects[sweep_id] = sweep_id

    print("\nExpecting to start runs within the following sweeps:")
    total_runs = 0
    for sweep_id in sweep_objects:
        num_runs = _count_runs(_sweeps_dir() / f"{sweep_id}.txt")
        total_runs += num_runs
        print(f"\t- {sweep_id} ({num_runs} runs)")

    print(f"Total Runs: {total_runs}")

    response = input("Proceed? [y/n]: ").strip().lower()
    if response != "y":
        print("Aborted.")
        return

    iterator = tqdm.tqdm(ids)
    for sweep_id in iterator:
        iterator.set_description(f"{sweep_id}")
        sweep = sweep_objects.get(sweep_id)

        cmd = " ".join(
            [
                "cat",
                f"cc/sweeps/{sweep_id}.txt",
                "|",
                "parallel -q uv run cc/slurm/run-starter.py --runtime.short"
                f"--run_id={{}} --wandb-proj {project} --jobname='\"{sweep}\"'",
            ]
        )
        process_out = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
        )
        process_stderr = process_out.stderr.decode("utf-8").strip()
        if len(process_stderr) != 0:
            raise Exception("Could not start job: " + process_stderr)


if __name__ == "__main__":
    tyro.cli(main)
