import subprocess
import time

import tqdm
import tyro

import wandb


def main(sweep_ids: list[str]):
    api = wandb.Api()
    iterator = tqdm.tqdm(sweep_ids)
    for sweep_id in iterator:
        iterator.set_description(f"{sweep_id}")
        try:
            sweep = api.sweep(f"psaunder/Testing Mu-gdp/{sweep_id}")
        except wandb.Error as e:
            print("Error", e)
            continue

        cmd = f"cat cc/sweeps/{sweep_id}.txt | parallel -q uv run cc/slurm/run-starter.py --run_id={{}} --jobname='\"{sweep.name}\"'"
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
