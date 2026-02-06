import subprocess

import wandb


def main():
    api = wandb.Api()
    for sweep_id in [
        "0c82kz7f",
        "c7nouohm",
        "i9pjj7lx",
        "ytk0kihd",
        "f4ygd9ca",
        "qsl1z58c",
        "fu353o8e",
        "nbvden06",
        "3aewd5rh",
        "cyfr40bb",
        "mmzaqd58",
        "g0htg6ku",
    ]:
        sweep = api.sweep(f"psaunder/Testing Mu-gdp/{sweep_id}")
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
    main()
