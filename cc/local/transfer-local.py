"""Run the transfer-evaluation DAG on this machine instead of SLURM (ADR 0008).

The local twin of ``cc/slurm/transfer-run-starter.py``. Both build their task list
with :func:`transfer_launch.plan_jobs`, so a cell run here is the *same* command,
writing the *same* file, that the cluster would have produced — you can start a
matrix locally and finish it on SLURM, or the reverse, and the skip filter joins
them up.

What replaces SLURM:

  * the job array  -> a fixed pool of workers, one per GPU, pulling from a shared
    queue. Producers are single-device (the eval core is a Python loop over
    ``train_with_noise``), so one process per GPU saturates the box.
  * ``--gpus 1``   -> ``CUDA_VISIBLE_DEVICES=<n>`` on each worker. This is load
    bearing: without it every process would see all four GPUs and JAX would
    pre-allocate a chunk of each, so the second process onward dies at import.
  * ``afterok``    -> the preflight simply runs to completion first, in this
    process. It stays sequential for the same reason it does on the cluster:
    ``util/dataloaders.py`` downloads into the repo with no locking.
  * the wall clock -> nothing. Tasks run to completion; ctrl-C stops scheduling new
    ones. Since cell writes are atomic and finished cells are skipped, re-running
    the identical command resumes.

Examples:
    # Plan only — print what would run, per stage, and touch nothing:
    uv run cc/local/transfer-local.py \\
        --cache_root cache \\
        --schedules_parquet cache/results/psaunder__NoMomentumSweep/schedules.parquet \\
        --target_datasets eyepacs imagenet chexpert \\
        --target_eps 1.0 8.0 --target_T 200 5000 --stages curve reference --dry-run

    # Run it across all four GPUs:
    uv run cc/local/transfer-local.py ... --gpu_ids 0 1 2 3
"""

import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import tyro

os.environ["PROJECT_ROOT"] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", ".."),
)
os.environ["PROJECT_SOURCE_ROOT"] = os.path.abspath(
    os.path.join(os.environ["PROJECT_ROOT"], "src"),
)

sys.path.insert(0, os.environ["PROJECT_SOURCE_ROOT"])
from transfer_launch import (
    Job,
    ProducerArgs,
    absolute_path,
    condition_grid,
    expand_targets,
    plan_jobs,
    preflight_command,
)


@dataclass
class TransferLocalConfig:
    """Run the transfer-evaluation DAG for one target cross-product on this machine."""

    cache_root: str
    """Where cells are written and the skip filter looks. Required, no default."""

    target_datasets: tuple[str, ...] = ("eyepacs", "imagenet", "chexpert")
    target_eps: tuple[float, ...] = (1.0, 8.0)
    target_T: tuple[int, ...] = (200, 5000)
    target_delta: float = 1e-7

    schedules_parquet: str = ""
    """Source schedules.parquet from compile_results_fetch. Required for the curve stage."""
    eval_dir: str = ""
    """SR eval dir (equations.csv + category_map.json). Required for the equation stage."""

    stages: tuple[str, ...] = ("curve", "equation", "reference")
    with_plot: bool = False
    """Also run the assembler once every producer job has succeeded."""

    num_reps: int = 8
    seed: int = 0
    batch_size: int = 250

    gpu_ids: tuple[int, ...] = (0, 1, 2, 3)
    """Physical GPUs to use. One worker is pinned to each."""
    workers_per_gpu: int = 1
    """Processes sharing each GPU. >1 disables JAX pre-allocation so they can co-exist."""
    skip_preflight: bool = False
    """Skip the dataset warm-up. Only safe once every target dataset is already cached."""
    logdir: str = os.path.join(os.environ["PROJECT_ROOT"], "cc", "logs", "transfer-local")
    dry_run: bool = False
    """Print the plan and the per-worker commands; run nothing."""

    @property
    def producer_args(self) -> ProducerArgs:
        # Absolute, always — see transfer_launch.absolute_path. Workers run with
        # cwd=src/ while you invoke this from anywhere, so a relative --cache_root
        # would have the skip filter testing one directory and the producer writing
        # another — silently, and only for the cells already computed.
        return ProducerArgs(
            cache_root=absolute_path(self.cache_root),
            schedules_parquet=absolute_path(self.schedules_parquet),
            eval_dir=absolute_path(self.eval_dir),
            num_reps=self.num_reps,
            seed=self.seed,
            batch_size=self.batch_size,
        )


def worker_env(gpu: int, workers_per_gpu: int) -> dict[str, str]:
    """The environment one worker's producer runs under.

    ``CUDA_VISIBLE_DEVICES`` is the whole isolation mechanism: the child sees a
    single device and indexes it as GPU 0, so nothing in ``src/`` needs to know it
    is one of several. When GPUs are shared, JAX's default 75%-up-front
    pre-allocation would let the first arrival starve the rest, so back it off to an
    even split and let the allocator grow on demand.
    """
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    if workers_per_gpu > 1:
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{0.9 / workers_per_gpu:.3f}"
    return env


def run_command(command: str, logfile: Path, env: dict[str, str]) -> int:
    """Run one producer invocation, teeing its output to ``logfile``.

    The manifest line is a shell command (``uv run ...``), so it goes through the
    shell exactly as the sbatch template's ``eval "$CMD"`` would. Output is
    file-only rather than interleaved on the terminal: four concurrent tqdm bars
    are unreadable, and the log is what you want on a failure anyway.
    """
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("w") as handle:
        handle.write(f"# {command}\n# CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n\n")
        handle.flush()
        return subprocess.run(
            command,
            shell=True,
            cwd=os.environ["PROJECT_SOURCE_ROOT"],
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        ).returncode


def run_pool(
    tasks: list[tuple[str, int, Job]],
    gpu_ids: tuple[int, ...],
    workers_per_gpu: int,
    logdir: Path,
) -> list[tuple[str, int, Job]]:
    """Run every task across a GPU-pinned worker pool; return the ones that failed.

    Work is pulled from a shared queue rather than dealt out up front, because task
    costs within a stage vary by an order of magnitude (T=200 against T=5000) and a
    static split would leave three GPUs idle behind the longest cell.
    """
    pending: queue.Queue = queue.Queue()
    for task in tasks:
        pending.put(task)

    failures: list[tuple[str, int, Job]] = []
    lock = threading.Lock()
    started = time.monotonic()

    def worker(gpu: int) -> None:
        env = worker_env(gpu, workers_per_gpu)
        while True:
            try:
                stage, index, job = pending.get_nowait()
            except queue.Empty:
                return
            logfile = logdir / stage / f"{index:04d}.log"
            code = run_command(job.args, logfile, env)
            with lock:
                done = len(tasks) - pending.qsize()
                status = "ok" if code == 0 else f"FAILED (exit {code})"
                elapsed = time.monotonic() - started
                print(
                    f"[{done}/{len(tasks)}] {elapsed / 60:6.1f}m gpu{gpu} "
                    f"{stage}[{index}] {status} -> {logfile}",
                    flush=True,
                )
                if code != 0:
                    failures.append((stage, index, job))

    threads = [
        threading.Thread(target=worker, args=(gpu,), daemon=True)
        for gpu in gpu_ids
        for _ in range(workers_per_gpu)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return failures


def main(conf: TransferLocalConfig) -> None:
    print("=== planning ===")
    targets = expand_targets(
        conf.target_datasets, conf.target_eps, conf.target_T, conf.target_delta
    )
    args = conf.producer_args
    grid = condition_grid(Path(args.eval_dir) / "category_map.json") if args.eval_dir else set()
    jobs = plan_jobs(conf.stages, targets, args, grid)

    # Stage order is preserved but stages are *not* barriered: the three producers
    # are independent (they share only the preflight), so holding GPUs idle at a
    # stage boundary would buy nothing.
    tasks = [
        (stage, index, job)
        for stage in conf.stages
        for index, job in enumerate(jobs.get(stage, []))
    ]
    if not tasks:
        raise SystemExit("nothing to do: every requested cell already exists")

    slots = len(conf.gpu_ids) * conf.workers_per_gpu
    logdir = Path(conf.logdir)
    print(f"  {len(tasks)} task(s) over {slots} worker(s) on GPUs {list(conf.gpu_ids)}")
    print(f"  logs: {logdir}")

    if conf.dry_run:
        print("\n=== preflight ===")
        print(f"  {preflight_command(targets, args)}")
        for stage, index, job in tasks:
            print(f"  [{stage} {index}] {job.args}")
        print("\n[dry-run] nothing was run")
        return

    if not conf.skip_preflight:
        print("\n=== preflight (sequential; dataset downloads are not lock-safe) ===")
        # CPU-only, and explicitly given no GPU so it cannot collide with a worker.
        code = run_command(
            preflight_command(targets, args),
            logdir / "preflight.log",
            dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        )
        if code != 0:
            raise SystemExit(f"preflight failed (exit {code}); see {logdir / 'preflight.log'}")
        print("  preflight ok")

    print(f"\n=== producers ({len(tasks)} tasks) ===")
    failures = run_pool(tasks, conf.gpu_ids, conf.workers_per_gpu, logdir)

    if failures:
        print(f"\n{len(failures)} task(s) failed:")
        for stage, index, _ in failures:
            print(f"  {stage}[{index}] -> {logdir / stage / f'{index:04d}.log'}")
        print("re-running this command retries only the failures (finished cells are skipped)")
        raise SystemExit(1)

    print("\nall tasks completed")
    if conf.with_plot:
        print("\n=== plot ===")
        code = run_command(
            f"uv run --no-sync transfer_plot.py --cache_root {args.cache_root}",
            logdir / "plot.log",
            dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        )
        if code != 0:
            raise SystemExit(f"plot failed (exit {code}); see {logdir / 'plot.log'}")
        print("  plot ok")


if __name__ == "__main__":
    main(tyro.cli(TransferLocalConfig))
