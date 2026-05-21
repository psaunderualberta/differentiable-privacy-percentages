def make_dp_psac_ref_cmd(
    run_id: str,
    entity: str,
    project: str,
    dataset: str,
    batch_size: int,
    lr: float,
    delta: float,
    arch: str,
) -> str:
    """Return the dp_psac_ref/run.py command that reproduces this run's evaluation settings."""
    return (
        f"cd dp_psac_ref && uv run run.py"
        f" --dataset {dataset}"
        f" --batch-size {batch_size}"
        f" --lr {lr}"
        f" --delta {delta}"
        f" --arch {arch}"
        " schedule:wandb-schedule"
        f" --schedule.run-id {run_id}"
        f" --schedule.entity {entity}"
        f" --schedule.project {project}"
    )
