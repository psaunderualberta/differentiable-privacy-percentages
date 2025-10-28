from conf.singleton_conf import SingletonConfig
import wandb
import os


__CC_ROOT = os.environ["PROJECT_ROOT"] = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "cc")
)


if __name__ == "__main__":
    wandb_config = SingletonConfig.get_wandb_config_instance()
    sweep_config = SingletonConfig.get_sweep_config_instance()
    sweep_id = wandb.sweep(
        sweep_config.to_wandb_sweep(),
        project=wandb_config.project,
        entity=wandb_config.entity,
    )

    def starter():
        run = wandb.init()
        with open(os.path.join(__CC_ROOT, "sweeps", f"{sweep_id}.txt"), "a") as f:
            f.write(run.id + "\n")
        run.finish()

    wandb.agent(
        sweep_id,
        function=starter,
        count=50,
    )
