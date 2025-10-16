from conf.singleton_conf import SingletonConfig
import wandb


if __name__ == "__main__":
    wandb_config = SingletonConfig.get_wandb_config_instance()
    sweep_config = SingletonConfig.get_sweep_config_instance()
    sweep_id = wandb.sweep(
        sweep_config.to_wandb_sweep(),
        project=wandb_config.project,
        entity=wandb_config.entity,
    )

    ids = []

    def starter():
        run = wandb.init()
        ids.append(run.id)
        run.finish()

    wandb.agent(
        sweep_id,
        function=starter,
        count=50,
    )
