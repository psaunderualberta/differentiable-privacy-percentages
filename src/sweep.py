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

    wandb.agent(
        sweep_id,
        function=lambda: wandb.init(),
        count=1,
    )
