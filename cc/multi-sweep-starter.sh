MOMENTUM=0.1
time for eps in 0.4 1.2 3.0; do
    for dataset in "fashion-mnist" "mnist"; do
        for schedule_type in "sigma_and_clip_schedule"; do
            time uv run sweep.py --wandb_conf.project="Testing Mu-gdp" --wandb-conf.entity psaunder --wandb-conf.mode online \
                --sweep.total_timesteps 3000  --sweep.policy.batch_size 12 --sweep.env.eps $eps --sweep.env.delta 1e-6 --sweep.env.network_type cnn \
                --sweep.env.max_steps_in_episode 3000 --sweep.env.optimizer sgd --sweep.policy.schedule_type=$schedule_type \
                --sweep.policy.momentum.value=$MOMENTUM \
                --sweep.with-baselines --sweep.dataset "$dataset" --sweep.name "$dataset, e=$eps, T=3000, $schedule_type, M=$MOMENTUM"
        done
    done
done
