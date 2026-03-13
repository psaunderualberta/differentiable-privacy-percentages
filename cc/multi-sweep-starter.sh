time for eps in 0.4; do
    for dataset in "mnist"; do
        time uv run sweep.py sweep.env.network:cnn \
        --wandb_conf.project="Testing Mu-gdp" --wandb-conf.entity psaunder --wandb-conf.mode online \
            --sweep.total_timesteps 2000  --sweep.policy.batch_size 12 --sweep.env.eps $eps --sweep.env.delta 1e-6  \
            --sweep.env.max_steps_in_episode 3000 --sweep.env.optimizer sgd  \
            --sweep.plotting_interval 10 --sweep.with-baselines --sweep.dataset "$dataset" \
            --sweep.name "$dataset, e=$eps, T=3000"
    done
done
