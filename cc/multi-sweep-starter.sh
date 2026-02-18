# MOMENTUM=0.7
MOMENTUMS=(0.1 0.7)
SCHEDULE_TYPES=("alternating_schedule" "sigma_and_clip_schedule")
for i in {0..1}; do
    MOMENTUM=${MOMENTUMS[i]}
    SCHEDULE_TYPE=${SCHEDULE_TYPES[i]}
    time for eps in 0.4 1.2 3.0; do
        for dataset in "fashion-mnist" "mnist"; do
            time uv run sweep.py --wandb_conf.project="Testing Mu-gdp" --wandb-conf.entity psaunder --wandb-conf.mode online \
                --sweep.total_timesteps 2000  --sweep.policy.batch_size 12 --sweep.env.eps $eps --sweep.env.delta 1e-6 --sweep.env.network_type cnn \
                --sweep.env.max_steps_in_episode 3000 --sweep.env.optimizer sgd --sweep.policy.schedule_type=$SCHEDULE_TYPE \
                --sweep.policy.momentum.value=$MOMENTUM \
                --sweep.with-baselines --sweep.dataset "$dataset" --sweep.name "$dataset, e=$eps, T=3000, $SCHEDULE_TYPE, M=$MOMENTUM"
        done
    done
done
