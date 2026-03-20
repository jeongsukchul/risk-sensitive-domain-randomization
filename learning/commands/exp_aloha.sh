wandb_project="alpha-sampler"
for seed in 0 1 2 3 4 5 6 7
    do
    python run.py policy=ppo wandb_project=$wandb_project impl=warp task=AlohaSinglePegInsertion seed=$seed sampler_update_freq=1 n_sampler_iters=1
    done
for beta in -30 -20 -10 0 5 10
    do
    for seed in 0 1 2 3 4 5 6 7
        do
        python run.py policy=gbsppo beta=$beta wandb_project=$wandb_project impl=warp task=AlohaSinglePegInsertion seed=$seed sampler_update_freq=1 n_sampler_iters=1
        done
    done