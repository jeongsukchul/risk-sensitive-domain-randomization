wandb_project="franka-sampler3"

for seed in 0 1 2 3 4 5 6 7
        do
        python run.py policy=ppo wandb_project=$wandb_project impl=warp task=PandaPickCubeOrientation seed=$seed sampler_update_freq=1 n_sampler_iters=1
        python run.py policy=gbsppo beta=-20 wandb_project=$wandb_project impl=warp task=PandaPickCubeOrientation seed=$seed sampler_update_freq=1 n_sampler_iters=1
        done
for beta in -30 -10 0 5 10
    do
    for seed in 0 1 2 3 4 5 6 7
        do
        python run.py policy=gbsppo beta=$beta wandb_project=$wandb_project impl=warp task=PandaPickCubeOrientation seed=$seed sampler_update_freq=1 n_sampler_iters=1
        done
    done