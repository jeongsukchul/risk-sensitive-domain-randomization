wandb_project="franka-sampler"
for seed in 0 1 2 3 4 5 6
    do
    python run.py policy=ppo wandb_project=$wandb_project task=PandaPickCubeOrientation seed=$seed impl=warp sampler_update_freq=1
    # python run.py policy=gmmppo beta=-30 wandb_project=$wandb_project task=PandaPickCubeOrientation seed=$seed impl=warp sampler_update_freq=1
    # python run.py policy=gmmppo beta=-20 wandb_project=$wandb_project task=PandaPickCubeOrientation seed=$seed impl=warp sampler_update_freq=1
    # python run.py policy=gmmppo beta=-10 wandb_project=$wandb_project task=PandaPickCubeOrientation seed=$seed impl=warp sampler_update_freq=1
    done