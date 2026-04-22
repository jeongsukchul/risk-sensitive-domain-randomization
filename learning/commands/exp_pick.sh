wandb_project="pick-samplerD"
for seed in 1 2 3 4
    do
    python run.py policy=gmmppo beta=-20 wandb_project=$wandb_project use_grad_info=true task=PandaPickCubeOrientation seed=$seed impl=warp reset_randomization_in_domain_randomization=false sampler_visualization=true
    python run.py policy=gmmppo beta=-20 wandb_project=$wandb_project task=PandaPickCubeOrientation seed=$seed impl=warp reset_randomization_in_domain_randomization=false sampler_visualization=true
    python run.py policy=ppo wandb_project=$wandb_project task=PandaPickCubeOrientation seed=$seed impl=warp reset_randomization_in_domain_randomization=false sampler_visualization=false
    done