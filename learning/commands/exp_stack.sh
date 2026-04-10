wandb_project="stack-sampler"
for seed in 1 2 3 4
    do
    python run.py policy=gmmppo beta=-20 wandb_project=$wandb_project task=PandaStackCube seed=$seed impl=warp reset_randomization_in_domain_randomization=false sampler_visualization=false
    python run.py policy=gbsppo beta=-20 wandb_project=$wandb_project task=PandaStackCube seed=$seed impl=warp reset_randomization_in_domain_randomization=false sampler_visualization=false
    python run.py policy=ppo wandb_project=$wandb_project task=PandaStackCube seed=$seed impl=warp reset_randomization_in_domain_randomization=false sampler_visualization=false
    done