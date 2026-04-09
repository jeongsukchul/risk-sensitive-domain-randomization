wandb_project="push-sampler3"
for seed in 1 2 3 4 5 6 7
    do
    python run.py policy=gmmppo start_beta=5 end_beta=-30 use_scheduling=True wandb_project=$wandb_project task=PandaRobotiqPushCube seed=$seed impl=warp reset_randomization_in_domain_randomization=false e sampler_visualization=false
    python run.py policy=gbsppo start_beta=5 end_beta=-30 use_scheduling=True wandb_project=$wandb_project task=PandaRobotiqPushCube seed=$seed impl=warp reset_randomization_in_domain_randomization=false sampler_visualization=false
    python run.py policy=ppo wandb_project=$wandb_project task=PandaRobotiqPushCube seed=$seed impl=warp reset_randomization_in_domain_randomization=false sampler_visualization=false
    done

# for beta in -30 -10 0 5 10
#     do
#     for seed in 0 1 2 3 4 5 6 7
#         do
#         python run.py policy=gbsppo beta=$beta wandb_project=$wandb_project task=PandaRobotiqPushCube seed=$seed impl=warp reset_randomization_in_domain_randomization=true  sampler_visualization=false
#         done
#     done
