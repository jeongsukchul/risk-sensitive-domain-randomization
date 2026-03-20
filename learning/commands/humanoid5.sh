wandb_project="humanoid"
task=HumanoidWalk
for beta in 5
    do
    for seed in 100 101 102 103 104
        do
        python run.py policy=gmmppo beta=$beta wandb_project=$wandb_project task=$task seed=$seed sampler_update_freq=1
        done
    done
# use_scheduling=true
# start_beta=10
# end_beta=-30
# scheduler_mode="linear"

# for seed in 0 1 2
# do
# python run.py policy=gmmppo use_scheduling=true start_beta=$start_beta end_beta=$end_beta scheduler_mode=$scheduler_mode wandb_project=$wandb_project task=$task seed=$seed sampler_update_freq=1
# done