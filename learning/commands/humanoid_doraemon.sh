wandb_project="humanoid"
task=HumanoidWalk

success_threshold=.6
success_rate_condition=.6
for trust_region in 0.01 0.005
do
for success_rate_condition in 0.5 0.7 0.8
do
    for seed in 1 2 3 4
    do
       python run.py policy=doraemonppo task=$task wandb_project=$wandb_project success_threshold=$success_threshold success_rate_condition=$success_rate_condition seed=$seed trust_region=$trust_region
    done
done
done
