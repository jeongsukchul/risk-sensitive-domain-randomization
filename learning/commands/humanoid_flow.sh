wandb_project="humanoid"
task=HumanoidWalk

# success_threshold=.6
# success_rate_condition=.6
# trust_region=0.005
# for seed in 100 101 102 103 104
# do
#     python run.py policy=flowppo wandb_project=$wandb_project task=$task beta=1. seed=$seed gamma=.5
# done


for beta in 2 1 0.66 0.5
do
for gamma in 0. .5 1. 2.
do
    for seed in 0 1 2 3 4
    do
       python run.py policy=flowppo task=$task wandb_project=$wandb_project beta=$beta gamma=$gamma seed=$seed 
    done
done
done
