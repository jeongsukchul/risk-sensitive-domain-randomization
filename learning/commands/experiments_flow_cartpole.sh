wandb_project=cartpole-sampler10
task=CartpoleSwingup
success_threshold=.7

for beta in 1 
do
for gamma in 1. 2.
do
    for seed in 0 1 2 3
    do
       python run.py policy=flowppo task=$task wandb_project=$wandb_project beta=$beta gamma=$gamma seed=$seed 
    done
done
done
for beta in 0.66 0.5
do
for gamma in 0. .5 1. 2.
do
    for seed in 0 1 2 3 
    do
       python run.py policy=flowppo task=$task wandb_project=$wandb_project beta=$beta gamma=$gamma seed=$seed 
    done
done
done
