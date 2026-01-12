# wandb_project="walkerwalk-sampler"
wandb_project="cartpole-sampler6"

for beta in 5
    do
    for seed in 2 3 4
        do
        python run.py policy=gmmppo beta=$beta wandb_project=$wandb_project task=CartpoleSwingupSparse seed=$seed sampler_update_freq=1
        done
    done
for beta in 10
    do
    for seed in 0 1 2 3 4
        do
        python run.py policy=gmmppo beta=$beta wandb_project=$wandb_project task=CartpoleSwingupSparse seed=$seed sampler_update_freq=1
        done
    done
