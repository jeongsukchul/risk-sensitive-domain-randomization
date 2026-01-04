wandb_project="walker-sampler2"

for beta in -30 -20 -10 -5 -2 -1 0 1 2 5 10
    do
    for seed in 0 1 2 3 4
        do
        python train.py policy=gmmppo beta=$beta wandb_project=$wandb_project task=WalkerWalk seed=$seed sampler_update_freq=1
        done
    done