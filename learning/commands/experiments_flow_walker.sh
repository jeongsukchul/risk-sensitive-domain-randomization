wandb_project="cheetah-sampler2"
for beta in 0.5 0.666 1 2
    do
    for seed in 0 1 2 3 4
    do
        python train.py policy=flowppo beta=$beta wandb_project=$wandb_project n_sampler_iters=30 seed=$seed
    done
    done