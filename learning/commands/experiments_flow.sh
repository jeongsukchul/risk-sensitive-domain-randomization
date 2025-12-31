wandb_project="cheetah-sampler"
for beta in 0 0.5 1 2
    do
    python train.py policy=flowppo beta=$beta wandb_project=$wandb_project n_sampler_iters=30
    done