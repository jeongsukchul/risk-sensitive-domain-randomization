wandb_project="cheetah-sampler"
for beta in 0 0.5 1 2
    do
    python train.py policy=flowppo beta=$beta n_sampler_iters=100 wandb_project=$wandb_project n_sampler_iters=50
    done