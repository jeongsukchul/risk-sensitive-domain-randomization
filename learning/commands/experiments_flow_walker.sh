wandb_project="walker-sampler3"
for seed in 0 1 4
do
    python run.py policy=adrppo wandb_project=$wandb_project seed=$seed task=WalkerWalk
done
for beta in 0.5 0.666 1 2
    do
    for seed in 0 1 2 3 4
    do
        python run.py policy=flowppo beta=$beta wandb_project=$wandb_project n_sampler_iters=30 seed=$seed task=WalkerWalk
    done
    done