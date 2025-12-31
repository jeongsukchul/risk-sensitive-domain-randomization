wandb_project="walkerwalk-sampler"

for beta in -30 -20 -10 -5 -2 -1 0 1 2 5 10
    do
    python train.py policy=gmmppo beta=$beta wandb_project=$wandb_project task=WalkerWalk
    done