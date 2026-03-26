wandb_project="aloha-sampler"
for seed in 0 1 2 3 4 5 6 7
    do
    python run.py policy=ppo impl=warp wandb_project=$wandb_project impl=warp task=AlohaSinglePegInsertion seed=$seed 
    python run.py policy=gmmppo impl=warp beta=-20 wandb_project=$wandb_project impl=warp task=AlohaSinglePegInsertion seed=$seed 
    done
for beta in -30 -10 0 5 10
    do
    for seed in 0 1 2 3 4 5 6 7
        do
        python run.py policy=gbsppo impl=warp beta=$beta wandb_project=$wandb_project impl=warp task=AlohaSinglePegInsertion seed=$seed 
        done
    done