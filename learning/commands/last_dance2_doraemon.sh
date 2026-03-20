wandb_project="last_dance2"

task="PandaPickCubeOrientation"

for seed in 100 101 
do
    python run.py policy=doraemonppo success_threshold=.6 success_rate_condition=.7 task=$task wandb_project=$wandb_project seed=$seed
    # python run.py policy=gmmppo task=$task wandb_project=$wandb_project seed=$seed beta=-5
done
    