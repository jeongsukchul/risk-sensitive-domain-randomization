wandb_project="panda_cube"

task="PandaPickCubeOrientation"

for seed in 100 101 102 103 104
do
    python run.py policy=gmmppo task=$task wandb_project=$wandb_project seed=$seed beta=-20
done
    