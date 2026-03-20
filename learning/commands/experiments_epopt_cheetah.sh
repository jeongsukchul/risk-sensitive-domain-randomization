task=CheetahRun
wandb_project="cheetah-sampler2"

for epsilon in 0.8 0.7 0.6
do
    for seed in 105 106 107 108 109
    do
    python run.py policy=epoptppo task=$task wandb_project=$wandb_project seed=$seed epsilon=$epsilon
    # python run.py policy=ppo_nodr task=$task wandb_project=$wandb_project seed=$seed
    done
done