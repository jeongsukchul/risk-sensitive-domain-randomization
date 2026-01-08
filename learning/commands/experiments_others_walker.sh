wandb_project="walker-sampler3"

# for seed in 0 1 2 3 4
# do
# python run.py policy=ppo task=WalkerWalk wandb_project=$wandb_project seed=$seed
# python run.py policy=ppo_nodr task=WalkerWalk wandb_project=$wandb_project seed=$seed
# done
success_threshold=.7
success_rate_condition=.8
for seed in 0 1 2 3 4
do
    python run.py policy=adrppo wandb_project=$wandb_project task=WalkerWalk success_threshold=$success_threshold seed=$seed
    python run.py policy=doraemonppo wandb_project=$wandb_project success_threshold=$success_threshold task=WalkerWalk seed=$seed success_rate_condition=$success_rate_condition
done

