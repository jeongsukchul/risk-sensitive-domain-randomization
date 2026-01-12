task=CartpoleSwingupSparse
wandb_project="cartpole-sampler6"
for seed in 5
do
python run.py policy=ppo task=$task wandb_project=$wandb_project seed=$seed
python run.py policy=ppo_nodr task=$task wandb_project=$wandb_project seed=$seed
done
success_threshold= 0.7
success_rate_condition= 0.8
for seed in 0 1 2 3 4
do
    python run.py policy=adrppo wandb_project=$wandb_project  task=$task success_threshold=$success_threshold seed=$seed
    python run.py policy=doraemonppo wandb_project=$wandb_project success_threshold=$success_threshold task=$task success_rate_condition=$success_rate_condition seed=$seed
done

