wandb_project="cartpole-sampler3"
for seed in 0 1 2 3 4
do
python run.py policy=ppo task=CartpoleSwingup wandb_project=$wandb_project seed=$seed
python run.py policy=ppo_nodr task=CartpoleSwingup wandb_project=$wandb_project seed=$seed
done
for success_threshold in 0.65 0.7
do
    python run.py policy=adrppo wandb_project=$wandb_project  task=CartpoleSwingup success_threshold=$success_threshold
    for success_rate_condition in 0.6 0.7 0.8
    do
       python run.py policy=doraemonppo wandb_project=$wandb_project success_threshold=$success_threshold task=CartpoleSwingup success_rate_condition=$success_rate_condition
    done
done

