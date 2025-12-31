wandb_project="cheetah-sampler"
success_threshold=0.6
python train.py policy=ppo  wandb_project=$wandb_project 
python train.py policy=ppo_nodr wandb_project=$wandb_project
for success_threshold in 0.6 0.65 0.7
do
    python train.py policy=adrppo wandb_project=$wandb_project success_threshold=$success_threshold
    for success_rate_condition in 0.6 0.7 0.8
    do
       python train.py policy=doraemonppo wandb_project=$wandb_project success_threshold=$success_threshold success_rate_condition=$success_rate_condition
    done
done

