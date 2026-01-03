wandb_project="cheetah-sampler2"
success_threshold=0.6
# for seed in 0 1 2 3 4
# do
#     python train.py policy=ppo  wandb_project=$wandb_project seed=$seed
# done
# for seed in 0 1 2 3 4
# do
#     python train.py policy=ppo_nodr wandb_project=$wandb_project seed=$seed
# done
sampler_update_freq=1
for success_threshold in 0.6 0.65 0.7
do
    python train.py policy=adrppo wandb_project=$wandb_project success_threshold=$success_threshold sampler_update_freq=$sampler_update_freq
    for success_rate_condition in 0.6 0.7 0.8
    do
       python train.py policy=doraemonppo wandb_project=$wandb_project success_threshold=$success_threshold success_rate_condition=$success_rate_condition sampler_update_freq=$sampler_update_freq
    done
done

