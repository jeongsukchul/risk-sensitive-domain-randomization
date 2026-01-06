# wandb_project="walkerwalk-sampler"
wandb_project="cheetah-sampler3"


for seed in 0 1 2 3 4 5 6 7 8 9
do
    # python train.py policy=ppo wandb_project=$wandb_project impl=jax
    python train.py policy=gmmppo beta=0 wandb_project=$wandb_project task=CheetahRun seed=$seed sampler_update_freq=1
done