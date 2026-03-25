wandb_project="leap-sampler"
for seed in 0 1 2 3 4 5 6 7
    do
    python run.py policy=ppo wandb_project=$wandb_project task=LeapCubeReorient seed=$seed impl=jax
    python run.py policy=gbsppo beta=-20 wandb_project=$wandb_project task=LeapCubeReorient seed=$seed impl=jax
    done

for beta in -30 -10 0 5 10
    do
    for seed in 0 1 2 3 4 5 6 7
        do
        python run.py policy=gbsppo beta=$beta wandb_project=$wandb_project task=LeapCubeReorient seed=$seed impl=jax
        done
    done