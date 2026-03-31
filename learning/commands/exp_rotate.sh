wandb_project="rotate-sampler2"
for seed in 0 1 2 3 4 5 6 7
    do
    python run.py policy=ppo wandb_project=$wandb_project task=LeapCubeRotateZAxis seed=$seed impl=warp sampler_visualization=true
    python run.py policy=gmmppo beta=-20 wandb_project=$wandb_project task=LeapCubeRotateZAxis seed=$seed impl=warp sampler_visualization=true
    python run.py policy=gbsppo beta=-20 wandb_project=$wandb_project task=LeapCubeRotateZAxis seed=$seed impl=warp sampler_visualization=true
    done

for beta in -30 -10 0 5 10
    do
    for seed in 0 1 2 3 4 5 6 7
        do
        python run.py policy=gbsppo beta=$beta wandb_project=$wandb_project task=LeapCubeRotateZAxis seed=$seed impl=warp sampler_visualization=true
        done
    done