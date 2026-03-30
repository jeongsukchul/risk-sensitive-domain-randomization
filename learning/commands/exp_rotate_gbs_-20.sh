wandb_project="rotate-sampler"
for seed in 0 1 2 3 4 5 6 7
    do
    python run.py policy=gbsppo beta=-20 wandb_project=$wandb_project task=LeapCubeRotateZAxis seed=$seed impl=warp sampler_visualization=true
    done
