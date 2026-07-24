wandb_project="pick-fixedkl"
#fixed kl version
task=PandaPickCubeOrientation
dual_lr=1.
for seed in 10 11 12 13 14 15
    do
    python run.py policy=gmmppo fixed_radius=true kl_radius=0.01 \
        dual_lr=$dual_lr task=$task wandb_project=$wandb_project seed=$seed
    python run.py policy=gmmppo fixed_radius=true kl_radius=0.05 \
        dual_lr=$dual_lr task=$task wandb_project=$wandb_project seed=$seed
    python run.py policy=gmmppo fixed_radius=true kl_radius=0.1 \
        dual_lr=$dual_lr task=$task wandb_project=$wandb_project seed=$seed
    python run.py policy=gmmppo fixed_radius=true kl_radius=0.5 \
        dual_lr=$dual_lr task=$task wandb_project=$wandb_project seed=$seed

    done
#fixed beta version
for beta in -30 -20 -10 do
do
    for seed in 11 12 13 14 15 
    do
    python run.py policy=gmmppo fixed_radius=false \
        beta=$beta task=$task wandb_project=$wandb_project seed=$seed
    done
done

