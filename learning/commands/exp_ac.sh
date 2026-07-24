wandb_project="ac-fixedkl"
#fixed kl version
dual_lr=1.
for seed in 10 11 12 13 14 15
    do
    python run.py policy=gmmppo fixed_radius=true kl_radius=0.1 \
        dual_lr=$dual_lr task=AcrobotSwingup wandb_project=$wandb_project seed=$seed
    python run.py policy=gmmppo fixed_radius=true kl_radius=0.5 \
        dual_lr=$dual_lr task=AcrobotSwingup wandb_project=$wandb_project seed=$seed
    python run.py policy=gmmppo fixed_radius=true kl_radius=1 \
        dual_lr=$dual_lr task=AcrobotSwingup wandb_project=$wandb_project seed=$seed
    python run.py policy=gmmppo fixed_radius=true kl_radius=5\
        dual_lr=$dual_lr task=AcrobotSwingup wandb_project=$wandb_project seed=$seed
    done
#fixed beta version
for beta in -30 -20 -10 do
    for seed in 11 12 13 14 15 
    do
    python run.py policy=gmmppo fixed_radius=false \
        beta=$beta task=AcrobotSwingup wandb_project=$wandb_project seed=$seed
    done
done

