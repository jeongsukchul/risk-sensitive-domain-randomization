#!/usr/bin/env bash
# Run from any directory using the installed project.
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

wandb_project="ac-fixedkl"
#fixed kl version
dual_lr=1.
for seed in 16 17 18 19
    do
    python -m learning.run policy=gmmppo fixed_radius=true kl_radius=0.1 \
        dual_lr=$dual_lr task=AcrobotSwingup wandb_project=$wandb_project seed=$seed
    python -m learning.run policy=gmmppo fixed_radius=true kl_radius=0.5 \
        dual_lr=$dual_lr task=AcrobotSwingup wandb_project=$wandb_project seed=$seed
    python -m learning.run policy=gmmppo fixed_radius=true kl_radius=1 \
        dual_lr=$dual_lr task=AcrobotSwingup wandb_project=$wandb_project seed=$seed
    python -m learning.run policy=gmmppo fixed_radius=true kl_radius=1.5 \
        dual_lr=$dual_lr task=AcrobotSwingup wandb_project=$wandb_project seed=$seed
    python -m learning.run policy=gmmppo fixed_radius=true kl_radius=2.0 \
        dual_lr=$dual_lr task=AcrobotSwingup wandb_project=$wandb_project seed=$seed
#     python -m learning.run policy=gmmppo fixed_radius=true kl_radius=5\
#         dual_lr=$dual_lr task=AcrobotSwingup wandb_project=$wandb_project seed=$seed
    done
# #fixed beta version
# for beta in -30 -20 -10; do
#     for seed in 11 12 13 14 15
#     do
#     python -m learning.run policy=gmmppo fixed_radius=false \
#         beta=$beta task=AcrobotSwingup wandb_project=$wandb_project seed=$seed
#     done
# done
