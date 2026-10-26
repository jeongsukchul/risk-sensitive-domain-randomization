#!/usr/bin/env bash
# Run from any directory using the installed project.
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

wandb_project="cheetah-fixedkl"
#fixed kl version
task=CheetahRun
for seed in 21 22
    do
    # python -m learning.run policy=gmmppo fixed_radius=true kl_radius=0.1 \
    #      task=$task wandb_project=$wandb_project seed=$seed
    # python -m learning.run policy=gmmppo fixed_radius=true kl_radius=0.5 \
    #      task=$task wandb_project=$wandb_project seed=$seed
    python -m learning.run policy=gmmppo fixed_radius=true kl_radius=1.0 \
         task=$task wandb_project=$wandb_project seed=$seed
    # python -m learning.run policy=gmmppo fixed_radius=true kl_radius=1.5 \
    #      task=$task wandb_project=$wandb_project seed=$seed
    # python -m learning.run policy=gmmppo fixed_radius=true kl_radius=2.0 \
    #      task=$task wandb_project=$wandb_project seed=$seed
    done
