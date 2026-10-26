#!/usr/bin/env bash
# Run from any directory using the installed project.
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

wandb_project="comp"
profile_training_cost=true
#fixed kl version
# #fixed beta version
task=CheetahRun
for seed in 11 12 13
do
python -m learning.run policy=gmmppo fixed_radius=false \
    beta=-10 task=$task wandb_project=$wandb_project seed=$seed profile_training_cost=$profile_training_cost
python -m learning.run policy=ppo task=$task wandb_project=$wandb_project seed=$seed profile_training_cost=$profile_training_cost
python -m learning.run policy=epoptppo epsilon=0.4 task=$task wandb_project=$wandb_project seed=$seed profile_training_cost=$profile_training_cost
done
task=WalkerWalk
for seed in 11 12 13
do
python -m learning.run policy=gmmppo fixed_radius=false \
    beta=-10 task=$task wandb_project=$wandb_project seed=$seed profile_training_cost=$profile_training_cost
python -m learning.run policy=ppo task=$task wandb_project=$wandb_project seed=$seed profile_training_cost=$profile_training_cost
python -m learning.run policy=epoptppo epsilon=0.8 task=$task wandb_project=$wandb_project seed=$seed profile_training_cost=$profile_training_cost
done
