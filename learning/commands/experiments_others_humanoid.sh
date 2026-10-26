#!/usr/bin/env bash
# Run from any directory using the installed project.
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

wandb_project="humanoid-sampler"
task=HumanoidWalk
for seed in 0 1 2 3 4
do
python -m learning.run policy=ppo task=$task wandb_project=$wandb_project seed=$seed
python -m learning.run policy=ppo_nodr task=$task wandb_project=$wandb_project seed=$seed
done
for success_threshold in 0.65 0.7
do
    python -m learning.run policy=adrppo wandb_project=$wandb_project  task=$task success_threshold=$success_threshold
    for success_rate_condition in 0.6 0.7 0.8
    do
       python -m learning.run policy=doraemonppo wandb_project=$wandb_project success_threshold=$success_threshold task=$task success_rate_condition=$success_rate_condition
    done
done
