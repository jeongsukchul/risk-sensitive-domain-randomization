#!/usr/bin/env bash
# Run from any directory using the installed project.
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

wandb_project="cheetah-sampler2"
for beta in 0 0.5 1 2
    do
    for seed in 0 1 2 3 4
    do
        python -m learning.run policy=flowppo beta=$beta wandb_project=$wandb_project n_sampler_iters=30 seed=$seed
    done
    done
