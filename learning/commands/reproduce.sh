#!/usr/bin/env bash
# Fixed-temperature RSDR sweep. Extra arguments are Hydra overrides.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

read -r -a tasks <<< "${TASKS:-AcrobotSwingup CartpoleSwingup CheetahRun WalkerWalk PandaPickCubeOrientation PandaOpenCabinet}"
read -r -a seeds <<< "${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
read -r -a betas <<< "${BETAS:--30 -20 -10 -5 -2 -1 0 1 2 5 10}"

for task in "${tasks[@]}"; do
    for beta in "${betas[@]}"; do
        for seed in "${seeds[@]}"; do
            command=("${PYTHON:-python}" -m learning.run
                "task=$task" policy=gmmppo "beta=$beta" "seed=$seed"
                fixed_radius=false use_scheduling=false eval_grid_size_2d=64
                "wandb_project=${WANDB_PROJECT:-rsdr}")
            command+=("$@")
            printf '%q ' "${command[@]}"
            printf '\n'
            if [[ "${DRY_RUN:-0}" != 1 ]]; then
                "${command[@]}"
            fi
        done
    done
done
