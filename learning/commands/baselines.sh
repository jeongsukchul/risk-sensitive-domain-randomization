#!/usr/bin/env bash
# Baseline sweep; tune sampler hyperparameters through Hydra overrides.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

read -r -a tasks <<< "${TASKS:-AcrobotSwingup CartpoleSwingup CheetahRun WalkerWalk PandaPickCubeOrientation PandaOpenCabinet}"
read -r -a seeds <<< "${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
read -r -a policies <<< "${POLICIES:-ppo ppo_nodr epoptppo adrppo doraemonppo flowppo}"

for task in "${tasks[@]}"; do
    threshold=0.7
    if [[ "$task" == Panda* ]]; then threshold=0.5; fi
    for policy in "${policies[@]}"; do
        for seed in "${seeds[@]}"; do
            command=("${PYTHON:-python}" -m learning.run
                "task=$task" "policy=$policy" "seed=$seed"
                fixed_radius=false use_scheduling=false eval_grid_size_2d=64
                "success_threshold=$threshold" "wandb_project=${WANDB_PROJECT:-rsdr}")
            if [[ "$policy" == flowppo ]]; then command+=(beta=1 gamma=0.5); fi
            command+=("$@")
            printf '%q ' "${command[@]}"
            printf '\n'
            if [[ "${DRY_RUN:-0}" != 1 ]]; then
                "${command[@]}"
            fi
        done
    done
done
