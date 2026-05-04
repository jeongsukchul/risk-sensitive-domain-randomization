#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="${RESULTS_DIR:-compare_GBS_GMMVI}"
DIMS=(2 5 10 20)
SEEDS=(0 1 2 3 4)
ALGORITHMS=(gmmvi dis_lv)
TARGETS=(target_A target_B target_C)

declare -A TARGET_BETAS=(
  [target_A]=-5
  [target_B]=-10
  [target_C]=-10
)

for target in "${TARGETS[@]}"; do
  beta="${TARGET_BETAS[$target]}"
  for dim in "${DIMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo "Running ${target} | dim=${dim} | seed=${seed}"
      python algorithm_comparison.py \
        --dim "$dim" \
        --target-version "$target" \
        --algorithms "${ALGORITHMS[@]}" \
        --beta "$beta" \
        --x_axis policy_update_steps \
        --seed "$seed" \
        --output_dir "$RESULTS_DIR"
    done
  done
done

python make_metric_table.py \
  --results-dir "$RESULTS_DIR" \
  --dims "${DIMS[@]}" \
  --output "$RESULTS_DIR/initial_final_metrics_table.tex"
