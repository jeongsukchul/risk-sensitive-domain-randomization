#!/usr/bin/env bash
# Evaluate saved policies. Pass --sampler_checkpoint for a shared Langevin attack.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
read -r -a seeds <<< "${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
"${PYTHON:-python}" -m learning.eval_nonstationary \
    --task "${TASK:-AcrobotSwingup}" \
    --policy "${POLICY:-ppo}" \
    --dist_type "${DIST_TYPE:-random_walk}" \
    --seeds "${seeds[@]}" "$@"
