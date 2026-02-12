#!/bin/bash
set -euo pipefail

SEEDS=(42 123 256 512 1024 2048 4096 8192 16384 32768)
RESPONSE_LENGTH=${RESPONSE_LENGTH:-256}
TOTAL_EPISODES=${TOTAL_EPISODES:-50000}
OUTPUT_BASE=${OUTPUT_BASE:-/scratch/s/shahradm}

for seed in "${SEEDS[@]}"; do
    echo "Submitting seed ${seed}"
    sbatch \
      --job-name "dartppo-s${seed}" \
      --export=ALL,SEED=${seed},RESPONSE_LENGTH=${RESPONSE_LENGTH},TOTAL_EPISODES=${TOTAL_EPISODES},OUTPUT_BASE=${OUTPUT_BASE} \
      ppo_vs_dart_seed_job.sh
done

echo "Submitted ${#SEEDS[@]} jobs (1 node each), each running DART+PPO for one seed."
