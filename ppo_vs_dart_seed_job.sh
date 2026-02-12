#!/bin/bash
#SBATCH --job-name=dart_ppo_seed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=0
#SBATCH --account=aip-rrabba
#SBATCH --time=7:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

source .env

SEED=${SEED:?SEED is required, e.g. sbatch --export=ALL,SEED=42 ppo_vs_dart_seed_job.sh}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-256}
TOTAL_EPISODES=${TOTAL_EPISODES:-50000}
OUTPUT_BASE=${OUTPUT_BASE:-/scratch/s/shahradm}

# NCCL config
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=3600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

COMMON_ARGS="\
    --dataset_name trl-lib/tldr \
    --dataset_test_split validation \
    --num_ppo_epochs 4 \
    --num_mini_batches 1 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --total_episodes ${TOTAL_EPISODES} \
    --response_length ${RESPONSE_LENGTH} \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 16 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --kl_coef 0.05 \
    --eval_strategy steps \
    --eval_steps 200 \
    --report_to wandb \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_task_type CAUSAL_LM"

echo "Running seed ${SEED} on node $(hostname)"
echo "response_length=${RESPONSE_LENGTH}, total_episodes=${TOTAL_EPISODES}"

# DART on GPUs 0,1
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes 2 \
    --main_process_port 29500 \
    examples/scripts/ppo/dart.py \
    ${COMMON_ARGS} \
    --seed ${SEED} \
    --dart_enabled true \
    --dart_warmup_frac 0.4 \
    --output_dir ${OUTPUT_BASE}/dart-seed-${SEED} \
    --run_name dart-tldr-seed-${SEED} &

# PPO baseline on GPUs 2,3
CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes 2 \
    --main_process_port 29501 \
    examples/scripts/ppo/dart.py \
    ${COMMON_ARGS} \
    --seed ${SEED} \
    --dart_enabled false \
    --output_dir ${OUTPUT_BASE}/ppo-seed-${SEED} \
    --run_name ppo-tldr-seed-${SEED} &

wait
echo "Seed ${SEED} finished."
