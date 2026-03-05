#!/bin/bash
#SBATCH --job-name=dart_math
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=0
#SBATCH --account=aip-rrabba
#SBATCH --time=12:00:00
#SBATCH --output=slurm-dart-%j.out
#SBATCH --error=slurm-dart-%j.err

source .env

export NCCL_TIMEOUT=3600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEEDS=(42 123 256 512)
TOTAL_EPISODES=50000
RESPONSE_LENGTH=512
OUTPUT_BASE="$SCRATCH/math"
BASE_PORT=29500
MODEL="Qwen/Qwen2.5-Math-1.5B"
ACCEL_CFG="examples/accelerate_configs/deepspeed_zero2_4gpu.yaml"

DART_COMMON="\
    --model_name_or_path ${MODEL} \
    --sft_model_path ${MODEL} \
    --dataset_name EleutherAI/hendrycks_math \
    --dataset_train_split train \
    --dataset_test_split test \
    --num_ppo_epochs 4 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --total_episodes ${TOTAL_EPISODES} \
    --response_length ${RESPONSE_LENGTH} \
    --local_rollout_forward_batch_size 4 \
    --kl_coef 0.05 \
    --gradient_checkpointing \
    --dart_enabled true \
    --dart_warmup_frac 0.4 \
    --eval_strategy steps \
    --eval_steps 200 \
    --report_to wandb \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules all-linear \
    --lora_task_type CAUSAL_LM"

# Get list of allocated nodes
NODES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
echo "Running 4 DART seeds across ${#NODES[@]} nodes"
echo "Nodes: ${NODES[*]}"
echo "Seeds: ${SEEDS[*]}"

for i in 0 1 2 3; do
    SEED=${SEEDS[$i]}
    NODE=${NODES[$i]}
    echo "Launching seed ${SEED} on node ${NODE} (4 GPUs)"
    srun --nodes=1 --ntasks=1 --nodelist=${NODE} \
        accelerate launch \
        --config_file ${ACCEL_CFG} --num_processes 4 \
        --main_process_port $((BASE_PORT + i)) \
        examples/scripts/ppo/dart.py ${DART_COMMON} \
        --seed ${SEED} \
        --output_dir ${OUTPUT_BASE}/dart-seed-${SEED} \
        --run_name dart-math-seed-${SEED} &
done

wait
echo "All 4 DART runs complete."
