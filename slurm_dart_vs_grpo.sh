#!/bin/bash
#SBATCH --job-name=dart_vs_grpo_math
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=0
#SBATCH --account=aip-rrabba
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source .env

export NCCL_TIMEOUT=3600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

SEEDS=(42 123 256 512)
TOTAL_EPISODES=50000
RESPONSE_LENGTH=512
OUTPUT_BASE="$SCRATCH/math"
BASE_PORT=29500
MODEL="Qwen/Qwen2.5-Math-1.5B"
ACCEL_CFG_DART="examples/accelerate_configs/deepspeed_zero2.yaml"
ACCEL_CFG_GRPO="examples/accelerate_configs/single_gpu.yaml"

S0=${SEEDS[0]}
S1=${SEEDS[1]}
S2=${SEEDS[2]}
S3=${SEEDS[3]}

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
echo "Nodes: ${nodes_array[*]}"
echo "Seeds: $S0 $S1 $S2 $S3"
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
    --gradient_accumulation_steps 16 \
    --total_episodes ${TOTAL_EPISODES} \
    --response_length ${RESPONSE_LENGTH} \
    --local_rollout_forward_batch_size 4 \
    --kl_coef 0.05 \
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

GRPO_COMMON="\
    --model_name_or_path ${MODEL} \
    --dataset_name EleutherAI/hendrycks_math \
    --dataset_train_split train \
    --dataset_test_split test \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --num_generations 8 \
    --max_completion_length ${RESPONSE_LENGTH} \
    --beta 0.05 \
    --loss_type bnpo \
    --scale_rewards group \
    --eval_strategy steps \
    --eval_steps 200 \
    --report_to wandb"
# ── Node 0: DART ──────────────────────────────────────────────────────────────
srun --nodes=1 --nodelist="${nodes_array[0]}" bash -c "
    source .env
    export NCCL_TIMEOUT=3600 TORCH_NCCL_ASYNC_ERROR_HANDLING=1

    CUDA_VISIBLE_DEVICES=0 accelerate launch \
        --config_file ${ACCEL_CFG_DART} --num_processes 1 \
        --main_process_port $((BASE_PORT + 0)) \
        examples/scripts/ppo/dart.py ${DART_COMMON} \
        --seed ${S0} \
        --output_dir ${OUTPUT_BASE}/dart-seed-${S0} \
        --run_name dart-math-seed-${S0} &

    CUDA_VISIBLE_DEVICES=1 accelerate launch \
        --config_file ${ACCEL_CFG_DART} --num_processes 1 \
        --main_process_port $((BASE_PORT + 1)) \
        examples/scripts/ppo/dart.py ${DART_COMMON} \
        --seed ${S1} \
        --output_dir ${OUTPUT_BASE}/dart-seed-${S1} \
        --run_name dart-math-seed-${S1} &

    CUDA_VISIBLE_DEVICES=2 accelerate launch \
        --config_file ${ACCEL_CFG_DART} --num_processes 1 \
        --main_process_port $((BASE_PORT + 2)) \
        examples/scripts/ppo/dart.py ${DART_COMMON} \
        --seed ${S2} \
        --output_dir ${OUTPUT_BASE}/dart-seed-${S2} \
        --run_name dart-math-seed-${S2} &

    CUDA_VISIBLE_DEVICES=3 accelerate launch \
        --config_file ${ACCEL_CFG_DART} --num_processes 1 \
        --main_process_port $((BASE_PORT + 3)) \
        examples/scripts/ppo/dart.py ${DART_COMMON} \
        --seed ${S3} \
        --output_dir ${OUTPUT_BASE}/dart-seed-${S3} \
        --run_name dart-math-seed-${S3} &

    wait
" &

# ── Node 1: GRPO ──────────────────────────────────────────────────────────────
srun --nodes=1 --nodelist="${nodes_array[1]}" bash -c "
    source .env
    export NCCL_TIMEOUT=3600 TORCH_NCCL_ASYNC_ERROR_HANDLING=1

    CUDA_VISIBLE_DEVICES=0 accelerate launch \
        --config_file ${ACCEL_CFG_GRPO} --num_processes 1 \
        --main_process_port $((BASE_PORT + 0)) \
        examples/scripts/grpo/gsm8k_grpo.py ${GRPO_COMMON} \
        --seed ${S0} \
        --output_dir ${OUTPUT_BASE}/grpo-seed-${S0} \
        --run_name grpo-math-seed-${S0} &

    CUDA_VISIBLE_DEVICES=1 accelerate launch \
        --config_file ${ACCEL_CFG_GRPO} --num_processes 1 \
        --main_process_port $((BASE_PORT + 1)) \
        examples/scripts/grpo/gsm8k_grpo.py ${GRPO_COMMON} \
        --seed ${S1} \
        --output_dir ${OUTPUT_BASE}/grpo-seed-${S1} \
        --run_name grpo-math-seed-${S1} &

    CUDA_VISIBLE_DEVICES=2 accelerate launch \
        --config_file ${ACCEL_CFG_GRPO} --num_processes 1 \
        --main_process_port $((BASE_PORT + 2)) \
        examples/scripts/grpo/gsm8k_grpo.py ${GRPO_COMMON} \
        --seed ${S2} \
        --output_dir ${OUTPUT_BASE}/grpo-seed-${S2} \
        --run_name grpo-math-seed-${S2} &

    CUDA_VISIBLE_DEVICES=3 accelerate launch \
        --config_file ${ACCEL_CFG_GRPO} --num_processes 1 \
        --main_process_port $((BASE_PORT + 3)) \
        examples/scripts/grpo/gsm8k_grpo.py ${GRPO_COMMON} \
        --seed ${S3} \
        --output_dir ${OUTPUT_BASE}/grpo-seed-${S3} \
        --run_name grpo-math-seed-${S3} &

    wait
" &

wait
echo "All 8 runs complete (4 DART + 4 GRPO on MATH)."

