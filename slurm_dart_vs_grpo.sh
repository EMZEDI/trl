#!/bin/bash
#SBATCH --job-name=dart_vs_grpo_gsm8k
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48          # all 48 cores per node (4 jobs × ~12 each)
#SBATCH --gpus-per-node=h100:4      # TamIA requires ALL 4 GPUs per node
#SBATCH --mem=0                     # all memory
#SBATCH --account=aip-rrabba
#SBATCH --time=24:00:00             # TamIA max is 24h
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source .env

# TamIA uses InfiniBand NDR — P2P and IB are available, but keep NCCL safe
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_TIMEOUT=3600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# ── IMPORTANT: no internet on compute nodes ───────────────────────────────────
# Pre-download before submitting:
#   python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
#     AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Math-1.5B'); \
#     AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-1.5B')"
#   python -c "from datasets import load_dataset; load_dataset('openai/gsm8k','main')"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ── Experiment config ─────────────────────────────────────────────────────────
SEEDS=(42 123 256 512 1024 2048 4096 8192 16384 32768)
TOTAL_EPISODES=50000
RESPONSE_LENGTH=512
OUTPUT_BASE="/scratch/s/shahradm/gsm8k"
BASE_PORT=29500
MODEL="Qwen/Qwen2.5-Math-1.5B"

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
echo "Allocated ${#nodes_array[@]} nodes: ${nodes_array[*]}"

# ── DART common args ──────────────────────────────────────────────────────────
DART_COMMON="\
    --model_name_or_path ${MODEL} \
    --sft_model_path ${MODEL} \
    --dataset_name lighteval/MATH \
    --dataset_config all \
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

# ── GRPO common args ──────────────────────────────────────────────────────────
# lora_r=32 set inside gsm8k_grpo.py to match DART total params
GRPO_COMMON="\
    --model_name_or_path ${MODEL} \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --num_generations 8 \
    --max_completion_length ${RESPONSE_LENGTH} \
    --beta 0.05 \
    --loss_type bnpo \
    --scale_rewards true \
    --total_episodes ${TOTAL_EPISODES} \
    --eval_strategy steps \
    --eval_steps 200 \
    --report_to wandb"

# ── Accelerate config: single-process (1 GPU per run) ────────────────────────
# TamIA nodes are whole-node allocated; we manually assign 1 GPU per run.
# No multi-GPU per run needed for Qwen2.5-Math-1.5B with LoRA.
ACCEL_CFG="examples/accelerate_configs/deepspeed_zero2.yaml"

# ──────────────────────────────────────────────────────────────────────────────
# Node 0: DART seeds 0,1,2,3  (all 4 GPUs busy)
# ──────────────────────────────────────────────────────────────────────────────
srun --nodes=1 --nodelist="${nodes_array[0]}" bash -c "
    source .env
    export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
    export NCCL_TIMEOUT=3600 TORCH_NCCL_ASYNC_ERROR_HANDLING=1

    CUDA_VISIBLE_DEVICES=0 accelerate launch \
        --config_file ${ACCEL_CFG} --num_processes 1 \
        --main_process_port $((BASE_PORT+0)) \
        examples/scripts/ppo/dart.py ${DART_COMMON} \
        --seed ${SEEDS[0]} \
        --output_dir ${OUTPUT_BASE}/dart-seed-${SEEDS[0]} \
        --run_name dart-gsm8k-seed-${SEEDS[0]} &

    CUDA_VISIBLE_DEVICES=1 accelerate launch \
        --config_file ${ACCEL_CFG} --num_processes 1 \
        --main_process_port $((BASE_PORT+1)) \
        examples/scripts/ppo/dart.py ${DART_COMMON} \
        --seed ${SEEDS[1]} \
        --output_dir ${OUTPUT_BASE}/dart-seed-${SEEDS[1]} \
        --run_name dart-gsm8k-seed-${SEEDS[1]} &

    CUDA_VISIBLE_DEVICES=2 accelerate launch \
        --config_file ${ACCEL_CFG} --num_processes 1 \
        --main_process_port $((BASE_PORT+2)) \
        examples/scripts/ppo/dart.py ${DART_COMMON} \
        --seed ${SEEDS[2]} \
        --output_dir ${OUTPUT_BASE}/dart-seed-${SEEDS[2]} \
        --run_name dart-gsm8k-seed-${SEEDS[2]} &

    CUDA_VISIBLE_DEVICES=3 accelerate launch \
        --config_file ${ACCEL_CFG} --num_processes 1 \
        --main_process_port $((BASE_PORT+3)) \
        examples/scripts/ppo/dart.py ${DART_COMMON} \
        --seed ${SEEDS[3]} \
        --output_dir ${OUTPUT_BASE}/dart-seed-${SEEDS[3]} \
        --run_name dart-gsm8k-seed-${SEEDS[3]} &

    wait
" &

# ──────────────────────────────────────────────────────────────────────────────
# Node 1: DART seeds 4,5,6,7  (all 4 GPUs busy)
# ──────────────────────────────────────────────────────────────────────────────
srun --nodes=1 --nodelist="${nodes_array[1]}" bash -c "
    source .env
    export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
    export NCCL_TIMEOUT=3600 TORCH_NCCL_ASYNC_ERROR_HANDLING=1

    CUDA_VISIBLE_DEVICES=0 accelerate launch \
        --config_file ${ACCEL_CFG} --num_processes 1 \
        --main_process_port $((BASE_PORT+10)) \
        examples/scripts/ppo/dart.py ${DART_COMMON} \
        --seed ${SEEDS[4]} \
        --output_dir ${OUTPUT_BASE}/dart-seed-${SEEDS[4]} \
        --run_name dart-gsm8k-seed-${SEEDS[4]} &

    CUDA_VISIBLE_DEVICES=1 accelerate launch \
        --config_file ${ACCEL_CFG} --num_processes 1 \
        --main_process_port $((BASE_PORT+11)) \
        examples/scripts/ppo/dart.py ${DART_COMMON} \
        --seed ${SEEDS[5]} \
        --output_dir ${OUTPUT_BASE}/dart-seed-${SEEDS[5]} \
        --run_name dart-gsm8k-seed-${SEEDS[5]} &

    CUDA_VISIBLE_DEVICES=2 accelerate launch \
        --config_file ${ACCEL_CFG} --num_processes 1 \
        --main_process_port $((BASE_PORT+12)) \
        examples/scripts/ppo/dart.py ${DART_COMMON} \
        --seed ${SEEDS[6]} \
        --output_dir ${OUTPUT_BASE}/dart-seed-${SEEDS[6]} \
        --run_name dart-gsm8k-seed-${SEEDS[6]} &

    CUDA_VISIBLE_DEVICES=3 accelerate launch \
        --config_file ${ACCEL_CFG} --num_processes 1 \
        --main_process_port $((BASE_PORT+13)) \
        examples/scripts/ppo/dart.py ${DART_COMMON} \
        --seed ${SEEDS[7]} \
        --output_dir ${OUTPUT_BASE}/dart-seed-${SEEDS[7]} \
        --run_name dart-gsm8k-seed-${SEEDS[7]} &

    wait
" &

# ──────────────────────────────────────────────────────────────────────────────
# Node 2: DART seeds 8,9 on GPUs 0,1 + GRPO seeds 8,9 on GPUs 2,3
# All 4 GPUs busy — mixed node satisfies TamIA policy
# ──────────────────────────────────────────────────────────────────────────────
srun --nodes=1 --nodelist="${nodes_array[2]}" bash -c "
    source .env
    export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
    export NCCL_TIMEOUT=3600 TORCH_NCCL_ASYNC_ERROR_HANDLING=1

    # DART seeds 8,9
    CUDA_VISIBLE_DEVICES=0 accelerate launch \
        --config_file ${ACCEL_CFG} --num_processes 1 \
        --main_process_port $((BASE_PORT+20)) \
        examples/scripts/ppo/dart.py ${DART_COMMON} \
        --seed ${SEEDS[8]} \
        --output_dir ${OUTPUT_BASE}/dart-seed-${SEEDS[8]} \
        --run_name dart-gsm8k-seed-${SEEDS[8]} &

    CUDA_VISIBLE_DEVICES=1 accelerate launch \
        --config_file ${ACCEL_CFG} --num_processes 1 \
        --main_process_port $((BASE_PORT+21)) \
        examples/scripts/ppo/dart.py ${DART_COMMON} \
        --seed ${SEEDS[9]} \
        --output_dir ${OUTPUT_BASE}/dart-seed-${SEEDS[9]} \
        --run_name dart-gsm8k-seed-${SEEDS[9]} &

    # GRPO seeds 8,9
    CUDA_VISIBLE_DEVICES=2 accelerate launch \
        --config_file ${ACCEL_CFG} --num_processes 1 \
        --main_process_port $((BASE_PORT+22)) \
        examples/scripts/grpo/gsm8k_grpo.py ${GRPO_COMMON} \
        --seed ${SEEDS[8]} \
        --output_dir ${OUTPUT_BASE}/grpo-seed-${SEEDS[8]} \
        --run_name grpo-gsm8k-seed-${SEEDS[8]} &

    CUDA_VISIBLE_DEVICES=3 accelerate launch \
        --config_file ${ACCEL_CFG} --num_processes 1 \
        --main_process_port $((BASE_PORT+23)) \
        examples/scripts/grpo/gsm8k_grpo.py ${GRPO_COMMON} \
        --seed ${SEEDS[9]} \
        --output_dir ${OUTPUT_BASE}/grpo-seed-${SEEDS[9]} \
        --run_name grpo-gsm8k-seed-${SEEDS[9]} &

    wait
" &

# ──────────────────────────────────────────────────────────────────────────────
# Node 3: GRPO seeds 0,1,2,3  (all 4 GPUs busy)
# ──────────────────────────────────────────────────────────────────────────────
srun --nodes=1 --nodelist="${nodes_array[3]}" bash -c "
    source .env
    export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
    export NCCL_TIMEOUT=3600 TORCH_NCCL_ASYNC_ERROR_HANDLING=1

    CUDA_VISIBLE_DEVICES=0 accelerate launch \
        --config_file ${ACCEL_CFG} --num_processes 1 \
        --main_process_port $((BASE_PORT+30)) \
        examples/scripts/grpo/gsm8k_grpo.py ${GRPO_COMMON} \
        --seed ${SEEDS[0]} \
        --output_dir ${OUTPUT_BASE}/grpo-seed-${SEEDS[0]} \
        --run_name grpo-gsm8k-seed-${SEEDS[0]} &

    CUDA_VISIBLE_DEVICES=1 accelerate launch \
        --config_file ${ACCEL_CFG} --num_processes 1 \
        --main_process_port $((BASE_PORT+31)) \
        examples/scripts/grpo/gsm8k_grpo.py ${GRPO_COMMON} \
        --seed ${SEEDS[1]} \
        --output_dir ${OUTPUT_BASE}/grpo-seed-${SEEDS[1]} \
        --run_name grpo-gsm8k-seed-${SEEDS[1]} &

    CUDA_VISIBLE_DEVICES=2 accelerate launch \
        --config_file ${ACCEL_CFG} --num_processes 1 \
        --main_process_port $((BASE_PORT+32)) \
        examples/scripts/grpo/gsm8k_grpo.py ${GRPO_COMMON} \
        --seed ${SEEDS[2]} \
        --output_dir ${OUTPUT_BASE}/grpo-seed-${SEEDS[2]} \
        --run_name grpo-gsm8k-seed-${SEEDS[2]} &

    CUDA_VISIBLE_DEVICES=3 accelerate launch \
        --config_file ${ACCEL_CFG} --num_processes 1 \
        --main_process_port $((BASE_PORT+33)) \
        examples/scripts/grpo/gsm8k_grpo.py ${GRPO_COMMON} \
        --seed ${SEEDS[3]} \
        --output_dir ${OUTPUT_BASE}/grpo-seed-${SEEDS[3]} \
        --run_name grpo-gsm8k-seed-${SEEDS[3]} &

    wait
" &

# ──────────────────────────────────────────────────────────────────────────────
# Node 4: GRPO seeds 4,5,6,7  (all 4 GPUs busy)
# ──────────────────────────────────────────────────────────────────────────────
srun --nodes=1 --nodelist="${nodes_array[4]}" bash -c "
    source .env
    export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
    export NCCL_TIMEOUT=3600 TORCH_NCCL_ASYNC_ERROR_HANDLING=1

    CUDA_VISIBLE_DEVICES=0 accelerate launch \
        --config_file ${ACCEL_CFG} --num_processes 1 \
        --main_process_port $((BASE_PORT+40)) \
        examples/scripts/grpo/gsm8k_grpo.py ${GRPO_COMMON} \
        --seed ${SEEDS[4]} \
        --output_dir ${OUTPUT_BASE}/grpo-seed-${SEEDS[4]} \
        --run_name grpo-gsm8k-seed-${SEEDS[4]} &

    CUDA_VISIBLE_DEVICES=1 accelerate launch \
        --config_file ${ACCEL_CFG} --num_processes 1 \
        --main_process_port $((BASE_PORT+41)) \
        examples/scripts/grpo/gsm8k_grpo.py ${GRPO_COMMON} \
        --seed ${SEEDS[5]} \
        --output_dir ${OUTPUT_BASE}/grpo-seed-${SEEDS[5]} \
        --run_name grpo-gsm8k-seed-${SEEDS[5]} &

    CUDA_VISIBLE_DEVICES=2 accelerate launch \
        --config_file ${ACCEL_CFG} --num_processes 1 \
        --main_process_port $((BASE_PORT+42)) \
        examples/scripts/grpo/gsm8k_grpo.py ${GRPO_COMMON} \
        --seed ${SEEDS[6]} \
        --output_dir ${OUTPUT_BASE}/grpo-seed-${SEEDS[6]} \
        --run_name grpo-gsm8k-seed-${SEEDS[6]} &

    CUDA_VISIBLE_DEVICES=3 accelerate launch \
        --config_file ${ACCEL_CFG} --num_processes 1 \
        --main_process_port $((BASE_PORT+43)) \
        examples/scripts/grpo/gsm8k_grpo.py ${GRPO_COMMON} \
        --seed ${SEEDS[7]} \
        --output_dir ${OUTPUT_BASE}/grpo-seed-${SEEDS[7]} \
        --run_name grpo-gsm8k-seed-${SEEDS[7]} &

    wait
" &

wait
echo "All 20 runs complete (10 DART + 10 GRPO on GSM8K)."
