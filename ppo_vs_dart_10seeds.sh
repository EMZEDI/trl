#!/bin/bash
#SBATCH --job-name=dart_vs_ppo_10seeds
#SBATCH --nodes=10                # 20 runs total, 2 per node (4 GPUs → 2+2)
#SBATCH --ntasks-per-node=1       # 1 srun task per node; we split GPUs inside
#SBATCH --cpus-per-task=12        # 6 CPUs per experiment × 2 experiments per node
#SBATCH --gres=gpu:h100:4         # 4 GPUs per node
#SBATCH --mem=0
#SBATCH --account=aip-rrabba
#SBATCH --time=7:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source .env

# ──────────────────────────────────────────────────────────────────────────────
# NCCL config
# ──────────────────────────────────────────────────────────────────────────────
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=3600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# ──────────────────────────────────────────────────────────────────────────────
# Experiment knobs
# ──────────────────────────────────────────────────────────────────────────────
SEEDS=(42 123 256 512 1024 2048 4096 8192 16384 32768)
RESPONSE_LENGTH=256       # 2.4× longer horizon → harder credit assignment
TOTAL_EPISODES=50000
OUTPUT_BASE="/scratch/s/shahradm"

# ──────────────────────────────────────────────────────────────────────────────
# Resolve node list
# ──────────────────────────────────────────────────────────────────────────────
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
echo "Allocated ${#nodes_array[@]} nodes: ${nodes_array[*]}"

# ──────────────────────────────────────────────────────────────────────────────
# Common arguments shared by every run
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# Layout:
#   Nodes 0-4 → DART   seeds 0-9  (2 seeds per node, GPUs 0,1 and 2,3)
#   Nodes 5-9 → PPO    seeds 0-9  (2 seeds per node, GPUs 0,1 and 2,3)
# ──────────────────────────────────────────────────────────────────────────────

BASE_PORT=29500

# ── DART runs (nodes 0-4) ────────────────────────────────────────────────────
for node_idx in $(seq 0 4); do
    node=${nodes_array[$node_idx]}
    seed_a=${SEEDS[$((node_idx * 2))]}
    seed_b=${SEEDS[$((node_idx * 2 + 1))]}
    port_a=$((BASE_PORT + node_idx * 2))
    port_b=$((BASE_PORT + node_idx * 2 + 1))

    echo "[DART] Node $node: seed_a=$seed_a (port $port_a), seed_b=$seed_b (port $port_b)"

    srun --nodes=1 --nodelist="$node" --gres=gpu:4 bash -c "
        source .env
        export NCCL_IB_DISABLE=1
        export NCCL_P2P_DISABLE=1
        export NCCL_TIMEOUT=3600
        export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
        export TORCH_NCCL_BLOCKING_WAIT=1

        # ── Seed A on GPUs 0,1 ──
        CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
            --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
            --num_processes 2 \
            --main_process_port ${port_a} \
            examples/scripts/ppo/dart.py \
            ${COMMON_ARGS} \
            --seed ${seed_a} \
            --dart_enabled true \
            --dart_warmup_frac 0.4 \
            --output_dir ${OUTPUT_BASE}/dart-seed-${seed_a} \
            --run_name dart-tldr-seed-${seed_a} &

        # ── Seed B on GPUs 2,3 ──
        CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
            --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
            --num_processes 2 \
            --main_process_port ${port_b} \
            examples/scripts/ppo/dart.py \
            ${COMMON_ARGS} \
            --seed ${seed_b} \
            --dart_enabled true \
            --dart_warmup_frac 0.4 \
            --output_dir ${OUTPUT_BASE}/dart-seed-${seed_b} \
            --run_name dart-tldr-seed-${seed_b} &

        wait
    " &
done

# ── PPO baseline runs (nodes 5-9) ────────────────────────────────────────────
for node_idx in $(seq 5 9); do
    node=${nodes_array[$node_idx]}
    local_idx=$((node_idx - 5))
    seed_a=${SEEDS[$((local_idx * 2))]}
    seed_b=${SEEDS[$((local_idx * 2 + 1))]}
    port_a=$((BASE_PORT + node_idx * 2))
    port_b=$((BASE_PORT + node_idx * 2 + 1))

    echo "[PPO]  Node $node: seed_a=$seed_a (port $port_a), seed_b=$seed_b (port $port_b)"

    srun --nodes=1 --nodelist="$node" --gres=gpu:4 bash -c "
        source .env
        export NCCL_IB_DISABLE=1
        export NCCL_P2P_DISABLE=1
        export NCCL_TIMEOUT=3600
        export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
        export TORCH_NCCL_BLOCKING_WAIT=1

        # ── Seed A on GPUs 0,1 ──
        CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
            --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
            --num_processes 2 \
            --main_process_port ${port_a} \
            examples/scripts/ppo/dart.py \
            ${COMMON_ARGS} \
            --seed ${seed_a} \
            --dart_enabled false \
            --output_dir ${OUTPUT_BASE}/ppo-seed-${seed_a} \
            --run_name ppo-tldr-seed-${seed_a} &

        # ── Seed B on GPUs 2,3 ──
        CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
            --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
            --num_processes 2 \
            --main_process_port ${port_b} \
            examples/scripts/ppo/dart.py \
            ${COMMON_ARGS} \
            --seed ${seed_b} \
            --dart_enabled false \
            --output_dir ${OUTPUT_BASE}/ppo-seed-${seed_b} \
            --run_name ppo-tldr-seed-${seed_b} &

        wait
    " &
done

# ──────────────────────────────────────────────────────────────────────────────
# Wait for all 10 srun background jobs (one per node) to finish
# ──────────────────────────────────────────────────────────────────────────────
echo "All 20 runs launched (10 DART + 10 PPO) across 10 nodes. Waiting..."
wait
echo "All runs complete."
