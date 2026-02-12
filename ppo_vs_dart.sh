#!/bin/bash
#SBATCH --job-name=dart_vs_ppo_v2
#SBATCH --nodes=2                 # Request 2 distinct nodes
#SBATCH --ntasks-per-node=1       # 1 main task per node
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:h100:4        # 4 GPUs per node
#SBATCH --mem=0
#SBATCH --account=aip-rrabba
#SBATCH --time=3:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

source .env

# NCCL config
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=3600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

# Get the list of nodes allocated to the job
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)

node1=${nodes_array[0]}
node2=${nodes_array[1]}

echo "Node 1: $node1 (Running DART)"
echo "Node 2: $node2 (Running PPO Baseline)"

# --- Run 1: DART Version (Running on Node 1) ---
srun --nodes=1 --nodelist=$node1 --exclusive --gres=gpu:4 bash -c "
    source .env; 
    accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes 2 \
    examples/scripts/ppo/dart.py \
    --dataset_name trl-lib/tldr \
    --dataset_test_split validation \
    --output_dir /scratch/s/shahradm/pythia-1b-dart-v2 \
    --num_ppo_epochs 4 \
    --num_mini_batches 1 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --total_episodes 30000 \
    --response_length 53 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 16 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --kl_coef 0.05 \
    --dart_enabled true \
    --dart_warmup_frac 0.4 \
    --eval_strategy steps \
    --eval_steps 200 \
    --report_to wandb \
    --run_name dart-tldr-v2 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_task_type CAUSAL_LM
" & 

# --- Run 2: Standard PPO baseline (Running on Node 2) ---
srun --nodes=1 --nodelist=$node2 --exclusive --gres=gpu:4 bash -c "
    source .env;
    accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes 2 \
    examples/scripts/ppo/dart.py \
    --dataset_name trl-lib/tldr \
    --dataset_test_split validation \
    --output_dir /scratch/s/shahradm/pythia-1b-ppo-v2 \
    --num_ppo_epochs 4 \
    --num_mini_batches 1 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --total_episodes 30000 \
    --response_length 53 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 16 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --kl_coef 0.05 \
    --dart_enabled false \
    --eval_strategy steps \
    --eval_steps 200 \
    --report_to wandb \
    --run_name ppo-tldr-v2 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_task_type CAUSAL_LM
" & 

# Wait for both background jobs to finish before exiting the SLURM job
wait
