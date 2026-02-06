# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

import os
import shutil

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import ModelConfig, ScriptArguments, get_kbit_device_map, get_peft_config, get_quantization_config
import trl.experimental.ppo.ppo_trainer
from trl.experimental.ppo import PPOConfig, PPOTrainer
from trl.experimental.utils import first_true_indices

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


# Monkey-patch get_reward to handle mismatched tokenizers (Qwen Policy vs DeBERTa Reward)
original_get_reward = trl.experimental.ppo.ppo_trainer.get_reward

def custom_get_reward(model, query_responses, pad_token_id, context_length):
    # Check if this is the reward model requiring re-tokenization
    if getattr(model, "needs_retokenization", False):
        # query_responses contains Policy (Qwen) token IDs
        # We must:
        # 1. Decode back to text
        # 2. Re-encode using Reward (DeBERTa) tokenizer
        # 3. specific forward pass to get scalar reward
        
        # Access tokenizers attached to the model
        policy_tokenizer = model.policy_tokenizer
        reward_tokenizer = model.reward_tokenizer
        device = query_responses.device
        
        # 1. Decode
        texts = policy_tokenizer.batch_decode(query_responses, skip_special_tokens=True)
        
        # 2. Encode (Ensure we don't exceed model limits e.g. 512)
        # We generally expect the reward model to have a max_position_embeddings config
        max_len = getattr(model.config, "max_position_embeddings", 512)
        
        encoded = reward_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        
        # 3. Run Forward Pass
        # We cannot use model.score(hidden_states) based flow easily because we don't have hidden states of Qwen input.
        # We run the full model forward on new inputs.
        with torch.no_grad():
            outputs = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                return_dict=True
            )
            # Depending on model type (SeqClass), logits should be (batch, 1) or (batch, num_labels)
            # We assume it's set up as 1 label regression
            scores = outputs.logits # (batch, 1)

        # 4. Map scalar scores back to the sequence timeline for PPO
        # PPO expects `reward_logits` of shape (batch, seq_len, 1) usually,
        # where the last non-pad token holds the reward score.
        
        batch_size, seq_len = query_responses.shape
        # Create a tensor of zeros/neutral value
        dummy_logits = torch.zeros(batch_size, seq_len, 1, device=device, dtype=scores.dtype)
        
        # Find the end of the sequence in the POLICY's timeline
        # (This aligns the reward with the end of generation)
        seq_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
        
        final_rewards = scores.squeeze(-1) # (batch,)
        
        for i in range(batch_size):
            # Clamp index to be safe
            idx = min(seq_lengths[i], seq_len - 1)
            dummy_logits[i, idx, 0] = final_rewards[i]
            
        return dummy_logits, final_rewards, seq_lengths
    
    else:
        # Fallback to standard behavior for Value Model (Qwen) and Ref Model
        return original_get_reward(model, query_responses, pad_token_id, context_length)

# Apply the patch
trl.experimental.ppo.ppo_trainer.get_reward = custom_get_reward


"""
DART (Dual Adaptive Residual Tracking) Training Script
=======================================================

This script demonstrates DART training with LoRA on SFT'd models.

Basic usage (single GPU):
python examples/scripts/ppo/dart.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --output_dir pythia-1b-deduped-dart \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --missing_eos_penalty 1.0 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32

Multi-GPU with DeepSpeed:
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/ppo/dart.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --output_dir pythia-1b-deduped-dart \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path EleutherAI/pythia-1b-deduped \
    --reward_model_path EleutherAI/pythia-1b-deduped \
    --local_rollout_forward_batch_size 1 \
    --missing_eos_penalty 1.0 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32

Compare with standard PPO (disable DART):
python examples/scripts/ppo/dart.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --output_dir pythia-1b-deduped-ppo-baseline \
    --per_device_train_batch_size 64 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --dart_enabled false \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32
"""


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    # Enable DART by default (can be disabled via --dart_enabled false for baseline comparison)
    if not hasattr(training_args, 'dart_enabled') or training_args.dart_enabled is None:
        training_args.dart_enabled = True
    
    # remove output_dir if exists
    shutil.rmtree(training_args.output_dir, ignore_errors=True)

    # Helper: add a .score method for models that only expose a classifier head
    def ensure_score(model):
        if hasattr(model, "score"):
            return model
        if hasattr(model, "classifier"):
            def score(hidden_states):
                pooled = hidden_states[:, 0, :] if hidden_states.dim() == 3 else hidden_states
                logits = model.classifier(pooled)
                return logits.unsqueeze(-1) if logits.dim() == 2 else logits
            model.score = score
        return model

    def resize_if_needed(model, tokenizer_to_use=None):
        if model is None:
            return model
        if tokenizer_to_use is None:
            tokenizer_to_use = tokenizer
        try:
            emb = model.get_input_embeddings()
            if emb is not None and emb.num_embeddings != len(tokenizer_to_use):
                model.resize_token_embeddings(len(tokenizer_to_use))
        except Exception:
            pass
        return model

    ################
    # Model & Tokenizer
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    # Load Reward Tokenizer (DeBERTa) explicitly
    if training_args.reward_model_path != training_args.sft_model_path:
        reward_tokenizer = AutoTokenizer.from_pretrained(
            training_args.reward_model_path, trust_remote_code=model_args.trust_remote_code
        )
    else:
        reward_tokenizer = tokenizer

    # Base value model
    # IMPORTANT: We load Value Model from SFT/Policy path (Qwen) to ensure tokenizers match for token-level updates.
    # Using DeBERTa (Reward Model) as Value Model with a Qwen Policy causes tokenizer mismatches for the Critic.
    print(f"Loading Value Model from {training_args.sft_model_path} (to match Policy structure)")
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.sft_model_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
        **model_kwargs,
    )
    value_model = ensure_score(resize_if_needed(value_model, tokenizer))
    
    # DART: Create residual value model (only if DART is enabled)
    if training_args.dart_enabled:
        print(f"DART enabled - creating residual value model")
        print(f"  - dart_lambda_res: {training_args.dart_lambda_res}")
        print(f"  - dart_lr_scale: {training_args.dart_lr_scale}")
        print(f"  - dart_warmup_frac: {training_args.dart_warmup_frac}")
        # TODO: this currently just creates a copy of the value model, but in practice one might want to
        # start from a random initialization. Consider adding a flag to control this behavior.
        value_model_residual = AutoModelForSequenceClassification.from_pretrained(
            training_args.sft_model_path,
            trust_remote_code=model_args.trust_remote_code,
            num_labels=1,
            **model_kwargs,
        )
        value_model_residual = ensure_score(resize_if_needed(value_model_residual, tokenizer))
    else:
        print("DART disabled - running standard PPO baseline")
        value_model_residual = None
    
    # Reward model
    # This remains DeBERTa as requested
    print(f"Loading Reward Model from {training_args.reward_model_path}")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
        **model_kwargs,
    )
    # Check if we need re-tokenization (if arch differs)
    if training_args.reward_model_path != training_args.sft_model_path:
        print("Enabling re-tokenization for Reward Model (Policy != Reward)")
        reward_model.needs_retokenization = True
        reward_model.policy_tokenizer = tokenizer
        reward_model.reward_tokenizer = reward_tokenizer
        # Don't resize using Qwen tokenizer! Use reward_tokenizer if needed.
        # Usually standard RM doesn't need resize unless we added tokens.
        reward_model = resize_if_needed(reward_model, reward_tokenizer)
    else:
        reward_model.needs_retokenization = False
        reward_model = ensure_score(resize_if_needed(reward_model, tokenizer))

    reward_model = ensure_score(reward_model)
    
    # Policy model
    
    # Policy model
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    policy = resize_if_needed(policy)

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_policy = AutoModelForCausalLM.from_pretrained(
            training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
        )
        ref_policy = resize_if_needed(ref_policy)
    else:
        ref_policy = None

    ################
    # Dataset
    ################
    dataset = load_dataset(
        script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_train_split
    )
    eval_samples = 100
    train_dataset = dataset.select(range(len(dataset) - eval_samples))
    eval_dataset = dataset.select(range(len(dataset) - eval_samples, len(dataset)))
    dataset_text_field = "prompt"

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field],
                padding=False,
                truncation=True,
                max_length=512 - training_args.response_length,
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    ################
    # Training
    ################
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        value_model_residual=value_model_residual,  # DART: pass residual value model
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()
