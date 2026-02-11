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
from trl.experimental.ppo import PPOConfig, PPOTrainer


def _no_zero3_init():
    """Context manager that temporarily pauses DeepSpeed Zero3 Init if active.

    Models loaded inside this context will NOT be sharded during from_pretrained.
    Use for models that will be prepared separately later (reward, ref, residual).
    Uses Accelerate's official API to toggle the zero3_init_flag.
    """
    from contextlib import nullcontext
    try:
        from accelerate import PartialState
        state = PartialState()
        ds_plugin = getattr(state, "deepspeed_plugin", None)
        if ds_plugin is not None and getattr(ds_plugin, "is_zero3_init_enabled", lambda: False)():
            return ds_plugin.zero3_init_context_manager(enable=False)
    except Exception:
        pass
    return nullcontext()

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


"""
DART (Dual Adaptive Residual Tracking) Training Script
=======================================================

This script demonstrates DART training on the TL;DR summarization task.

DART extends PPO with a *residual critic*: a lightweight second value head that
captures long-horizon value that the base critic misses.  During a configurable
warmup phase only the base critic is active (identical to standard PPO); once
warmup completes, the residual critic is activated with a higher GAE λ and its
own (typically lower) learning rate, and the combined value V = V_base + V_res
is used for advantage computation.

IMPORTANT: The reward model and value model must share the same architecture as
the policy model (same tokenizer, same backbone). The value model is loaded from
``--reward_model_path`` so that its ``score`` head is pre-trained, not random.

Recommended setup (Pythia-1B, TL;DR):
  - Policy/SFT:   cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr
  - Reward/Value:  cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr
  - Dataset:       trl-lib/tldr

Single GPU — DART:
python examples/scripts/ppo/dart.py \
    --dataset_name trl-lib/tldr \
    --dataset_test_split validation \
    --learning_rate 3e-6 \
    --output_dir pythia-1b-dart \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --total_episodes 30000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --response_length 53 \
    --dart_enabled true

Multi-GPU with DeepSpeed — DART:
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/ppo/dart.py \
    --dataset_name trl-lib/tldr \
    --dataset_test_split validation \
    --output_dir pythia-1b-dart \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --total_episodes 100000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 16 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --dart_enabled true \
    --report_to wandb

Compare with standard PPO (disable DART):
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/ppo/dart.py \
    --dataset_name trl-lib/tldr \
    --dataset_test_split validation \
    --output_dir pythia-1b-ppo-baseline \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --total_episodes 100000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 16 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --dart_enabled false \
    --report_to wandb
"""


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    
    # Enable DART by default (can be disabled via --dart_enabled false for baseline comparison)
    if not hasattr(training_args, 'dart_enabled') or training_args.dart_enabled is None:
        training_args.dart_enabled = True

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

    # ── Value model ──────────────────────────────────────────────────────────
    # Load from reward_model_path so the score head is PRETRAINED (not random).
    # This matches the official TRL ppo_tldr.py pattern.
    print(f"Loading Value Model from {training_args.reward_model_path}")
    value_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        trust_remote_code=model_args.trust_remote_code,
        num_labels=1,
        **model_kwargs,
    )
    value_model.config.pad_token_id = tokenizer.pad_token_id

    # ── DART: Residual value model ───────────────────────────────────────────
    if training_args.dart_enabled:
        print(f"DART enabled - creating residual value model from {training_args.reward_model_path}")
        print(f"  - dart_lambda_res: {training_args.dart_lambda_res}")
        print(f"  - dart_lr_scale: {training_args.dart_lr_scale}")
        print(f"  - dart_warmup_frac: {training_args.dart_warmup_frac}")
        with _no_zero3_init():
            value_model_residual = AutoModelForSequenceClassification.from_pretrained(
                training_args.reward_model_path,
                trust_remote_code=model_args.trust_remote_code,
                num_labels=1,
                **model_kwargs,
            )
        # Set pad_token_id so GPT-NeoX forward doesn't crash with batch > 1
        value_model_residual.config.pad_token_id = tokenizer.pad_token_id
    else:
        print("DART disabled - running standard PPO baseline")
        value_model_residual = None

    # ── Reward model ─────────────────────────────────────────────────────────
    print(f"Loading Reward Model from {training_args.reward_model_path}")
    with _no_zero3_init():
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            training_args.reward_model_path,
            trust_remote_code=model_args.trust_remote_code,
            num_labels=1,
            **model_kwargs,
        )
    reward_model.config.pad_token_id = tokenizer.pad_token_id

    # ── Policy model ─────────────────────────────────────────────────────────
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        with _no_zero3_init():
            ref_policy = AutoModelForCausalLM.from_pretrained(
                training_args.sft_model_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
            )
    else:
        ref_policy = None

    ################
    # Dataset
    ################
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None
    if eval_dataset is None:
        # Fallback: split off a small eval set from the end of train
        eval_samples = 100
        eval_dataset = train_dataset.select(range(len(train_dataset) - eval_samples, len(train_dataset)))
        train_dataset = train_dataset.select(range(len(train_dataset) - eval_samples))

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            input_ids = tokenizer(element["prompt"], padding=False)["input_ids"]
            return {"input_ids": input_ids, "lengths": len(input_ids)}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        if eval_dataset is not None:
            eval_dataset = prepare_dataset(eval_dataset, tokenizer)
        # Filter out prompts that are too long
        train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)
        if eval_dataset is not None:
            eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=training_args.dataset_num_proc)

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
