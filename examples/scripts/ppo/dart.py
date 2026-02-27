# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0

# /// script
# dependencies = ["trl", "peft", "wandb"]
# ///

"""
DART (Dual Adaptive Residual Tracking) Training Script — GSM8K Edition
=======================================================================

Differences from the TL;DR version:
  - NO pretrained reward model: rewards are rule-based (answer correctness + format)
  - Value model and residual critic are initialized from the SFT/policy backbone
    with a randomly initialized scalar score head (standard for math RL)
  - Dataset: openai/gsm8k with chat-template formatted prompts
  - Policy: Qwen2.5-Math-1.5B (math-pretrained base, no instruction tuning needed)

Single GPU — DART:
python examples/scripts/ppo/dart.py \
    --dataset_name openai/gsm8k \
    --dataset_config main \
    --dataset_test_split test \
    --learning_rate 3e-6 \
    --output_dir qwen-math-dart-gsm8k \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --total_episodes 50000 \
    --model_name_or_path Qwen/Qwen2.5-Math-1.5B \
    --sft_model_path Qwen/Qwen2.5-Math-1.5B \
    --response_length 512 \
    --dart_enabled true \
    --dart_warmup_frac 0.4 \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --report_to wandb

Disable DART → standard PPO baseline:
    --dart_enabled false
"""

import os
import re

import torch
from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import (
    ModelConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.experimental.ppo import PPOConfig, PPOTrainer


# ── DeepSpeed Zero3 helper ────────────────────────────────────────────────────

def _no_zero3_init():
    """Temporarily pause DeepSpeed Zero3 Init if active.
    Models loaded inside will NOT be sharded during from_pretrained.
    """
    from contextlib import nullcontext
    try:
        state = PartialState()
        ds_plugin = getattr(state, "deepspeed_plugin", None)
        if ds_plugin is not None and getattr(ds_plugin, "is_zero3_init_enabled", lambda: False)():
            return ds_plugin.zero3_init_context_manager(enable=False)
    except Exception:
        pass
    return nullcontext()


# ── Rule-based reward functions ───────────────────────────────────────────────
# These replace the pretrained reward model used in the TL;DR version.
# PPOTrainer calls reward_fn(queries, responses) → List[float]

def extract_answer(text):
    m = re.search(r"\\boxed\{(.*?)\}", text)
    if m: return m.group(1).strip()
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m: return m.group(1).strip()
    # fallback: GSM8K native #### pattern
    m = re.search(r"####\s*([\d,.\-]+)", text)
    if m: return m.group(1).replace(",", "").strip()
    return ""

def compute_rewards(queries, responses, ground_truths):
    """
    Returns a list of scalar rewards for each (query, response) pair.
    Correctness: +2.0 if extracted answer matches ground truth
    Format:      +0.5 if response has <reasoning>...</reasoning><answer>...</answer>
    """
    rewards = []
    fmt_pat = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    for response, gt in zip(responses, ground_truths):
        r = 0.0
        if extract_answer(response) == gt.strip():
            r += 2.0
        if re.search(fmt_pat, response, re.DOTALL):
            r += 0.5
        rewards.append(r)
    return rewards


# ── Dataset preparation ───────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Solve the problem step by step. "
    "Wrap your reasoning in <reasoning>...</reasoning> "
    "and your final numeric answer in <answer>...</answer> and \\boxed{}."
)

def prepare_gsm8k(example, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": example["question"]},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ground_truth = example["answer"].split("####")[-1].strip()
    return {
        "prompt": prompt,
        "ground_truth": ground_truth,
    }

def tokenize_dataset(dataset, tokenizer, max_prompt_length, num_proc):
    def tokenize(element):
        input_ids = tokenizer(element["prompt"], padding=False)["input_ids"]
        return {"input_ids": input_ids, "lengths": len(input_ids)}

    dataset = dataset.map(tokenize, remove_columns=["prompt"], num_proc=num_proc)
    dataset = dataset.filter(lambda x: x["lengths"] <= max_prompt_length, num_proc=num_proc)
    return dataset


# ── Value model builder ───────────────────────────────────────────────────────
# For GSM8K there is no pretrained RM checkpoint.
# We initialize both value heads from the policy backbone with a random score head.
# This is standard practice for math RL (same as DeepSeekMath, RLVR papers).

def build_value_model(backbone_path, tokenizer, model_kwargs, trust_remote_code, no_zero3=False):
    ctx = _no_zero3_init() if no_zero3 else __import__("contextlib").nullcontext()
    with ctx:
        model = AutoModelForSequenceClassification.from_pretrained(
            backbone_path,
            trust_remote_code=trust_remote_code,
            num_labels=1,
            ignore_mismatched_sizes=True,   # score head is new/random — expected
            **model_kwargs,
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    # DART default on
    if not hasattr(training_args, "dart_enabled") or training_args.dart_enabled is None:
        training_args.dart_enabled = True

    # ── dtype / quantization ──────────────────────────────────────────────────
    dtype = (
        model_args.dtype
        if model_args.dtype in ["auto", None]
        else getattr(torch, model_args.dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # ── Value model (base critic) ─────────────────────────────────────────────
    # Initialized from the policy backbone; score head is random (no pretrained RM).
    print(f"Building base value model from {training_args.sft_model_path}")
    value_model = build_value_model(
        training_args.sft_model_path,
        tokenizer,
        model_kwargs,
        model_args.trust_remote_code,
        no_zero3=False,   # primary model, let ZeRO shard it normally
    )
    v_params = sum(p.numel() for p in value_model.parameters() if p.requires_grad)
    print(f"  Base value model trainable params: {v_params:,}")

    # ── DART: residual critic ─────────────────────────────────────────────────
    if training_args.dart_enabled:
        print("DART enabled — building residual critic")
        print(f"  dart_lambda_res:  {training_args.dart_lambda_res}")
        print(f"  dart_lr_scale:    {training_args.dart_lr_scale}")
        print(f"  dart_warmup_frac: {training_args.dart_warmup_frac}")
        # Load with _no_zero3_init so residual is not sharded during init;
        # PPOTrainer prepares it separately via accelerator.prepare()
        value_model_residual = build_value_model(
            training_args.sft_model_path,
            tokenizer,
            model_kwargs,
            model_args.trust_remote_code,
            no_zero3=True,
        )
        r_params = sum(p.numel() for p in value_model_residual.parameters() if p.requires_grad)
        print(f"  Residual critic trainable params: {r_params:,}")
        print(f"  DART total critic params: {v_params + r_params:,}")
    else:
        print("DART disabled — running standard PPO baseline")
        value_model_residual = None

    # ── Reward model ──────────────────────────────────────────────────────────
    # For GSM8K: rule-based rewards, no neural reward model.
    # We pass reward_model=None and override the reward computation via
    # a reward_fn hook in PPOTrainer (see training_args.reward_fn below).
    # If your PPOTrainer version requires a reward_model object, pass a dummy
    # wrapper — see RuleBasedRewardModel below.
    reward_model = None   # <-- set to RuleBasedRewardModel() if trainer requires it

    # ── Policy ────────────────────────────────────────────────────────────────
    print(f"Loading policy from {training_args.sft_model_path}")
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    # Resize embeddings if pad token was freshly added
    policy.resize_token_embeddings(len(tokenizer))
    value_model.resize_token_embeddings(len(tokenizer))
    if value_model_residual is not None:
        value_model_residual.resize_token_embeddings(len(tokenizer))

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        with _no_zero3_init():
            ref_policy = AutoModelForCausalLM.from_pretrained(
                training_args.sft_model_path,
                trust_remote_code=model_args.trust_remote_code,
                **model_kwargs,
            )
        ref_policy.resize_token_embeddings(len(tokenizer))
    else:
        ref_policy = None   # PEFT: ref policy is implicit (merged adapter off)

    total_policy_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Policy trainable params: {total_policy_params:,}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    raw = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    with PartialState().local_main_process_first():
        train_dataset = raw[script_args.dataset_train_split].map(
            lambda ex: prepare_gsm8k(ex, tokenizer),
            remove_columns=raw[script_args.dataset_train_split].column_names,
            num_proc=training_args.dataset_num_proc,
        )
        eval_dataset = raw[training_args.dataset_test_split if hasattr(training_args, "dataset_test_split")
                           else script_args.dataset_test_split].map(
            lambda ex: prepare_gsm8k(ex, tokenizer),
            remove_columns=raw["test"].column_names,
            num_proc=training_args.dataset_num_proc,
        )
        # Tokenize and filter prompts that are too long
        train_dataset = tokenize_dataset(
            train_dataset, tokenizer,
            max_prompt_length=256,   # GSM8K questions are short
            num_proc=training_args.dataset_num_proc,
        )
        eval_dataset = tokenize_dataset(
            eval_dataset, tokenizer,
            max_prompt_length=256,
            num_proc=training_args.dataset_num_proc,
        )

    # ── Reward fn wrapper ─────────────────────────────────────────────────────
    # Store ground truths alongside dataset so PPOTrainer can pass them to
    # compute_rewards. The exact hook depends on your PPOTrainer implementation.
    # If your trainer exposes a reward_fn argument, pass this lambda:
    #
    #   reward_fn = lambda queries, responses, batch: compute_rewards(
    #       queries, responses, batch["ground_truth"]
    #   )
    #
    # If it requires a reward_model with a .forward(), use this shim:

    class RuleBasedRewardModel(torch.nn.Module):
        """Shim so PPOTrainer can call reward_model(input_ids, attention_mask).
        Returns scalar rewards computed by rule-based functions, not a neural net.
        Requires ground_truth to be stored on the batch — set via trainer hook.
        """
        def __init__(self):
            super().__init__()
            # Dummy parameter so accelerator.prepare() doesn't complain
            self._dummy = torch.nn.Linear(1, 1, bias=False)

        def forward(self, input_ids, attention_mask, ground_truth=None, **kwargs):
            # Decode responses
            decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            if ground_truth is None:
                # Fallback: return zeros (should not happen in practice)
                return torch.zeros(len(decoded))
            rewards = compute_rewards([""] * len(decoded), decoded, ground_truth)
            return torch.tensor(rewards, dtype=torch.float32)

    reward_model = RuleBasedRewardModel()

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = PPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        value_model_residual=value_model_residual,  # DART extension
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()
