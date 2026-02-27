# Copyright 2020-2026 The HuggingFace Team. Apache License 2.0

import os
import re
import torch
from datasets import load_dataset
from transformers import HfArgumentParser
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments


# ── Reward functions ──────────────────────────────────────────────────────────

def extract_answer(text):
    m = re.search(r"\\boxed\{(.*?)\}", text)
    if m: return m.group(1).strip()
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"####\s*([\d,.\-]+)", text)
    if m: return m.group(1).replace(",", "").strip()
    return ""

def correctness_reward(completions, ground_truth, **kwargs):
    return [2.0 if extract_answer(c) == gt.strip() else 0.0
            for c, gt in zip(completions, ground_truth)]

def format_reward(completions, **kwargs):
    pat = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    return [0.5 if re.search(pat, c, re.DOTALL) else 0.0
            for c in completions]


# ── Dataset ───────────────────────────────────────────────────────────────────

def prepare_gsm8k(example):
    answer = example["answer"].split("####")[-1].strip()
    return {
        "prompt": [
            {"role": "system", "content": (
                "Solve the problem step by step. "
                "Wrap your reasoning in <reasoning>...</reasoning> "
                "and your final numeric answer in <answer>...</answer> and \\boxed{}."
            )},
            {"role": "user", "content": example["question"]},
        ],
        "ground_truth": answer,
    }


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    # ── Dataset ───────────────────────────────────────────────────────────────
    raw = load_dataset("openai/gsm8k", "main")
    train_dataset = raw["train"].map(
        prepare_gsm8k, remove_columns=raw["train"].column_names
    )
    eval_dataset = raw["test"].map(
        prepare_gsm8k, remove_columns=raw["test"].column_names
    )

    # ── LoRA config ───────────────────────────────────────────────────────────
    # DART trains: policy LoRA r=16  +  residual critic LoRA r=16
    # → GRPO uses r=32 on all-linear to match total trainable param count
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[correctness_reward, format_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # Log trainable params for paper reporting
    total = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"[GRPO] Trainable params: {total:,}")

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name="openai/gsm8k")
