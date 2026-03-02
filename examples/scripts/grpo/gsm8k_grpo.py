# Copyright 2020-2026 The HuggingFace Team. Apache License 2.0

import re

from datasets import DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer, HfArgumentParser
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments


# ── Constants ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Solve the problem step by step. "
    "Wrap your reasoning in <reasoning>...</reasoning> "
    "and your final numeric answer in <answer>...</answer> and \\boxed{}."
)

MATH_CONFIGS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]


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


# ── Dataset helpers ───────────────────────────────────────────────────────────

def load_math_dataset(name, config=None):
    """Load dataset, handling multi-config datasets like EleutherAI/hendrycks_math."""
    if config:
        return load_dataset(name, config)
    try:
        return load_dataset(name)
    except ValueError:
        # Multi-config dataset: concatenate all MATH subsets
        splits = {}
        for cfg in MATH_CONFIGS:
            ds = load_dataset(name, cfg)
            for split in ds:
                splits.setdefault(split, []).append(ds[split])
        return DatasetDict({k: concatenate_datasets(v) for k, v in splits.items()})


def prepare_math(example, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["problem"]},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sol = example["solution"]
    m = re.search(r"\\boxed\{(.*?)\}", sol)
    ground_truth = m.group(1).strip() if m else sol.strip()
    return {"prompt": prompt, "ground_truth": ground_truth}


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    # ── Tokenizer (needed for chat template in prepare_math) ──────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # ── Dataset ───────────────────────────────────────────────────────────────
    raw = load_math_dataset(
        script_args.dataset_name,
        config=getattr(script_args, "dataset_config", None),
    )
    train_split = getattr(script_args, "dataset_train_split", "train")
    test_split = getattr(script_args, "dataset_test_split", "test")

    train_dataset = raw[train_split].map(
        lambda ex: prepare_math(ex, tokenizer),
        remove_columns=raw[train_split].column_names,
        num_proc=training_args.dataset_num_proc,
    )
    eval_dataset = raw[test_split].map(
        lambda ex: prepare_math(ex, tokenizer),
        remove_columns=raw[test_split].column_names,
        num_proc=training_args.dataset_num_proc,
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
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
