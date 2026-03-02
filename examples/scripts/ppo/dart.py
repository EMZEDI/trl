# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0

# /// script
# dependencies = ["trl", "peft", "wandb"]
# ///

"""
DART (Dual Adaptive Residual Tracking) Training Script — GSM8K Edition
=======================================================================

Key differences from TL;DR dart.py:
  - Rule-based rewards (correctness + format), NO pretrained reward model
  - Value heads initialized from policy backbone with random score head
  - RuleBasedRewardModel shim makes get_reward() work with decoded text
  - ground_truth column preserved through collation via _PassthroughCollator
    (single addition to PPOTrainer.__init__ — see ppo_trainer.py)
  - Dataset: openai/gsm8k, chat-templated prompts
  - Model: Qwen/Qwen2.5-Math-1.5B

Single GPU:
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

Standard PPO baseline (disable DART):
    --dart_enabled false
"""

import os
import re

import torch
import torch.nn as nn
from accelerate import PartialState
from datasets import DatasetDict, concatenate_datasets, load_dataset
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

def extract_answer(text):
    m = re.search(r"\\boxed\{(.*?)\}", text)
    if m: return m.group(1).strip()
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"####\s*([\d,.\-]+)", text)
    if m: return m.group(1).replace(",", "").strip()
    return ""

def compute_rewards(responses_text, ground_truths):
    fmt_pat = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    rewards = []
    for response, gt in zip(responses_text, ground_truths):
        r = 0.0
        if extract_answer(response) == gt.strip():
            r += 2.0
        if re.search(fmt_pat, response, re.DOTALL):
            r += 0.5
        rewards.append(r)
    return rewards


# ── Multi-config dataset loader ───────────────────────────────────────────────

MATH_CONFIGS = [
    "algebra", "counting_and_probability", "geometry",
    "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
]

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


# ── RuleBasedRewardModel ──────────────────────────────────────────────────────
# Shim that makes get_reward() work without a neural reward model.
# PPOTrainer calls: get_reward(reward_model, postprocessed_query_response,
#                              pad_token_id, context_length)
# get_reward internally calls reward_model(input_ids, attention_mask)
# and reads .logits from the output.
# We intercept by storing the tokenizer + current batch ground_truths,
# decoding input_ids on the fly, and returning rule-based scores as .logits.

class RuleBasedRewardModel(nn.Module):
    _is_rule_based = True  # flag so PPOTrainer skips get_reward() decomposition

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self._current_ground_truths = None   # set before each get_reward call
        # Dummy param so accelerator.prepare() / deepspeed.initialize() don't crash
        self._dummy = nn.Linear(1, 1, bias=False)

    def set_ground_truths(self, ground_truths):
        """Call this before each rollout batch with the current ground truths."""
        self._current_ground_truths = ground_truths

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        decoded = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        if self._current_ground_truths is None:
            scores = torch.zeros(len(decoded), device=input_ids.device)
        else:
            raw = compute_rewards(decoded, self._current_ground_truths)
            scores = torch.tensor(raw, dtype=torch.float32, device=input_ids.device)

        # get_reward expects output.logits of shape (B, seq_len, 1) or (B, 1)
        # It reads the score at the last non-pad position — return (B, 1) scalar logits
        from transformers.utils import ModelOutput
        return ModelOutput(logits=scores.unsqueeze(-1))


# ── Dataset ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "Solve the problem step by step. "
    "Wrap your reasoning in <reasoning>...</reasoning> "
    "and your final numeric answer in <answer>...</answer> and \\boxed{}."
)

def prepare_math(example, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": example["problem"]},   # MATH uses "problem" not "question"
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # MATH answers are already in \boxed{} form in the "solution" field
    # Extract just the final boxed answer for reward comparison
    sol = example["solution"]
    m = re.search(r"\\boxed\{(.*?)\}", sol)
    ground_truth = m.group(1).strip() if m else sol.strip()
    return {"prompt": prompt, "ground_truth": ground_truth}


def build_tokenized_dataset(raw_split, tokenizer, max_prompt_length, num_proc):
    def tokenize(ex):
        ids = tokenizer(ex["prompt"], padding=False)["input_ids"]
        return {"input_ids": ids, "lengths": len(ids), "ground_truth": ex["ground_truth"]}

    ds = raw_split.map(
        tokenize,
        remove_columns=[c for c in raw_split.column_names if c not in ("ground_truth",)],
        num_proc=num_proc,
    )
    # Remove "prompt" explicitly if it survived (tokenize keeps ground_truth)
    if "prompt" in ds.column_names:
        ds = ds.remove_columns(["prompt"])
    ds = ds.filter(lambda x: x["lengths"] <= max_prompt_length, num_proc=num_proc)
    return ds


# ── Value model builder ───────────────────────────────────────────────────────

def build_value_model(backbone_path, tokenizer, model_kwargs, trust_remote_code, no_zero3=False):
    ctx = _no_zero3_init() if no_zero3 else __import__("contextlib").nullcontext()
    with ctx:
        model = AutoModelForSequenceClassification.from_pretrained(
            backbone_path,
            trust_remote_code=trust_remote_code,
            num_labels=1,
            ignore_mismatched_sizes=True,  # score head is randomly initialized — expected
            **model_kwargs,
        )
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

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

    # ── Reward model (rule-based shim) ────────────────────────────────────────
    # No pretrained checkpoint needed. Scores come from extract_answer() correctness.
    reward_model = RuleBasedRewardModel(tokenizer)
    print("[dart.py] Using rule-based reward model (no neural RM checkpoint needed)")

    # ── Value model (base critic) ─────────────────────────────────────────────
    print(f"Building base value model from {training_args.sft_model_path}")
    value_model = build_value_model(
        training_args.sft_model_path, tokenizer, model_kwargs,
        model_args.trust_remote_code, no_zero3=False,
    )
    v_params = sum(p.numel() for p in value_model.parameters() if p.requires_grad)
    print(f"  Base value model trainable params: {v_params:,}")

    # ── DART: residual critic ─────────────────────────────────────────────────
    if training_args.dart_enabled:
        print("DART enabled — building residual critic")
        value_model_residual = build_value_model(
            training_args.sft_model_path, tokenizer, model_kwargs,
            model_args.trust_remote_code, no_zero3=True,
        )
        r_params = sum(p.numel() for p in value_model_residual.parameters() if p.requires_grad)
        print(f"  Residual critic trainable params: {r_params:,}")
        print(f"  DART total critic params:         {v_params + r_params:,}")
    else:
        print("DART disabled — running standard PPO baseline")
        value_model_residual = None

    # ── Policy ────────────────────────────────────────────────────────────────
    print(f"Loading policy from {training_args.sft_model_path}")
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    # Resize all models if pad token was freshly added
    new_vocab = len(tokenizer)
    for m in [policy, value_model, value_model_residual]:
        if m is not None:
            m.resize_token_embeddings(new_vocab)

    p_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Policy trainable params: {p_params:,}")

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        with _no_zero3_init():
            ref_policy = AutoModelForCausalLM.from_pretrained(
                training_args.sft_model_path,
                trust_remote_code=model_args.trust_remote_code,
                **model_kwargs,
            )
        ref_policy.resize_token_embeddings(new_vocab)
    else:
        ref_policy = None  # PEFT: ref is implicit via disabled adapter

    # ── Dataset ───────────────────────────────────────────────────────────────
    raw = load_math_dataset(
        script_args.dataset_name,
        config=getattr(script_args, "dataset_config", None),
    )

    with PartialState().local_main_process_first():
        train_dataset = raw[script_args.dataset_train_split].map(
            lambda ex: prepare_math(ex, tokenizer),
            remove_columns=raw[script_args.dataset_train_split].column_names,
            num_proc=training_args.dataset_num_proc,
        )
        eval_split = getattr(script_args, "dataset_test_split", "test")
        eval_dataset = raw[eval_split].map(
            lambda ex: prepare_math(ex, tokenizer),
            remove_columns=raw[eval_split].column_names,
            num_proc=training_args.dataset_num_proc,
        )
        train_dataset = build_tokenized_dataset(
            train_dataset, tokenizer,
            max_prompt_length=512,
            num_proc=training_args.dataset_num_proc,
        )
        eval_dataset = build_tokenized_dataset(
            eval_dataset, tokenizer,
            max_prompt_length=512,
            num_proc=training_args.dataset_num_proc,
        )

    # ── Ground truth hook ─────────────────────────────────────────────────────
    # PPOTrainer will now have ground_truth in each batch dict (via
    # _PassthroughCollator added to ppo_trainer.py). We subclass PPOTrainer
    # to inject ground_truths into the reward model before each get_reward call.

    class GSM8KPPOTrainer(PPOTrainer):
        \"\"\"Thin subclass — ground_truth injection into the RuleBasedRewardModel
        is now handled directly in ppo_trainer.py's rule-based scoring path.
        This subclass is kept for any future MATH/GSM8K-specific overrides.\"\"\"
        pass

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = GSM8KPPOTrainer(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        value_model_residual=value_model_residual,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    trainer.generate_completions()
