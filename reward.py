# reward.py — shared reward functions for DART and GRPO
import re

def extract_answer(text):
    m = re.search(r"\\boxed\{(.*?)\}", text)
    if m: return m.group(1).strip()
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m: return m.group(1).strip()
    # fallback: #### pattern from GSM8K native format
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
