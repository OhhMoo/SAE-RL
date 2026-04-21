#!/usr/bin/env python3
"""
09_eval_checkpoints.py
Run greedy inference on every merged PPO checkpoint and record solve rate,
response length, and format rate.  Produces results/training_curves.csv in
the same schema expected by 11_temporal_precedence.py.

Usage:
    python scripts/09_eval_checkpoints.py
    python scripts/09_eval_checkpoints.py --n_samples 500 --batch_size 16
"""

import argparse
import csv
import os
import re
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are a math problem solver. Think step by step. "
    "You MUST end your response with '#### <number>' where <number> is the "
    "final numerical answer (digits only, no units or markdown)."
)
INSTRUCTION_SUFFIX = "Let's think step by step and output the final answer after '####'."

# Ordered list of (ppo_step, model_path).
# Step 0 = base instruct model (no PPO). Subsequent steps are merged PPO
# checkpoints — see the top-level README for how to produce them from the
# verl FSDP checkpoints in ../ppo_run/checkpoints/flexible/global_step_*.
# Edit this list to match the checkpoints you actually have on disk.
CHECKPOINTS = [
    (0,   "Qwen/Qwen2.5-0.5B-Instruct"),
    (10,  "checkpoints/ppo_merged/step_10"),
    (50,  "checkpoints/ppo_merged/step_50"),
    (100, "checkpoints/ppo_merged/step_100"),
    (120, "checkpoints/ppo_merged/step_120"),
    (140, "checkpoints/ppo_merged/step_140"),
    (160, "checkpoints/ppo_merged/step_160"),
    (180, "checkpoints/ppo_merged/step_180"),
    (200, "checkpoints/ppo_merged/step_200"),
]


def extract_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    after = text.split("####")[-1].strip()
    # grab first token after ####
    token = after.split()[0] if after.split() else None
    if token:
        return re.sub(r"[^\d\-\.]", "", token)
    return None


def evaluate(model, tokenizer, prompts, answers, batch_size, max_new_tokens, device):
    model.eval()
    correct = format_ok = total = 0
    total_resp_len = 0

    for i in tqdm(range(0, len(prompts), batch_size), leave=False):
        bp = prompts[i : i + batch_size]
        ba = answers[i : i + batch_size]

        enc = tokenizer(
            bp, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = enc["input_ids"].shape[1]
        for j, ids in enumerate(out):
            new_ids = ids[prompt_len:]
            resp = tokenizer.decode(new_ids, skip_special_tokens=True)
            total_resp_len += len(new_ids)
            pred = extract_answer(resp)
            if pred is not None:
                format_ok += 1
            if pred is not None and pred == ba[j]:
                correct += 1
            total += 1

    return {
        "solve_rate":      correct / total if total else 0,
        "format_rate":     format_ok / total if total else 0,
        "response_length": total_resp_len / total if total else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",         default="results/training_curves.csv")
    parser.add_argument("--n_samples",      type=int, default=200)
    parser.add_argument("--batch_size",     type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--split",          default="test",
                        help="GSM8k split to evaluate on (test or train)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading GSM8k {args.split} split ({args.n_samples} samples)...")
    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    examples = list(dataset)[: args.n_samples]

    rows = []

    for ppo_step, model_path in CHECKPOINTS:
        if not Path(model_path).exists():
            print(f"\n[skip] step {ppo_step}: {model_path} not found")
            continue

        print(f"\n{'='*50}")
        print(f"Step {ppo_step:>4}  —  {model_path}")
        print(f"{'='*50}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )

        # Build prompts
        prompts, answers = [], []
        for ex in examples:
            chat = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": ex["question"] + "\n" + INSTRUCTION_SUFFIX},
            ]
            p = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
            gt = ex["answer"].split("####")[-1].strip().replace(",", "")
            prompts.append(p)
            answers.append(gt)

        metrics = evaluate(
            model, tokenizer, prompts, answers,
            args.batch_size, args.max_new_tokens, args.device,
        )

        print(f"  solve_rate={metrics['solve_rate']:.3f}  "
              f"format_rate={metrics['format_rate']:.3f}  "
              f"response_length={metrics['response_length']:.1f}")

        rows.append({
            "step":            ppo_step,
            "solve_rate":      metrics["solve_rate"],
            "format_rate":     metrics["format_rate"],
            "response_length": metrics["response_length"],
            "kl_div":          "",   # not available from inference
            "reward":          "",
        })

        del model
        torch.cuda.empty_cache()

    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["step", "solve_rate", "format_rate",
                           "response_length", "kl_div", "reward"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} rows → {args.output}")
    print("\nSummary:")
    print(f"{'step':>6}  {'solve_rate':>10}  {'resp_len':>10}")
    for r in rows:
        print(f"{r['step']:>6}  {r['solve_rate']:>10.3f}  {r['response_length']:>10.1f}")


if __name__ == "__main__":
    main()
