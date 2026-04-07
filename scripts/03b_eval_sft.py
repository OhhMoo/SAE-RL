"""
Step 3b: Evaluate SFT model on GSM8k before starting PPO.

Checks two things:
  1. Format compliance: what fraction of outputs contain '#### <number>'
  2. Accuracy: what fraction of those answers are correct

Exits with code 1 if either metric is below the threshold, so this can be
used as a gate in a shell pipeline before launching PPO.

Usage:
    python scripts/03b_eval_sft.py \
        --model_path checkpoints/sft_merged \
        --data_path data/gsm8k/test.parquet \
        --n_samples 200 \
        --min_format_rate 0.80 \
        --min_accuracy 0.25
"""

import argparse
import re
import sys

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_answer(text):
    """Return the number after #### if present, else None."""
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if m is None:
        return None
    return m.group(1).replace(",", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", default="data/gsm8k/test.parquet")
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--min_format_rate", type=float, default=0.80,
                        help="Abort if format compliance is below this")
    parser.add_argument("--min_accuracy", type=float, default=0.25,
                        help="Abort if accuracy is below this")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    df = pd.read_parquet(args.data_path)
    df = df.head(args.n_samples)
    print(f"Evaluating on {len(df)} examples from {args.data_path}")

    n_format_ok = 0
    n_correct = 0

    for i in tqdm(range(0, len(df), args.batch_size), desc="Evaluating"):
        batch = df.iloc[i : i + args.batch_size]

        # Build prompts using the chat template (same as PPO will see)
        prompts = []
        for _, row in batch.iterrows():
            text = tokenizer.apply_chat_template(
                row["prompt"], tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=600,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs.input_ids.shape[1]
        for j, (_, row) in enumerate(batch.iterrows()):
            new_tokens = outputs[j][input_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            ground_truth = row["reward_model"]["ground_truth"]

            predicted = extract_answer(response)
            if predicted is not None:
                n_format_ok += 1
                if predicted == ground_truth:
                    n_correct += 1

    n = len(df)
    format_rate = n_format_ok / n
    accuracy = n_correct / n

    print()
    print("=" * 50)
    print(f"Samples evaluated : {n}")
    print(f"Format compliance : {n_format_ok}/{n} = {format_rate:.1%}")
    print(f"Accuracy          : {n_correct}/{n} = {accuracy:.1%}")
    print(f"Acc (fmt-only)    : {n_correct}/{n_format_ok} = {n_correct/max(n_format_ok,1):.1%}")
    print("=" * 50)

    failed = []
    if format_rate < args.min_format_rate:
        failed.append(
            f"Format compliance {format_rate:.1%} < required {args.min_format_rate:.0%}"
        )
    if accuracy < args.min_accuracy:
        failed.append(
            f"Accuracy {accuracy:.1%} < required {args.min_accuracy:.0%}"
        )

    if failed:
        print()
        print("GATE FAILED — do not proceed to PPO:")
        for msg in failed:
            print(f"  ✗ {msg}")
        sys.exit(1)
    else:
        print()
        print("GATE PASSED — SFT model is ready for PPO.")
        sys.exit(0)


if __name__ == "__main__":
    main()
