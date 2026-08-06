"""Greedy GSM8k test-set accuracy for the SFT model, via vllm.

Loads the trained HF model dir, builds chat prompts (same system+user format the
model was trained on) from data/gsm8k/test.parquet, greedy-decodes, parses the
'#### N' final answer, and reports exact-match accuracy.
"""

import argparse
import re

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

DEFAULT_MODEL = "checkpoints/gsm8k-sft/qwen05b-gsm8k-sft-instruct/global_step_348/huggingface"
TEST = "data/gsm8k/test.parquet"


def extract(text: str):
    m = re.findall(r"####\s*(-?[0-9][0-9,]*\.?[0-9]*)", text)
    if not m:
        return None
    return m[-1].replace(",", "").rstrip(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="0 = full test set")
    args = ap.parse_args()

    df = pd.read_parquet(TEST)
    if args.limit:
        df = df.iloc[: args.limit]

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = [
        tok.apply_chat_template(
            [{"role": x["role"], "content": x["content"]} for x in row["prompt"]],
            add_generation_prompt=True,
            tokenize=False,
        )
        for _, row in df.iterrows()
    ]
    golds = [str(row["reward_model"]["ground_truth"]).replace(",", "") for _, row in df.iterrows()]

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.6, max_model_len=2048)
    sp = SamplingParams(temperature=0.0, max_tokens=512, stop=["<|im_end|>"])
    outs = llm.generate(prompts, sp)

    correct = 0
    for o, gold in zip(outs, golds):
        pred = extract(o.outputs[0].text)
        if pred is not None and gold is not None:
            try:
                if abs(float(pred) - float(gold)) < 1e-6:
                    correct += 1
            except ValueError:
                if pred == gold:
                    correct += 1
    n = len(golds)
    print(f"GSM8k test accuracy: {correct}/{n} = {correct / n:.4f}")


if __name__ == "__main__":
    main()
