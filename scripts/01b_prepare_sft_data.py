"""
Step 1b: GSM8K parquet for verl SFT (MultiTurnSFTDataset).

Produces train.parquet / test.parquet with a ``messages`` column: user turn +
assistant turn (full chain-of-thought + #### answer). This is NOT the same as
01_prepare_data.py, which builds RL/PPO parquet with ``prompt`` and reward_model.

Usage:
    python scripts/01b_prepare_sft_data.py --save_dir data/gsm8k_sft
"""

import argparse
import os

import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="data/gsm8k_sft")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = datasets.load_dataset("openai/gsm8k", "main")
    instruction = 'Let\'s think step by step and output the final answer after "####".'
    system_prompt = (
        "You are a math problem solver. Think step by step. "
        "You MUST end your response with '#### <number>' where <number> is "
        "the final numerical answer (digits only, no units or markdown)."
    )

    def process_fn(example, _idx):
        question_raw = example.pop("question")
        question = question_raw + " " + instruction
        answer_raw = example.pop("answer")
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer_raw},
            ],
        }

    train_dataset = dataset["train"].map(function=process_fn, with_indices=True)
    test_dataset = dataset["test"].map(function=process_fn, with_indices=True)

    train_path = os.path.join(args.save_dir, "train.parquet")
    test_path = os.path.join(args.save_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"SFT: saved {len(train_dataset)} train -> {train_path}")
    print(f"SFT: saved {len(test_dataset)} test -> {test_path}")


if __name__ == "__main__":
    main()
