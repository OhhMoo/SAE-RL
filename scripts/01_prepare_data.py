"""
Step 1: Prepare GSM8k for PPO/RL (``prompt`` + reward_model).

For supervised fine-tuning use ``01b_prepare_sft_data.py`` instead (``messages`` column).

Usage:
    python scripts/01_prepare_data.py --save_dir data/gsm8k
"""

import argparse
import os
import re

import datasets


def extract_solution(solution_str):
    solution = re.search(r"#### (\-?[0-9\.\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


SYSTEM_PROMPT = (
    "You are a math problem solver. Think step by step. "
    "You MUST end your response with '#### <number>' where <number> is "
    "the final numerical answer (digits only, no units or markdown)."
)


def make_map_fn(split):
    instruction = 'Let\'s think step by step and output the final answer after "####".'

    def process_fn(example, idx):
        question_raw = example.pop("question")
        question = question_raw + " " + instruction
        answer_raw = example.pop("answer")
        solution = extract_solution(answer_raw)

        return {
            "data_source": "openai/gsm8k",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": answer_raw,
                "question": question_raw,
            },
        }

    return process_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="data/gsm8k")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = datasets.load_dataset("openai/gsm8k", "main")

    train_dataset = dataset["train"].map(function=make_map_fn("train"), with_indices=True)
    test_dataset = dataset["test"].map(function=make_map_fn("test"), with_indices=True)

    train_path = os.path.join(args.save_dir, "train.parquet")
    test_path = os.path.join(args.save_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"Saved {len(train_dataset)} train examples to {train_path}")
    print(f"Saved {len(test_dataset)} test examples to {test_path}")


if __name__ == "__main__":
    main()
