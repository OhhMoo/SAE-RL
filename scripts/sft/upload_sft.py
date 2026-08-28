"""Publish the GSM8k-SFT model to the Hugging Face Hub (fp32, as saved).

Creates the target model repository (public), writes a model card into the
HF model dir, and uploads the folder. Set HF_REPO_ID to the target repo
(e.g. "<namespace>/qwen05b-gsm8k-sft-instruct") and authenticate with a
write-enabled HF token beforehand.
"""

import os

from huggingface_hub import HfApi

REPO = os.environ.get("HF_REPO_ID", "<namespace>/qwen05b-gsm8k-sft-instruct")
MODEL_DIR = "checkpoints/gsm8k-sft/qwen05b-gsm8k-sft-instruct/global_step_348/huggingface"

SYSTEM_PROMPT = (
    "You are a math problem solver. Think step by step. You MUST end your response "
    "with '#### <number>' where <number> is the final numerical answer (digits only, "
    "no units or markdown)."
)

CARD = f"""---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B-Instruct
datasets:
- openai/gsm8k
language:
- en
pipeline_tag: text-generation
library_name: transformers
tags:
- gsm8k
- math
- sft
- qwen2.5
---

# Qwen2.5-0.5B-Instruct — GSM8k SFT

Full supervised fine-tune of [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
on the [GSM8k](https://huggingface.co/datasets/openai/gsm8k) training split (7,473 worked
solutions), trained with [verl](https://github.com/volcengine/verl)'s `sft_trainer`.

## Results

| Metric | Value |
|---|---|
| GSM8k test accuracy (greedy, 1319 examples) | **36.2%** (477/1319) |

## Training

- Base model: `Qwen/Qwen2.5-0.5B-Instruct`
- Method: full fine-tune (no LoRA), bf16 mixed precision
- Data: GSM8k train, chat format (system + user); assistant target is the full worked solution ending in `#### N`
- Epochs: 3 (348 steps); AdamW, lr 1e-5 cosine, warmup ratio 0.1, weight decay 0.1, grad clip 1.0
- Max sequence length: 1024

## Prompt format

System prompt used during training; the model ends its answer with `#### <number>`:

> {SYSTEM_PROMPT}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

m = AutoModelForCausalLM.from_pretrained("{REPO}")
tok = AutoTokenizer.from_pretrained("{REPO}")

messages = [
    {{"role": "system", "content": "{SYSTEM_PROMPT}"}},
    {{"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's think step by step."}},
]
ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
out = m.generate(ids, max_new_tokens=512, do_sample=False)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```
"""


def main():
    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "README.md"), "w") as f:
        f.write(CARD)
    api.upload_folder(
        repo_id=REPO,
        folder_path=MODEL_DIR,
        repo_type="model",
        commit_message="GSM8k-SFT Qwen2.5-0.5B-Instruct (full FT, 36.2% GSM8k test)",
    )
    print("UPLOADED: https://huggingface.co/" + REPO)


if __name__ == "__main__":
    main()
