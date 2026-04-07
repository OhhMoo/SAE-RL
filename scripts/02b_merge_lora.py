"""
Step 2b: Merge LoRA adapter weights back into the base model.

verl SFT with FSDP writes checkpoints under ``global_step_*`` with
``model_world_size_*_rank_*.pt`` and ``lora_train_meta.json``, not a HuggingFace
PEFT folder (no ``adapter_config.json``). Convert that checkpoint first, then
merge if you need a single flat HF model:

    python -m verl.model_merger merge --backend fsdp \\
        --local_dir checkpoints/sft/global_step_87 \\
        --target_dir checkpoints/sft_hf

That writes base weights under ``checkpoints/sft_hf`` and LoRA under
``checkpoints/sft_hf/lora_adapter``. Then either point PPO at base+adapter, or
run this script with ``--lora_path`` set to ``.../lora_adapter`` and
``--base_model`` set to ``checkpoints/sft_hf`` (local path).

If you already have a standard PEFT export (``adapter_config.json`` next to
weights), use:

    python scripts/02b_merge_lora.py \\
        --base_model Qwen/Qwen2.5-0.5B-Instruct \\
        --lora_path path/to/adapter_dir \\
        --output_path checkpoints/sft_merged
"""

import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading LoRA adapter: {args.lora_path}")
    model = PeftModel.from_pretrained(model, args.lora_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_path}")
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print("Done.")


if __name__ == "__main__":
    main()
