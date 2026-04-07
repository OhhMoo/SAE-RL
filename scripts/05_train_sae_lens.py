"""
Step 5 (SAELens): Train SAEs on Qwen checkpoints using SAELens.

Replaces the hand-rolled 05_train_sae.py with SAELens's LanguageModelSAETrainingRunner,
which uses BatchTopK (SOTA), proper dead-feature handling, and WandB logging.

Qwen2.5-0.5B specs:
    d_model = 896
    num_layers = 24
    hook pattern: model.layers.{N}   (HuggingFace AutoModelForCausalLM)

Usage:
    # Train SAEs for all three checkpoints × all target layers
    python scripts/05_train_sae_lens.py --checkpoint pretrained
    python scripts/05_train_sae_lens.py --checkpoint sft   --model_path checkpoints/sft_merged
    python scripts/05_train_sae_lens.py --checkpoint ppo   --model_path checkpoints/ppo_merged

    # Quick smoke-test (short run, one layer)
    python scripts/05_train_sae_lens.py --checkpoint pretrained --layers 12 --training_tokens 500000
"""

import argparse
import os
import sys

import torch

# SAELens is installed at C:\Users\OhhMoo\Desktop\git_repo\SAE_Tools\SAELens
SAELENS_PATH = r"C:\Users\OhhMoo\Desktop\git_repo\SAE_Tools\SAELens"
if SAELENS_PATH not in sys.path:
    sys.path.insert(0, SAELENS_PATH)

from sae_lens import (
    BatchTopKTrainingSAEConfig,
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    LoggingConfig,
)

# ---------------------------------------------------------------------------
# Qwen2.5-0.5B constants
# ---------------------------------------------------------------------------
D_MODEL = 896          # hidden_size
EXPANSION = 8          # d_sae = D_MODEL * EXPANSION = 7168
K = 32                 # mean active features per token (BatchTopK target)
DEFAULT_LAYERS = [6, 12, 18, 23]

CHECKPOINT_TO_MODEL = {
    "pretrained": "Qwen/Qwen2.5-0.5B-Instruct",
    "sft":        "checkpoints/sft_merged",
    "ppo":        "checkpoints/ppo_merged",
}


def train_for_layer(
    layer: int,
    model_path: str,
    checkpoint_name: str,
    training_tokens: int,
    save_dir: str,
    device: str,
) -> None:
    hook_name = f"model.layers.{layer}"
    d_sae = D_MODEL * EXPANSION

    total_training_steps = training_tokens // 4096
    lr_warm_up_steps = total_training_steps // 20
    lr_decay_steps = total_training_steps // 5

    cfg = LanguageModelSAERunnerConfig(
        # ---- Model ----
        model_name=model_path,
        model_class_name="AutoModelForCausalLM",
        hook_name=hook_name,
        model_from_pretrained_kwargs={"torch_dtype": "float16"},

        # ---- SAE architecture: BatchTopK (SOTA) ----
        sae=BatchTopKTrainingSAEConfig(
            d_in=D_MODEL,
            d_sae=d_sae,
            k=K,
            normalize_activations="expected_average_only_in",
        ),

        # ---- Dataset: stream GSM8k questions as training text ----
        dataset_path="openai/gsm8k",
        dataset_trust_remote_code=True,
        streaming=True,
        is_dataset_tokenized=False,
        context_size=512,
        prepend_bos=True,

        # ---- Training ----
        training_tokens=training_tokens,
        train_batch_size_tokens=4096,
        store_batch_size_prompts=16,
        n_batches_in_buffer=32,
        lr=3e-4,
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_decay_steps,

        # ---- Dead feature handling ----
        dead_feature_window=1000,
        dead_feature_threshold=1e-6,

        # ---- Memory ----
        device=device,
        act_store_device="cpu",
        autocast=True,
        autocast_lm=True,
        dtype="float32",

        # ---- Checkpointing ----
        checkpoint_path=os.path.join(save_dir, f"{checkpoint_name}_layer{layer}"),
        n_checkpoints=3,
        save_final_checkpoint=True,

        # ---- WandB ----
        logger=LoggingConfig(
            log_to_wandb=True,
            wandb_project="sae_rl_gsm8k",
            run_name=f"sae_{checkpoint_name}_layer{layer}",
            wandb_log_frequency=50,
            eval_every_n_wandb_logs=10,
        ),
    )

    print(f"\n{'='*60}")
    print(f"Training SAE: checkpoint={checkpoint_name}, layer={layer}")
    print(f"  model:   {model_path}")
    print(f"  hook:    {hook_name}")
    print(f"  d_model: {D_MODEL}  d_sae: {d_sae}  k: {K}")
    print(f"  tokens:  {training_tokens:,}")
    print(f"{'='*60}")

    runner = LanguageModelSAETrainingRunner(cfg)
    sae = runner.run()

    # Save a copy with a stable name for the analysis script
    stable_path = os.path.join(save_dir, f"{checkpoint_name}_layer{layer}")
    sae.save_model(stable_path)
    print(f"  Saved -> {stable_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True,
        choices=["pretrained", "sft", "ppo"],
        help="Which model stage to train SAEs on",
    )
    parser.add_argument(
        "--model_path", default=None,
        help="Override model path (defaults to CHECKPOINT_TO_MODEL mapping)",
    )
    parser.add_argument(
        "--layers", type=int, nargs="+", default=DEFAULT_LAYERS,
        help="Which transformer layers to train SAEs on",
    )
    parser.add_argument(
        "--training_tokens", type=int, default=10_000_000,
        help="Total tokens to train each SAE on",
    )
    parser.add_argument(
        "--save_dir", default="checkpoints/saes_lens",
        help="Root dir for saved SAEs",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    model_path = args.model_path or CHECKPOINT_TO_MODEL[args.checkpoint]
    os.makedirs(args.save_dir, exist_ok=True)

    for layer in args.layers:
        train_for_layer(
            layer=layer,
            model_path=model_path,
            checkpoint_name=args.checkpoint,
            training_tokens=args.training_tokens,
            save_dir=args.save_dir,
            device=args.device,
        )


if __name__ == "__main__":
    main()
