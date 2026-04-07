#!/bin/bash
# Step 2: Supervised Fine-Tuning of Qwen2.5-0.5B on GSM8k
#
# This gives the model a baseline ability to produce chain-of-thought
# math solutions before PPO refinement.
#
# Usage:
#   bash scripts/02_sft_qwen.sh <num_gpus> <save_path>
#   Example: bash scripts/02_sft_qwen.sh 2 checkpoints/sft
#
# Prerequisites:
#   - Run 01b_prepare_sft_data.py (SFT parquet with ``messages``; not 01_prepare_data.py)
#   - Install verl: pip install -e ../verl
#
# Uses PyTorch SDPA for attention (no flash-attn package). Hydra needs ``+`` to add keys to struct ``override_config``.
# To use FlashAttention 2: install flash_attn and remove the +model.override_config... line below.

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: 02_sft_qwen.sh <nproc_per_node> <save_path> [extra_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2
shift 2

# Resolve data paths relative to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="${PROJECT_DIR}/data/gsm8k_sft"

if [ ! -f "${DATA_DIR}/train.parquet" ]; then
    echo "ERROR: SFT data not found: ${DATA_DIR}/train.parquet"
    echo "Create it first:  python scripts/01b_prepare_sft_data.py --save_dir data/gsm8k_sft"
    exit 1
fi

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.sft_trainer \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.messages_key=messages \
    data.micro_batch_size_per_gpu=4 \
    data.max_length=1024 \
    optim.lr=1e-4 \
    engine=fsdp \
    model.path=Qwen/Qwen2.5-0.5B-Instruct \
    +model.override_config.attn_implementation=sdpa \
    trainer.default_local_dir=$save_path \
    trainer.project_name=sae_rl_gsm8k \
    trainer.experiment_name=sft_qwen2.5_0.5b \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=5 \
    model.lora_rank=32 \
    model.lora_alpha=32 \
    model.target_modules=all-linear \
    "$@"
