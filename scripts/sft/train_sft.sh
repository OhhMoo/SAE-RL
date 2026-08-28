#!/usr/bin/env bash
# Full fine-tune of Qwen2.5-0.5B-Instruct on GSM8k via the vendored verl sft_trainer.
# Single GPU, no LoRA. pad_mode=no_padding + use_remove_padding=False: the engine
# pads each micro-batch internally and uses a standard attention mask (sdpa), so no
# flash-attn is required.
# Produces an HF-format model at checkpoints/gsm8k-sft/qwen05b-gsm8k-sft-instruct/.
set -euo pipefail

# Paths are environment-configurable: REPO_DIR is this repository's root,
# VERL_DIR a local checkout of verl, ENV_BIN the python env's bin directory.
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VERL_DIR="${VERL_DIR:?Set VERL_DIR to a local verl checkout}"
ENV_BIN="${ENV_BIN:-$(dirname "$(which python)")}"
cd "$REPO_DIR"

export PYTHONPATH="$VERL_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

"$ENV_BIN/torchrun" --standalone --nnodes=1 --nproc-per-node=1 -m verl.trainer.sft_trainer \
    data.train_files=data/gsm8k_sft/train.parquet \
    data.val_files=null \
    data.messages_key=messages \
    data.pad_mode=no_padding \
    data.max_length=1024 \
    data.truncation=right \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=4096 \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=16 \
    model.path=Qwen/Qwen2.5-0.5B-Instruct \
    model.use_remove_padding=False \
    model.enable_gradient_checkpointing=False \
    model.lora_rank=0 \
    +model.override_config.attn_implementation=sdpa \
    optim.lr=1e-5 \
    optim.lr_scheduler_type=cosine \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.weight_decay=0.1 \
    optim.clip_grad=1.0 \
    engine.strategy=fsdp \
    engine.fsdp_size=1 \
    engine.use_torch_compile=False \
    trainer.total_epochs=3 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.resume_mode=disable \
    trainer.logger=[console,file] \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=qwen05b-gsm8k-sft-instruct \
    checkpoint.save_contents=[model,hf_model,extra]
