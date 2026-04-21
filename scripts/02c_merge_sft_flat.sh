#!/usr/bin/env bash
# 02c_merge_sft_flat.sh
#
# Problem: verl.model_merger writes base weights to model.safetensors and keeps
# the LoRA delta in a lora_adapter/ subdirectory.  AutoModelForCausalLM.from_pretrained
# silently loads only the base weights, ignoring the LoRA — so anything downstream
# that uses the "merged" path (PPO actor init, eval scripts) actually runs on the
# base Qwen model, not the SFT-trained one.
#
# Fix: run 02b_merge_lora.py to bake LoRA into a flat model, then gate on solve rate.
#
# Usage:
#   bash scripts/02c_merge_sft_flat.sh                    # default: sft_hf -> sft_merged
#   SFT_HF_DIR=checkpoints/sft_hf bash scripts/02c_merge_sft_flat.sh
#
# Outputs:
#   checkpoints/sft_merged/    (or SFT_FLAT_OUT)

set -euo pipefail
cd "$(dirname "$0")/.."

# ── Config ────────────────────────────────────────────────────────────────────
# Directory produced by verl.model_merger: contains model.safetensors + lora_adapter/
SFT_HF_DIR="${SFT_HF_DIR:-checkpoints/sft_hf}"
# Output: truly flat merged model (base + LoRA baked in)
SFT_FLAT_OUT="${SFT_FLAT_OUT:-checkpoints/sft_merged}"
# Gate thresholds (exit 1 if below)
MIN_ACCURACY="${MIN_ACCURACY:-0.25}"
MIN_FORMAT_RATE="${MIN_FORMAT_RATE:-0.80}"

echo "============================================================"
echo " 02c — Flatten SFT model"
echo "  Input : $SFT_HF_DIR"
echo "  Output: $SFT_FLAT_OUT"
echo "============================================================"

# ── 1. Sanity check ───────────────────────────────────────────────────────────
if [ ! -f "$SFT_HF_DIR/model.safetensors" ]; then
    echo "[error] $SFT_HF_DIR/model.safetensors not found."
    echo "        Run verl.model_merger on the SFT FSDP checkpoint first:"
    echo "          python -m verl.model_merger merge --backend fsdp \\"
    echo "            --local_dir checkpoints/sft/global_step_<N>/actor \\"
    echo "            --target_dir $SFT_HF_DIR"
    exit 1
fi

if [ ! -f "$SFT_HF_DIR/lora_adapter/adapter_config.json" ]; then
    echo "[error] No lora_adapter/adapter_config.json in $SFT_HF_DIR."
    echo "        Expected verl.model_merger output with separate LoRA adapter."
    exit 1
fi

echo "[ok] Found base weights + lora_adapter in $SFT_HF_DIR"

# ── 2. Merge LoRA into flat model ─────────────────────────────────────────────
if [ -f "$SFT_FLAT_OUT/model.safetensors" ] && [ ! -d "$SFT_FLAT_OUT/lora_adapter" ]; then
    echo "[skip] Flat model already exists at $SFT_FLAT_OUT"
else
    echo ""
    echo "[merge] Baking LoRA into flat model → $SFT_FLAT_OUT"
    python scripts/02b_merge_lora.py \
        --base_model  "$SFT_HF_DIR" \
        --lora_path   "$SFT_HF_DIR/lora_adapter" \
        --output_path "$SFT_FLAT_OUT"

    # Confirm the output has no lora_adapter subdirectory
    if [ -d "$SFT_FLAT_OUT/lora_adapter" ]; then
        echo "[error] lora_adapter/ still present in $SFT_FLAT_OUT after merge — merge failed"
        exit 1
    fi
    echo "[ok] Flat model saved to $SFT_FLAT_OUT"
fi

# ── 3. Accuracy gate ──────────────────────────────────────────────────────────
echo ""
echo "[eval] Verifying solve rate on 200 GSM8k test samples..."
python scripts/03b_eval_sft.py \
    --model_path      "$SFT_FLAT_OUT" \
    --data_path       data/gsm8k/test.parquet \
    --n_samples       200 \
    --max_new_tokens  512 \
    --min_accuracy    "$MIN_ACCURACY" \
    --min_format_rate "$MIN_FORMAT_RATE"

echo ""
echo "============================================================"
echo " Flat SFT model ready at: $SFT_FLAT_OUT"
echo " Next: bash scripts/03_ppo_qwen_v4.sh"
echo "============================================================"
