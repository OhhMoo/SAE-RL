#!/usr/bin/env bash
# run_sft_pipeline.sh
# SAE pipeline over the pure-SFT checkpoint chain (no RL).
# Same recipe as run_dense_pipeline.sh, but the SFT checkpoints are already
# merged HF models (checkpoints/gsm8k-sft/.../global_step_N/huggingface),
# so there is no FSDP merge stage. For each checkpoint:
#   1. Collect residual-stream activations (layers 6, 12, 18, 23)
#   2. Train a TopK SAE (k=64, 8x expansion, 20 epochs) per (step, layer),
#      warm-started from the previous stage so feature indices stay aligned.
# Warm-start chain: instruct_base -> sft_step29 -> sft_step58 -> ... -> sft_step348.
# Baseline instruct_base SAEs must already be in $SAE_DIR (pulled from
# OhhMoo/sae-rl-qwen05b-layers sae_flexible/, identical to the strict-chain
# baselines at L6/12/18); baseline activations are reused from data/activations.
#
# Run from sae_rl/ root: bash scripts/run_sft_pipeline.sh

set -euo pipefail
cd "$(dirname "$0")/.."

SFT_CKPT_ROOT="${SFT_CKPT_ROOT:-checkpoints/gsm8k-sft/qwen05b-gsm8k-sft-instruct-ckpts}"
ACT_DIR="${ACT_DIR:-data/activations_sft}"
SAE_DIR="${SAE_DIR:-checkpoints/saes_sft}"
LAYERS=(6 12 18 23)
# SFT save_freq=29 (quarter-epoch over 3 epochs / 348 steps). Spread mirrors
# the ~7-step density used for the PPO chains.
read -r -a SFT_STEPS <<< "${SFT_STEPS:-29 58 116 174 232 290 348}"

BASELINE_LABEL="instruct_base"

for L in "${LAYERS[@]}"; do
    if [ ! -f "$SAE_DIR/sae_${BASELINE_LABEL}_layer${L}.pt" ]; then
        echo "[error] Missing baseline SAE: $SAE_DIR/sae_${BASELINE_LABEL}_layer${L}.pt" >&2
        exit 1
    fi
done

mkdir -p "$ACT_DIR"

echo "============================================================"
echo " SFT SAE pipeline: steps ${SFT_STEPS[*]}"
echo "============================================================"

PREV_STAGE="$BASELINE_LABEL"

for STEP in "${SFT_STEPS[@]}"; do
    MODEL_DIR="$SFT_CKPT_ROOT/global_step_${STEP}/huggingface"
    STAGE_LABEL="sft_step${STEP}"

    echo ""
    echo "------------------------------------------------------------"
    echo " Step $STEP"
    echo "------------------------------------------------------------"

    if [ ! -d "$MODEL_DIR" ]; then
        echo "[warn] SFT checkpoint not found: $MODEL_DIR — skipping step $STEP"
        continue
    fi

    # ── 1. Collect activations ────────────────────────────────────────────
    all_acts_exist=true
    for L in "${LAYERS[@]}"; do
        if [ ! -f "$ACT_DIR/${STAGE_LABEL}_layer${L}_train.pt" ] && [ ! -f "$ACT_DIR/${STAGE_LABEL}_layer${L}.pt" ]; then
            all_acts_exist=false
            break
        fi
    done

    if $all_acts_exist; then
        echo "[skip] Activations already collected for $STAGE_LABEL"
    else
        echo "[activations] Collecting from $MODEL_DIR"
        python scripts/04_collect_activations.py \
            --model_path      "$MODEL_DIR" \
            --checkpoint_name "$STAGE_LABEL" \
            --layers          "${LAYERS[@]}" \
            --save_dir        "$ACT_DIR" \
            --max_length      512 \
            --batch_size      16 \
            --max_tokens      2000000
    fi

    # ── 2. Train SAEs for this step ───────────────────────────────────────
    TEMP_ACT_DIR="data/activations_tmp_sft_step${STEP}"
    mkdir -p "$TEMP_ACT_DIR"
    for L in "${LAYERS[@]}"; do
        for SUFFIX in _train _val ""; do
            SRC="$ACT_DIR/${STAGE_LABEL}_layer${L}${SUFFIX}.pt"
            DST="$TEMP_ACT_DIR/${STAGE_LABEL}_layer${L}${SUFFIX}.pt"
            if [ ! -f "$DST" ] && [ -f "$SRC" ]; then
                ln -s "$(realpath "$SRC")" "$DST"
            fi
        done
    done

    all_saes_exist=true
    for L in "${LAYERS[@]}"; do
        if [ ! -f "$SAE_DIR/sae_${STAGE_LABEL}_layer${L}.pt" ]; then
            all_saes_exist=false
            break
        fi
    done

    if $all_saes_exist; then
        echo "[skip] SAEs already trained for $STAGE_LABEL"
    else
        echo "[train SAEs] $STAGE_LABEL (warm-start from $PREV_STAGE)"
        python scripts/05_train_sae.py \
            --activations_dir "$TEMP_ACT_DIR" \
            --save_dir        "$SAE_DIR" \
            --expansion_factor 8 \
            --k               64 \
            --epochs          20 \
            --lr              1e-4 \
            --batch_size      512 \
            --device          cuda \
            --resample_interval 10 \
            --dead_threshold  1e-4 \
            --init_from_stage "$PREV_STAGE" \
            --init_from_dir   "$SAE_DIR"
    fi

    rm -rf "$TEMP_ACT_DIR"
    PREV_STAGE="$STAGE_LABEL"
    echo "[done] Step $STEP complete"
done

echo ""
echo "============================================================"
echo " SFT pipeline complete."
echo " Activations: $ACT_DIR"
echo " SAEs:        $SAE_DIR"
echo "============================================================"
