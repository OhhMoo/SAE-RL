#!/usr/bin/env bash
# run_dense_pipeline.sh
# Full post-PPO pipeline (instruct model → PPO-without-SFT → SAE analysis).
# For each checkpoint step:
#   1. Merge FSDP actor checkpoint → flat HF model  (ppo_merged/step_N)
#   2. Collect residual-stream activations (layers 6, 12, 18, 23)
#   3. Train a 40-epoch TopK SAE (k=64, 8× expansion) on each (step, layer) pair,
#      warm-started from the previous stage's SAE so feature indices stay aligned
#      across checkpoints (required for decoder-cosine drift to be meaningful).
# Also handles the step-0 baseline (Qwen/Qwen2.5-0.5B-Instruct directly).
# Warm-start chain: instruct_base → step10 → step30 → … → step200.
#
# Prerequisites:
#   - PPO run complete in the sibling ppo_run/ folder:
#       ../ppo_run/checkpoints/flexible/global_step_N/
#
# Overrides:
#   PPO_CKPT_ROOT=../ppo_run/checkpoints/strict bash scripts/run_dense_pipeline.sh
#   SAE_DIR=checkpoints/saes_custom bash scripts/run_dense_pipeline.sh
#
# Run from sae_rl/ root: bash scripts/run_dense_pipeline.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PPO_CKPT_ROOT="${PPO_CKPT_ROOT:-../ppo_run/checkpoints/flexible}"
MERGE_ROOT="${MERGE_ROOT:-checkpoints/ppo_merged}"
ACT_DIR="${ACT_DIR:-data/activations}"
SAE_DIR="${SAE_DIR:-checkpoints/saes}"
LAYERS=(6 12 18 23)
# PPO save_freq=10, so steps land on multiples of 10. Spread spans the early
# learning phase, the climb, and the post-plateau region (val acc plateaued
# ~step 140 at ~0.58 and stayed flat through step 200).
DENSE_STEPS=(10 30 60 100 140 180 200)

echo "============================================================"
echo " Full pipeline: steps ${DENSE_STEPS[*]}"
echo "============================================================"

# ── Step 0: instruct model baseline ──────────────────────────────────────────
BASELINE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
BASELINE_LABEL="instruct_base"

echo ""
echo "------------------------------------------------------------"
echo " Step 0 — baseline (${BASELINE_MODEL})"
echo "------------------------------------------------------------"

all_acts_exist=true
for L in "${LAYERS[@]}"; do
    [ ! -f "$ACT_DIR/${BASELINE_LABEL}_layer${L}.pt" ] && all_acts_exist=false && break
done

if $all_acts_exist; then
    echo "[skip] Baseline activations already collected"
else
    echo "[activations] Collecting from $BASELINE_MODEL"
    python scripts/04_collect_activations.py \
        --model_path      "$BASELINE_MODEL" \
        --checkpoint_name "$BASELINE_LABEL" \
        --layers          "${LAYERS[@]}" \
        --save_dir        "$ACT_DIR" \
        --max_length      512 \
        --batch_size      16 \
        --max_tokens      2000000
fi

all_saes_exist=true
for L in "${LAYERS[@]}"; do
    [ ! -f "$SAE_DIR/sae_${BASELINE_LABEL}_layer${L}.pt" ] && all_saes_exist=false && break
done

if $all_saes_exist; then
    echo "[skip] Baseline SAEs already trained"
else
    TEMP_ACT_DIR="data/activations_tmp_step0"
    mkdir -p "$TEMP_ACT_DIR"
    for L in "${LAYERS[@]}"; do
        SRC="$ACT_DIR/${BASELINE_LABEL}_layer${L}.pt"
        [ -f "$SRC" ] && ln -sf "$(realpath "$SRC")" "$TEMP_ACT_DIR/${BASELINE_LABEL}_layer${L}.pt"
    done

    echo "[train SAEs] $BASELINE_LABEL"
    python scripts/05_train_sae.py \
        --activations_dir "$TEMP_ACT_DIR" \
        --save_dir        "$SAE_DIR" \
        --expansion_factor 8 \
        --k               64 \
        --epochs          40 \
        --lr              1e-4 \
        --batch_size      512 \
        --device          cuda \
        --resample_interval 10 \
        --dead_threshold  1e-4
    rm -rf "$TEMP_ACT_DIR"
fi

echo "[done] Step 0 complete"

# Warm-start lineage: each step's SAEs are initialised from the previous stage's.
PREV_STAGE="$BASELINE_LABEL"

for STEP in "${DENSE_STEPS[@]}"; do
    CKPT_DIR="$PPO_CKPT_ROOT/global_step_${STEP}"
    MERGED_DIR="$MERGE_ROOT/step_${STEP}"
    STAGE_LABEL="ppo_step${STEP}"

    echo ""
    echo "------------------------------------------------------------"
    echo " Step $STEP"
    echo "------------------------------------------------------------"

    # ── 1. Merge FSDP checkpoint → HF flat model ─────────────────────────
    if [ -d "$MERGED_DIR" ]; then
        echo "[skip] Merged model already exists: $MERGED_DIR"
    else
        if [ ! -d "$CKPT_DIR" ]; then
            echo "[warn] PPO checkpoint not found: $CKPT_DIR — skipping step $STEP"
            continue
        fi

        echo "[merge] $CKPT_DIR → $MERGED_DIR"
        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir  "$CKPT_DIR/actor" \
            --target_dir "$MERGED_DIR"
    fi

    # ── 2. Collect activations ────────────────────────────────────────────
    # Check if all 4 layer files already exist
    all_acts_exist=true
    for L in "${LAYERS[@]}"; do
        if [ ! -f "$ACT_DIR/${STAGE_LABEL}_layer${L}.pt" ]; then
            all_acts_exist=false
            break
        fi
    done

    if $all_acts_exist; then
        echo "[skip] Activations already collected for $STAGE_LABEL"
    else
        echo "[activations] Collecting from $MERGED_DIR"
        python scripts/04_collect_activations.py \
            --model_path      "$MERGED_DIR" \
            --checkpoint_name "$STAGE_LABEL" \
            --layers          "${LAYERS[@]}" \
            --save_dir        "$ACT_DIR" \
            --max_length      512 \
            --batch_size      16 \
            --max_tokens      2000000
    fi

    # ── 3. Train SAEs for this step ───────────────────────────────────────
    # Build a temp activations dir with only this step's files so 05_train_sae.py
    # doesn't re-scan all 28 existing files.
    TEMP_ACT_DIR="data/activations_tmp_step${STEP}"
    mkdir -p "$TEMP_ACT_DIR"

    for L in "${LAYERS[@]}"; do
        SRC="$ACT_DIR/${STAGE_LABEL}_layer${L}.pt"
        DST="$TEMP_ACT_DIR/${STAGE_LABEL}_layer${L}.pt"
        if [ ! -f "$DST" ] && [ -f "$SRC" ]; then
            ln -s "$(realpath "$SRC")" "$DST"
        fi
    done

    # Check if SAEs already trained for all layers
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
            --epochs          40 \
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
echo " Dense pipeline complete."
echo " Activations: $ACT_DIR"
echo " SAEs:        $SAE_DIR"
echo " Next: bash scripts/run_analysis.sh"
echo "============================================================"
