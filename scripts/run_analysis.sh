#!/usr/bin/env bash
# run_analysis.sh
# Full analysis pipeline after SAEs are trained (v4: instruct model → PPO).
#   09  → greedy eval on every checkpoint → training_curves.csv
#   10  → classify features (concise_rise / verbose_rise / stable)
#   11  → format-transition figure per layer
#   12  → auto-interpret top features via Claude API
#   13  → causal ablation
#   14  → drift heatmap figure
#   15  → summary chart
#
# Set ANTHROPIC_API_KEY before running if you want auto-interpretation.
# Run from repo root: bash scripts/run_analysis.sh

set -euo pipefail
cd "$(dirname "$0")/.."

SAE_DIR="${SAE_DIR:-checkpoints/saes}"
ACT_DIR="data/activations"
ANALYSIS_DIR="results/precedence_analysis"
HEATMAP_DIR="results/feature_analysis"

echo "============================================================"
echo " Step 09 — Greedy eval on all checkpoints (training curves)"
echo "============================================================"
# Authoritative source: greedy decoding on each merged model.
# Step 0 = Qwen/Qwen2.5-0.5B-Instruct baseline (no PPO).
python scripts/09_eval_checkpoints.py \
    --output         results/training_curves.csv \
    --n_samples      200 \
    --batch_size     8 \
    --max_new_tokens 512 \
    --split          test

echo ""
echo "============================================================"
echo " Step 10 — Classify features (concise_rise / verbose_rise)"
echo "============================================================"
python scripts/10_classify_features.py \
    --sae_dir         "$SAE_DIR" \
    --dense_sae_dir   "$SAE_DIR" \
    --activations_dir "$ACT_DIR" \
    --output_dir      "$ANALYSIS_DIR" \
    --layers          6 12 18 23 \
    --device          cuda \
    --hacking_threshold 0.005

echo ""
echo "============================================================"
echo " Step 11 — Format-transition figure (per layer)"
echo "============================================================"
for LAYER in 6 12 18 23; do
    echo "  → Layer $LAYER"
    mkdir -p "$ANALYSIS_DIR/layer${LAYER}"
    python scripts/11_temporal_precedence.py \
        --training_curves results/training_curves.csv \
        --analysis_dir    "$ANALYSIS_DIR" \
        --output_dir      "$ANALYSIS_DIR/layer${LAYER}" \
        --focus_layer     "$LAYER" \
        --layers          6 12 18 23 \
        --n_sigma         2.0
done

echo ""
echo "============================================================"
echo " Step 12 — Auto-interpret top features (Claude API)"
echo "============================================================"
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "[warn] ANTHROPIC_API_KEY not set — dry run only"
    for LAYER in 23 18; do
        python scripts/12_auto_interp.py \
            --layer              "$LAYER" \
            --stage              ppo_step100 \
            --sae_dir            "$SAE_DIR" \
            --activations_dir    "$ACT_DIR" \
            --classification_dir "$ANALYSIS_DIR" \
            --output_dir         "$ANALYSIS_DIR" \
            --n_features         20 \
            --top_k_tokens       10 \
            --dry_run
    done
else
    for LAYER in 23 18; do
        echo "  → Layer $LAYER"
        python scripts/12_auto_interp.py \
            --layer              "$LAYER" \
            --stage              ppo_step100 \
            --sae_dir            "$SAE_DIR" \
            --activations_dir    "$ACT_DIR" \
            --classification_dir "$ANALYSIS_DIR" \
            --output_dir         "$ANALYSIS_DIR" \
            --n_features         20 \
            --top_k_tokens       10
    done
fi

echo ""
echo "============================================================"
echo " Step 13 — Causal ablation"
echo "============================================================"
python scripts/13_ablation.py \
    --sae_dir            "$SAE_DIR" \
    --classification_dir "$ANALYSIS_DIR" \
    --output_dir         results/ablation \
    --layers             23 18 \
    --n_hacking_feats    10 20 50 \
    --device             cuda

echo ""
echo "============================================================"
echo " Step 14 — Drift heatmap"
echo "============================================================"
python scripts/14_drift_heatmap.py \
    --sae_dir    "$SAE_DIR" \
    --output_dir "$HEATMAP_DIR" \
    --layers     6 12 18 23

echo ""
echo "============================================================"
echo " Step 15 — Summary chart"
echo "============================================================"
python scripts/15_summary_chart.py

echo ""
echo "============================================================"
echo " All analysis complete. Results: results/"
echo "============================================================"
