#!/usr/bin/env bash
# run_retrain_saes.sh
# Retrain all SAEs at 50 epochs (fix the undertrained 10-epoch SAEs).
# Saves to checkpoints/saes/ (canonical SAE dir).
# Run from repo root: bash scripts/run_retrain_saes.sh

set -euo pipefail
cd "$(dirname "$0")/.."

SAVE_DIR="checkpoints/saes"
ACT_DIR="data/activations"

echo "============================================================"
echo " Retraining all SAEs: 50 epochs → $SAVE_DIR"
echo "============================================================"

python scripts/05_train_sae.py \
    --activations_dir "$ACT_DIR" \
    --save_dir        "$SAVE_DIR" \
    --expansion_factor 8 \
    --k               32 \
    --epochs          50 \
    --lr              3e-4 \
    --batch_size      256 \
    --device          cuda \
    --resample_interval 5 \
    --dead_threshold  1e-4

echo ""
echo "Done. SAEs saved to $SAVE_DIR"
echo "Next: bash scripts/run_dense_pipeline.sh"
