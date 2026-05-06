#!/usr/bin/env bash
# sweep_l23.sh — Capacity sweep on layer 23, ppo_step100.
#
# Grid: k ∈ {64, 128, 256} × expansion ∈ {8, 16}  → 6 SAEs.
# Cold-start (no warm-start init), 20 epochs, lr 1e-4, batch 512.
# Trains on _train.pt (80%); val MSE selects best epoch; FVE/frac_rec on _val.pt.
#
# Prereqs:
#   data/activations/ppo_step100_layer23_{train,val}.pt   (split_activations.py)
#   checkpoints/ppo_merged/step_100                       (for delta loss)
#
# Run from sae_rl/:  bash scripts/sweep_l23.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/water/.conda/envs/sae_rl/bin/python
SWEEP_DIR=checkpoints/saes_sweep_l23
ACT_DIR=data/activations
STAGE=ppo_step100
LAYER=23
LOG=logs/sweep_l23_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$SWEEP_DIR" logs

KS=(64 128 256)
EXPS=(8 16)

echo "=== L23 capacity sweep: k×expansion = ${KS[*]} × ${EXPS[*]} ==="
echo "Log: $LOG"

{
for K in "${KS[@]}"; do
    for E in "${EXPS[@]}"; do
        SUFFIX="_k${K}_exp${E}"
        echo ""
        echo "================================================================"
        echo "[sweep] k=$K  expansion=$E"
        echo "================================================================"
        $PY scripts/05_train_sae.py \
            --activations_dir "$ACT_DIR" \
            --save_dir        "$SWEEP_DIR" \
            --source          "${STAGE}_layer${LAYER}" \
            --output_suffix   "$SUFFIX" \
            --expansion_factor "$E" \
            --k               "$K" \
            --epochs          20 \
            --lr              1e-4 \
            --batch_size      512 \
            --device          cuda \
            --resample_interval 10 \
            --dead_threshold  1e-4
    done
done
echo ""
echo "=== Training done. Running eval ==="
$PY scripts/eval_l23_sweep.py \
    --sae_dir         "$SWEEP_DIR" \
    --activations_dir "$ACT_DIR" \
    --stage           "$STAGE" \
    --layer           "$LAYER" \
    --model_path      checkpoints/ppo_merged/step_100 \
    --output_csv      results/l23_sweep.csv
} 2>&1 | tee "$LOG"

echo "Done. Metrics -> results/l23_sweep.csv"
