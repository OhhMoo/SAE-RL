#!/usr/bin/env bash
# sweep_hyperparams.sh
# Cole's STNJ 2026-05-01 ask: standardize SAE hyperparams across layers,
# try 16x expansion + LR variations.
#
# Sweep cells: expansion x lr = {8, 16} x {1e-4, 5e-4} = 4 cells.
# Per cell: instruct_base + ppo_step30 (warm-started), L6/L12/L18/L23.
# Strict chain only. k=64 held fixed across all cells (Cole's "standardize").
#
# Reuses existing artifacts (no re-extraction, no re-merge):
#   data/activations_strict/{instruct_base,ppo_step30}_layer{6,12,18,23}_{train,val}.pt
#
# Outputs:
#   checkpoints/saes_sweep/exp{E}_lr{LR}/sae_{stage}_layer{L}.pt
# Eval (run separately after the sweep):
#   for D in checkpoints/saes_sweep/exp*_lr*; do
#     CELL=$(basename "$D")
#     python scripts/topk/06b_eval_sae.py --sae_dir "$D" --activations_dir data/activations_strict \
#       --merged_dir checkpoints/ppo_merged_strict --output_csv "results/sae_eval_sweep_${CELL}.csv"
#   done
#
# Run from sae_rl/ root.

set -euo pipefail
cd "$(dirname "$0")/../.."

ACT_DIR="${ACT_DIR:-data/activations_strict}"
SWEEP_ROOT="${SWEEP_ROOT:-checkpoints/saes_sweep}"
LAYERS=(6 12 18 23)
STAGES=(instruct_base ppo_step30)
K=64
EPOCHS=20
BATCH=512

EXPANSIONS=(8 16)
LRS=(1e-4 5e-4)

for STAGE in "${STAGES[@]}"; do
    for L in "${LAYERS[@]}"; do
        F="$ACT_DIR/${STAGE}_layer${L}_train.pt"
        [ -f "$F" ] || { echo "[abort] missing $F"; exit 1; }
    done
done

mkdir -p "$SWEEP_ROOT"

for EXP in "${EXPANSIONS[@]}"; do
    for LR in "${LRS[@]}"; do
        CELL="exp${EXP}_lr${LR}"
        SAE_DIR="$SWEEP_ROOT/$CELL"
        mkdir -p "$SAE_DIR"

        echo ""
        echo "============================================================"
        echo " Cell $CELL (expansion=$EXP, lr=$LR, k=$K, epochs=$EPOCHS)"
        echo "============================================================"

        PREV_STAGE=""
        for STAGE in "${STAGES[@]}"; do
            all_done=true
            for L in "${LAYERS[@]}"; do
                [ -f "$SAE_DIR/sae_${STAGE}_layer${L}.pt" ] || { all_done=false; break; }
            done
            if $all_done; then
                echo "[skip] $CELL/$STAGE already trained"
                PREV_STAGE="$STAGE"
                continue
            fi

            TMP="data/activations_tmp_${CELL}_${STAGE}"
            mkdir -p "$TMP"
            for L in "${LAYERS[@]}"; do
                for SUFFIX in _train _val; do
                    SRC="$ACT_DIR/${STAGE}_layer${L}${SUFFIX}.pt"
                    [ -f "$SRC" ] && ln -sf "$(realpath "$SRC")" "$TMP/${STAGE}_layer${L}${SUFFIX}.pt"
                done
            done

            INIT_ARGS=()
            if [ -n "$PREV_STAGE" ]; then
                INIT_ARGS=(--init_from_stage "$PREV_STAGE" --init_from_dir "$SAE_DIR")
            fi

            echo "[train] $CELL/$STAGE"
            python scripts/topk/05_train_sae.py \
                --activations_dir "$TMP" \
                --save_dir        "$SAE_DIR" \
                --expansion_factor "$EXP" \
                --k                "$K" \
                --epochs           "$EPOCHS" \
                --lr               "$LR" \
                --batch_size       "$BATCH" \
                --device           cuda \
                --resample_interval 10 \
                --dead_threshold   1e-4 \
                "${INIT_ARGS[@]}"

            rm -rf "$TMP"
            PREV_STAGE="$STAGE"
        done
    done
done

echo ""
echo "============================================================"
echo " Sweep training complete: $SWEEP_ROOT/{exp8,exp16}_lr{1e-4,5e-4}/"
echo " Eval each cell with the snippet in this script's header."
echo "============================================================"
