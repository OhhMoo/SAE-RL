#!/usr/bin/env bash
# Stage 3 pipeline: merge PPO checkpoints → collect activations → train SAEs
# Run from the SAE_RL/ directory:
#   conda activate sae_rl && bash scripts/run_stage3_pipeline.sh

set -euo pipefail

CONDA_ENV=sae_rl
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

PPO_CKPT_BASE="$ROOT/checkpoints/sae_rl_gsm8k/ppo_qwen2.5_0.5b"
MERGED_DIR="$ROOT/checkpoints/ppo_merged"
ACTIVATIONS_DIR="$ROOT/data/activations"
SAES_DIR="$ROOT/checkpoints/saes"

# Checkpoints to analyse: SFT baseline + sparse PPO coverage
SFT_MERGED="$ROOT/checkpoints/sft_merged"
PPO_STEPS=(10 50 100 200 300 435)
LAYERS="6 12 18 23"

mkdir -p "$MERGED_DIR" "$ACTIVATIONS_DIR" "$SAES_DIR"

run_py() {
    conda run -n "$CONDA_ENV" --no-capture-output python "$@"
}

# ── Step 1: Merge PPO FSDP checkpoints ────────────────────────────────────────
echo "=== Merging PPO checkpoints ==="
for step in "${PPO_STEPS[@]}"; do
    out="$MERGED_DIR/step_${step}"
    if [ -f "$out/model.safetensors" ] || [ -f "$out/pytorch_model.bin" ]; then
        echo "  step_${step}: already merged, skipping"
        continue
    fi
    echo "  Merging step_${step}..."
    conda run -n "$CONDA_ENV" --no-capture-output \
        python -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$PPO_CKPT_BASE/global_step_${step}/actor" \
            --target_dir "$out"
done

# ── Step 2: Collect activations ────────────────────────────────────────────────
echo ""
echo "=== Collecting activations ==="

collect() {
    local name="$1"
    local model_path="$2"
    # Skip if all layer files already exist
    local missing=0
    for layer in $LAYERS; do
        [ -f "$ACTIVATIONS_DIR/${name}_layer${layer}.pt" ] || missing=1
    done
    if [ "$missing" -eq 0 ]; then
        echo "  ${name}: activations already exist, skipping"
        return
    fi
    echo "  Collecting activations for: ${name}"
    run_py "$SCRIPT_DIR/04_collect_activations.py" \
        --model_path "$model_path" \
        --checkpoint_name "$name" \
        --layers $LAYERS \
        --save_dir "$ACTIVATIONS_DIR" \
        --batch_size 16
}

collect "sft" "$SFT_MERGED"
for step in "${PPO_STEPS[@]}"; do
    collect "ppo_step${step}" "$MERGED_DIR/step_${step}"
done

# ── Step 3: Train SAEs ─────────────────────────────────────────────────────────
echo ""
echo "=== Training SAEs ==="
run_py "$SCRIPT_DIR/05_train_sae.py" \
    --activations_dir "$ACTIVATIONS_DIR" \
    --save_dir "$SAES_DIR" \
    --expansion_factor 8 \
    --k 32 \
    --epochs 10 \
    --lr 3e-4 \
    --batch_size 256

echo ""
echo "=== Stage 3 complete ==="
echo "Merged checkpoints: $MERGED_DIR"
echo "Activations:        $ACTIVATIONS_DIR"
echo "SAEs:               $SAES_DIR"
