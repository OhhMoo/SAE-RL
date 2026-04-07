#!/bin/bash
# Full pipeline: Data Prep -> SFT -> Merge LoRA -> PPO -> Collect Activations -> Train SAEs -> Analyze
#
# Backbone: Qwen/Qwen2.5-0.5B-Instruct (see 02_sft_qwen.sh, 03_ppo_qwen.sh, 02b_merge_lora.py).
#
# This is a reference script — you'll likely run each step individually and
# inspect intermediate results. Adjust paths and GPU counts to your setup.

set -e
set -x

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

NUM_GPUS=${NUM_GPUS:-2}

# ============================================================
# Step 1: Prepare GSM8k data
# ============================================================
echo ">>> Step 1: Preparing GSM8k data (RL / PPO parquet)..."
python scripts/01_prepare_data.py --save_dir data/gsm8k

echo ">>> Step 1b: Preparing GSM8k SFT parquet (messages column)..."
python scripts/01b_prepare_sft_data.py --save_dir data/gsm8k_sft

# ============================================================
# Step 2: SFT on GSM8k
# ============================================================
echo ">>> Step 2: SFT training..."
bash scripts/02_sft_qwen.sh $NUM_GPUS checkpoints/sft

# ============================================================
# Step 2b: Merge LoRA weights
# ============================================================
echo ">>> Step 2b: Merging LoRA weights..."
# NOTE: Update the lora_path to match your actual checkpoint path
SFT_CKPT=$(ls -d checkpoints/sft/sae_rl_gsm8k/sft_qwen2.5_0.5b/global_step_*/actor 2>/dev/null | tail -1)
python scripts/02b_merge_lora.py \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --lora_path "$SFT_CKPT" \
    --output_path checkpoints/sft_merged

# ============================================================
# Step 3: PPO training
# ============================================================
echo ">>> Step 3: PPO training..."
ACTOR_MODEL_PATH=checkpoints/sft_merged \
CRITIC_MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct \
NUM_GPUS=$NUM_GPUS \
    bash scripts/03_ppo_qwen.sh

# ============================================================
# Step 4: Collect activations from all 3 checkpoints
# ============================================================
echo ">>> Step 4: Collecting activations..."

# Find the latest PPO checkpoint
PPO_CKPT=$(ls -d checkpoints/sae_rl_gsm8k/ppo_qwen2.5_0.5b/global_step_*/actor 2>/dev/null | tail -1)

LAYERS="6 12 18 23"

# Pretrained baseline
python scripts/04_collect_activations.py \
    --model_path Qwen/Qwen2.5-0.5B-Instruct \
    --checkpoint_name pretrained \
    --layers $LAYERS \
    --save_dir data/activations

# Post-SFT
python scripts/04_collect_activations.py \
    --model_path checkpoints/sft_merged \
    --checkpoint_name sft \
    --layers $LAYERS \
    --save_dir data/activations

# Post-PPO
python scripts/04_collect_activations.py \
    --model_path "$PPO_CKPT" \
    --checkpoint_name ppo \
    --layers $LAYERS \
    --save_dir data/activations

# ============================================================
# Step 5: Train SAEs (SAELens BatchTopK)
# ============================================================
echo ">>> Step 5: Training SAEs via SAELens..."
python scripts/05_train_sae_lens.py --checkpoint pretrained
python scripts/05_train_sae_lens.py --checkpoint sft   --model_path checkpoints/sft_merged
python scripts/05_train_sae_lens.py --checkpoint ppo   --model_path "$PPO_CKPT"

# ============================================================
# Step 6: Analyze features
# ============================================================
echo ">>> Step 6: Analyzing features..."
python scripts/06_analyze_features.py \
    --sae_dir checkpoints/saes_lens \
    --activations_dir data/activations \
    --output_dir results/feature_analysis

echo ">>> Pipeline complete! Results in results/feature_analysis/"
