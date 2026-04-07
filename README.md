# SAE-RL: Using Sparse Autoencoders to Study RL Training Dynamics

**Central question:** Do RL training dynamics produce interpretable, structured changes in a model's feature space, and can SAEs detect them?

This project trains Qwen2.5-0.5B-Instruct on GSM8k math problems through two phases — supervised fine-tuning (SFT) followed by PPO — then uses SAELens to train Sparse Autoencoders on intermediate checkpoints. By comparing SAE features across the SFT → PPO training trajectory, we ask whether RL produces changes in internal representations that are structured enough to be interpreted.

---

## Motivation

RL fine-tuning demonstrably changes model behavior, but *how* it changes internal representations is poorly understood. Mechanistic interpretability tools like SAEs decompose activations into sparse, human-interpretable features. If RL training induces structured representational change — e.g., sharpening arithmetic features, suppressing uncertainty representations, creating new "answer-checking" circuits — SAEs should be able to detect and characterize those changes.

This project provides a controlled testbed: a fixed dataset (GSM8k), a fixed model family (Qwen2.5-0.5B), a clear reward signal (exact-match on the final answer), and a fixed SAE training procedure applied identically at each checkpoint. The goal is to move from "RL improved solve rate" to "here is *what changed internally* that caused that improvement."

---

## Pipeline

```
Stage 0: Environment setup
Stage 1: SFT on GSM8k           → baseline reasoner + SAE baseline checkpoint
Stage 2: PPO training            → RL trajectory + intermediate checkpoints
Stage 3: Collect activations     → per (checkpoint × layer) activation cache
Stage 4: Train SAEs              → one SAE per (checkpoint × layer)
Stage 5: Feature analysis        → track what changed, interpret why
(Stage 6: SAE regularization)   → stretch goal: add SAE sparsity loss during PPO
```

---

## Stage 0 — Environment Setup

**Purpose:** Validate the full stack before committing to long training runs.

Confirm that `verl`, `SAELens`, and Qwen2.5-0.5B are all working together: load the model, run a GSM8k prompt, check VRAM budget. PPO with actor + critic + vLLM rollout needs ~20–40 GB depending on batch size.

---

## Stage 1 — SFT Warm-Up

**Purpose:** Give the model a baseline ability to produce chain-of-thought math reasoning before PPO.

**Why SFT first?** Base Qwen2.5 may not produce `####`-formatted CoT answers without prompting. More importantly, starting PPO from a model that already reasons in the right format means RL is learning to *refine* an existing behavior — not learning the behavior from scratch. This makes the RL-induced representational changes smaller and more targeted, which is exactly what we want to study with SAEs.

**Scripts:**
```bash
# Prepare SFT data (messages format: user + full CoT assistant turn)
python scripts/01b_prepare_sft_data.py --save_dir data/gsm8k_sft

# Train: 3 epochs, lr=1e-4, LoRA rank 32, FSDP on 2 GPUs
bash scripts/02_sft_qwen.sh 2 checkpoints/sft

# Merge LoRA into flat HF weights for use as PPO actor init
python scripts/02b_merge_lora.py \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --lora_path checkpoints/sft/lora_adapter \
    --output_path checkpoints/sft_merged
```

**Target:** ~40–60% solve rate on GSM8k test before PPO.

**Checkpoint:** `checkpoints/sft_merged/` — this is both the RL starting point and the SAE baseline.

**Status: complete.** Checkpoint at `checkpoints/sft/global_step_87/`, merged weights at `checkpoints/sft_merged/`.

---

## Stage 2 — PPO Training

**Purpose:** Run RL on the SFT checkpoint and collect model snapshots at regular intervals for SAE analysis.

**Why dense checkpointing?** The checkpoints are the primary data source for this project. We need snapshots at early, mid, and late RL training to track how features evolve. A single final checkpoint would tell us *what changed*, but not *when* or *how gradually*.

The reward function is rule-based: the model's response is parsed for a `####` answer token and compared to the ground truth by exact match. This is simple, robust, and not gameable in subtle ways.

Key things to monitor during training:
- **Solve rate** — task performance
- **KL divergence from SFT** — how far the model has drifted behaviorally
- **Response length** — early sign of reward hacking if it spikes
- **Reward hacking** — watch for degenerate outputs that game the reward without actually solving problems

**Script:**
```bash
# Prepare RL data (prompt + reward_model format, different from SFT parquet)
python scripts/01_prepare_data.py --save_dir data/gsm8k

# Run PPO: 15 epochs, saves every 5 epochs
ACTOR_MODEL_PATH=checkpoints/sft_merged \
CRITIC_MODEL_PATH=checkpoints/sft_merged \
NUM_GPUS=2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
VLLM_USE_V1=0 \
bash scripts/03_ppo_qwen.sh
```

**Checkpoints saved to:** `checkpoints/sae_rl_gsm8k/ppo_qwen2.5_0.5b/global_step_*/`

**Checkpoints to keep for SAE analysis:** step 0 (SFT), step 50, step 100, step 200, final.

**Status: in progress** (step ~161/435 as of 2026-03-31).

**Known issues:**
- OOM during vLLM KV cache wake-up after actor updates: set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` and `VLLM_USE_V1=0`
- vLLM v1 memory profiling race on init: fixed by `VLLM_USE_V1=0`
- `data.truncation=error` will crash on sequences > 512 tokens; change to `'right'` to truncate silently

---

## Stage 3 — Collect Activations

**Purpose:** Cache residual stream activations from each checkpoint on a fixed set of GSM8k prompts.

**Why fixed prompts?** We want to compare the *same inputs* across different model checkpoints. If we used different prompts per checkpoint, we couldn't distinguish "this feature changed" from "this feature is activated differently because the prompt was different."

Activations are collected at four layers: 6, 12, 18, 23 (early, mid-early, mid-late, final). This gives a multi-layer picture of where in the network RL-induced changes are concentrated.

**Script:**
```bash
# Run once per checkpoint: pretrained, sft, and each PPO checkpoint
python scripts/04_collect_activations.py \
    --model_path <checkpoint_path> \
    --checkpoint_name <pretrained|sft|ppo_stepN> \
    --layers 6 12 18 23 \
    --save_dir data/activations
```

Output: one `.pt` tensor per (checkpoint × layer), e.g. `data/activations/sft_layer12.pt`. Shape: `(n_prompts, d_model)` — mean-pooled over sequence length.

---

## Stage 4 — Train SAEs

**Purpose:** Train a Sparse Autoencoder on activations from each (checkpoint, layer) pair, producing a comparable set of SAEs across the RL trajectory.

**Why train separate SAEs?** The SAE for each checkpoint learns a dictionary of features that best sparsely reconstruct *that checkpoint's* activations. By comparing the dictionaries across checkpoints — which features appear, disappear, drift, or sharpen — we can characterize what RL training actually changed in the model's internal representations.

The key constraint is **identical hyperparameters** across all SAEs. If we change the expansion factor or sparsity level between checkpoints, differences in the learned features could be artifacts of the training rather than genuine representational change.

SAE architecture: **BatchTopK** (SOTA; outperforms L1 penalty for dead-feature handling). d_model=896, expansion=8 → d_sae=7168, k=32 (mean active features per token).

**Script (SAELens — recommended):**
```bash
python scripts/05_train_sae_lens.py --checkpoint pretrained
python scripts/05_train_sae_lens.py --checkpoint sft   --model_path checkpoints/sft_merged
python scripts/05_train_sae_lens.py --checkpoint ppo   --model_path <ppo_checkpoint_path>
```

**Script (hand-rolled TopK, no SAELens dependency):**
```bash
python scripts/05_train_sae.py \
    --activations_dir data/activations \
    --save_dir checkpoints/saes \
    --expansion_factor 8 --k 32
```

**Monitor per SAE:** L0 (average active features — should be ~32), reconstruction loss, explained variance.

> **Note:** `SAELENS_PATH` is currently hardcoded to a Windows path in `05_train_sae_lens.py` and `06_analyze_features.py`. Update this if SAELens is not pip-installed.

---

## Stage 5 — Feature Analysis

**Purpose:** Identify which features change during RL and characterize what they represent. This is the core science of the project.

**Three analyses:**

**1. Feature activation frequency** — which features become more or less active after RL? Features that jump from near-zero frequency to high frequency are candidates for "RL-induced features."

**2. Feature geometry / drift** — compute cosine similarity between feature directions (decoder weight columns) across SAEs from different checkpoints. Low cosine similarity means the feature direction has meaningfully rotated in the residual stream — not just changing in frequency, but in *what concept it represents*.

**3. Dead / born / stable features** — count features that were inactive pre-RL but become active after (born), and vice versa (died). A large number of born features suggests RL is creating genuinely new representational structure, not just reweighting existing features.

**Script:**
```bash
python scripts/06_analyze_features.py \
    --sae_dir checkpoints/saes_lens \
    --activations_dir data/activations \
    --output_dir results/feature_analysis
```

Output: frequency histograms and drift plots per layer, plus a printed table of born/died/stable/always-dead feature counts across transitions (pretrained→sft, sft→ppo).

**Follow-up (manual):** For the features that change most, collect top-activating examples and inspect whether they correspond to interpretable concepts — e.g., "arithmetic carry operation," "final answer token," "uncertainty," "step-by-step enumeration." Auto-interpretation via Claude API is also supported.

---

## Stage 6 — SAE Regularization (Stretch Goal)

**Purpose:** Test whether explicitly optimizing for SAE sparsity *during* PPO training produces more interpretable or better-performing models.

**Hypothesis:** Adding an auxiliary loss — the L1 sparsity of activations as measured by a *frozen* SAE (trained on the SFT checkpoint) — may encourage the model to develop sparser, more structured internal representations during RL. This could make RL-induced features more interpretable and possibly improve generalization.

**Experimental design:**
- Baseline: standard PPO
- Treatment: PPO with auxiliary sparsity loss `L_total = L_PPO + λ * L_sparsity`
- Sweep λ ∈ {0, 1e-4, 1e-3}
- Compare: final solve rate, SAE reconstruction quality, feature interpretability (qualitative + auto-interp)

---

## Evaluation Metrics

| Metric | What it measures | When |
|--------|-----------------|------|
| GSM8k solve rate | Task performance | Every checkpoint |
| KL from SFT | Behavioral drift from base | Every PPO step |
| SAE L0 (avg active features) | Representation sparsity | Per SAE |
| SAE explained variance | Reconstruction quality | Per SAE |
| Feature cosine similarity across checkpoints | Feature stability / change | Across all SAEs |
| Top-activating example quality | Interpretability | Manually, for changed features |

---

## Current Status

| Stage | Status |
|-------|--------|
| 0 — Environment setup | Complete |
| 1 — SFT | Complete — `checkpoints/sft_merged/` |
| 2 — PPO | In progress — step ~161/435 |
| 3 — Collect activations | Not started |
| 4 — Train SAEs | Not started |
| 5 — Feature analysis | Not started |
| 6 — SAE regularization | Not started |

---

## Risks

| Risk | Mitigation |
|------|------------|
| Model doesn't improve on GSM8k with PPO | Use Qwen2.5-1.5B-Instruct instead |
| SAEs across checkpoints aren't comparable | Fix all SAE hyperparameters; use identical prompts |
| Features don't change meaningfully | Look at later layers; run a longer PPO run |
| VRAM insufficient | Enable gradient checkpointing; reduce batch size |
| Reward hacking / length exploitation | Add length penalty; monitor output length |

---

## Directory Layout

```
SAE_RL/
  data/
    gsm8k/            # PPO parquets (prompt + reward_model)
    gsm8k_sft/        # SFT parquets (messages)
    activations/      # Cached residual stream tensors — {checkpoint}_layer{N}.pt
  checkpoints/
    sft/              # verl FSDP SFT checkpoint shards
    sft_merged/       # Merged flat HF weights (actor init for PPO)
    sae_rl_gsm8k/     # PPO checkpoints (verl layout, global_step_*)
    saes_lens/        # SAELens SAEs — {checkpoint}_layer{N}/
    saes/             # Hand-rolled SAEs (if using 05_train_sae.py)
  results/
    feature_analysis/ # Plots + lifecycle tables from stage 5
  scripts/
    01_prepare_data.py        # GSM8k → PPO parquet
    01b_prepare_sft_data.py   # GSM8k → SFT parquet
    02_sft_qwen.sh            # Stage 1: SFT
    02b_merge_lora.py         # Stage 1: merge LoRA
    03_ppo_qwen.sh            # Stage 2: PPO
    04_collect_activations.py # Stage 3: cache activations
    05_train_sae_lens.py      # Stage 4: SAELens BatchTopK SAEs
    05_train_sae.py           # Stage 4: hand-rolled TopK SAEs
    06_analyze_features.py    # Stage 5: feature comparison
    run_all.sh                # Full pipeline reference script
  research_plan.md    # Detailed research plan with hypotheses and design rationale
  requirements.txt
```

---

## Model

**Qwen/Qwen2.5-0.5B-Instruct** — d_model=896, 24 transformer layers. SAEs target layers 6, 12, 18, 23.

## References

- [verl documentation](https://verl.readthedocs.io)
- [SAELens (decoderesearch)](https://github.com/decoderesearch/SAELens)
- [Qwen2.5 model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- See `research_plan.md` for full design rationale, hypotheses, and references to the SAE regularization paper.
