# Studying Reinforcement Learning with Sparse Autoencoders

Using Sparse Autoencoders (SAEs) to reverse-engineer what RL training actually changes inside a language model. The testbed is Qwen2.5-0.5B fine-tuned on GSM8k math reasoning, with SAEs trained on residual-stream activations at multiple checkpoints along the RL trajectory.

## Why This Matters

RL fine-tuning demonstrably changes model behavior, but *how* it changes internal representations is poorly understood. We designed a controlled experiment — fixed dataset (GSM8k), fixed model (Qwen2.5-0.5B), fixed SAE architecture (BatchTopK) — so that any differences in learned features across checkpoints reflect genuine representational change, not artifacts of the setup.

The goal: move from "RL improved solve rate" to "here is *what changed internally* that caused that improvement."

## Experimental Pipeline

The project proceeds in six stages. Stages 0–2 are complete or in progress; Stages 3–5 begin once PPO training finishes.

### Stage 1 — Environment Setup 

Validated that `verl`, `SAELens`, and Qwen2.5-0.5B work together end-to-end. Confirmed VRAM budget for PPO with actor + critic + vLLM rollout on 2 GPUs.


### Stage 2 — PPO Training 

We run PPO on the SFT checkpoint with dense checkpointing — snapshots at early, mid, and late training — because the evolution of features over time is the primary data source for this project. A single final checkpoint would tell us *what* changed but not *when* or *how gradually*.

The reward is rule-based: parse the model's response for a `####` answer token and compare to ground truth by exact match.

**Current status:** Step ~161/435 as of 2026-03-31.

**Checkpoints saved to:** `checkpoints/sae_rl_gsm8k/ppo_qwen2.5_0.5b/global_step_*/`

**Checkpoints selected for SAE analysis:** step 0 (SFT), step 50, step 100, step 200, final.

**Monitoring:** solve rate, KL divergence from SFT (behavioral drift), response length (reward hacking signal).

### Stage 3 — Collect Activations

We will cache residual-stream activations from each checkpoint on a *fixed* set of GSM8k prompts. Using identical prompts across checkpoints is critical — otherwise we can't distinguish "this feature changed" from "the prompt changed."

Activations will be collected at layers 6, 12, 18, and 23 (early → final) to map where in the network RL-induced changes concentrate. Each activation is mean-pooled over sequence length, yielding one tensor per (checkpoint × layer).

### Stage 4 — Train SAEs

We will train a BatchTopK SAE on each (checkpoint, layer) pair with *identical hyperparameters* throughout — d_model=896, expansion factor 8 (d_sae=7168), k=32 active features. Identical setup is essential: if we change the sparsity level between checkpoints, differences in learned features could be training artifacts rather than genuine representational change.

Two implementations are available: one using SAELens (recommended), one hand-rolled with no external SAE dependency.

> **Note:** `SAELENS_PATH` is currently hardcoded to a Windows path in `05_train_sae_lens.py` and `06_analyze_features.py`. Update this if SAELens is not pip-installed.

### Stage 5 — Feature Analysis

This is the core science. We will run three analyses:

**Feature activation frequency** — which features become more or less active after RL? Features that jump from near-zero to high frequency are candidates for "RL-induced features."

**Feature geometry / drift** — cosine similarity between feature directions (decoder weight columns) across checkpoints. Low similarity means a feature has meaningfully rotated in the residual stream — not just changing in frequency, but in *what it represents*.

**Dead / born / stable features** — features inactive pre-RL that become active after (born), and vice versa (died). Many born features would suggest RL creates genuinely new representational structure rather than reweighting existing features.

For the most-changed features, we will collect top-activating examples and inspect whether they correspond to interpretable concepts (e.g., "arithmetic carry operation," "final answer token," "uncertainty"). Auto-interpretation via Claude API is also supported.

### Stage 6 — SAE Regularization 

We plan to test whether adding SAE sparsity as an auxiliary loss *during* PPO produces more interpretable representations. The idea: a frozen SAE (trained on SFT activations) measures the L1 sparsity of the model's activations, and this becomes a regularization term in the PPO loss.

Experimental design: sweep λ ∈ {0, 1e-4, 1e-3}, compare final solve rate, SAE reconstruction quality, and feature interpretability.

## Evaluation

| Metric | Purpose |
|--------|---------|
| GSM8k solve rate | Task performance per checkpoint |
| KL from SFT | Behavioral drift during RL |
| SAE L0 | Representation sparsity (target: ~32 active features) |
| SAE explained variance | Reconstruction quality |
| Feature cosine similarity | Feature stability/change across checkpoints |
| Top-activating examples | Interpretability of changed features |

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Model doesn't improve with PPO | Fall back to Qwen2.5-1.5B-Instruct |
| SAEs across checkpoints aren't comparable | All SAE hyperparameters fixed; identical prompt set |
| Features don't change meaningfully | Focus on later layers; extend PPO training |
| VRAM insufficient | Gradient checkpointing; reduce batch size |
| Reward hacking / length exploitation | Add length penalty; monitor output length |

## Model

**Qwen/Qwen2.5-0.5B-Instruct** — d_model=896, 24 transformer layers. SAEs target layers 6, 12, 18, 23.

## Repository Structure

```
SAE_RL/
  data/
    gsm8k/            # PPO parquets (prompt + reward_model)
    gsm8k_sft/        # SFT parquets (messages)
    activations/      # Cached residual-stream tensors — {checkpoint}_layer{N}.pt
  checkpoints/
    sft/              # verl FSDP SFT checkpoint shards
    sft_merged/       # Merged HF weights (PPO actor init + SAE baseline)
    sae_rl_gsm8k/     # PPO checkpoints (global_step_*)
    saes_lens/        # SAELens SAEs — {checkpoint}_layer{N}/
    saes/             # Hand-rolled SAEs (alternative)
  results/
    feature_analysis/ # Plots + lifecycle tables from Stage 5
  scripts/
    01_prepare_data.py        # GSM8k → PPO parquet
    01b_prepare_sft_data.py   # GSM8k → SFT parquet
    02_sft_qwen.sh            # SFT training
    02b_merge_lora.py         # LoRA merge
    03_ppo_qwen.sh            # PPO training
    04_collect_activations.py # Activation caching
    05_train_sae_lens.py      # SAELens SAE training
    05_train_sae.py           # Hand-rolled SAE training
    06_analyze_features.py    # Feature comparison analysis
    run_all.sh                # Full pipeline reference
  research_plan.md
  requirements.txt
```

## References

- [verl documentation](https://verl.readthedocs.io)
- [SAELens (decoderesearch)](https://github.com/decoderesearch/SAELens)
- [Qwen2.5 model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- See `research_plan.md` for full design rationale, hypotheses, and references to the SAE regularization paper.
