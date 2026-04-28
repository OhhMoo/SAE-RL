---
license: apache-2.0
library_name: pytorch
base_model: Qwen/Qwen2.5-0.5B-Instruct
tags:
  - sparse-autoencoder
  - interpretability
  - mechanistic-interpretability
  - topk-sae
  - qwen2.5
  - ppo
  - rlhf
---

# SAE × RL: Qwen2.5-0.5B on GSM8k

Warm-start TopK SAEs trained on 8 PPO checkpoints × 4 residual-stream layers of Qwen2.5-0.5B-Instruct fine-tuned on GSM8k. Purpose: measure what RL changes inside the model.

## What I did

1. **PPO** on Qwen2.5-0.5B-Instruct on GSM8k with a flexible reward (last number in the response = correct). 2 GPUs, `verl` FSDP + vLLM rollout. Checkpointed every 10 steps; ran to step 200.
2. **Collected residual-stream activations** at layers {6, 12, 18, 23} from 8 stages — `instruct_base` and `ppo_step{10,30,60,100,140,180,200}` — 2M tokens per (stage, layer).
3. **Trained 32 warm-start TopK SAEs** (8 stages × 4 layers). Each stage's SAE is initialised from the previous stage's weights, so feature indices align across checkpoints and decoder-cosine drift is meaningful.
4. **Evaluated** SAE quality on a held-out 20% activation split (seeded shuffle): normalized MSE (`MSE / Var(x)`) and fraction of CE-loss recovered on GSM8k test prompts (`(L_mean − L_sae) / (L_mean − L_base)`).

## Hyperparameters

### PPO
| | |
|---|---|
| Base model | Qwen/Qwen2.5-0.5B-Instruct |
| Advantage estimator | GAE |
| Train / mini / micro batch | 256 / 64 / 8 |
| Actor LR / Critic LR | 1e-6 / 1e-5 |
| Entropy coeff | 0.001 |
| KL penalty (in reward) | fixed, kl_coef=0.005 |
| `use_kl_loss` | False |
| Rollout | vLLM, n=8, temp=1.0 |
| Max prompt / response length | 512 / 512 |
| Epochs | 4 |
| save_freq / test_freq | 10 / 5 |
| GPUs | 2 |
| Reward | flexible (last number correct) |

### SAE (TopK with pre-encoder centering)
| | |
|---|---|
| Architecture | TopK SAE + `b_pre` (pre-encoder bias, init = data mean) |
| d_model / d_sae / K | 896 / 7168 / 64 |
| Expansion factor | 8× |
| Epochs / LR / Batch | 20 / 1e-4 / 512 |
| Optimizer | Adam |
| LR schedule | Cosine annealing → lr/10 |
| Gradient clip | max_norm=1.0 |
| Dead resample | every 10 epochs, threshold 1e-4 |
| Aux-k loss coeff | 1/32 |
| Decoder constraint | unit-norm projection after every step |
| Warm-start init | previous stage's SAE weights |
| Train / val split | 80 / 20 random shuffle, seed=0 |
| Saved checkpoint | best-epoch val MSE (held-out 20%) |

### Warm-start chain
`instruct_base → ppo_step10 → ppo_step30 → ppo_step60 → ppo_step100 → ppo_step140 → ppo_step180 → ppo_step200`

### Activation collection
| | |
|---|---|
| Layers | 6, 12, 18, 23 |
| Max sequence length | 512 |
| Batch size | 16 |
| Tokens per (stage, layer) | 2,000,000 (≈ 445,744 token positions) |
| Dataset | GSM8k train split |
| Train / val split | 356,596 / 89,148 rows, random shuffle with seed=0 |

## Quality

Evaluated on the held-out 20% of activations (89,148 rows per (stage, layer)) and 100 GSM8k test prompts.

- **NMSE** = `MSE / Var(x)` pooled over all elements of the val activations. Lower is better; 0 = perfect reconstruction.
- **frac_rec** = `(L_mean − L_sae) / (L_mean − L_base)` where `L_base` is the model's CE on GSM8k test prompts untouched, `L_sae` is the CE with `sae(x)` spliced in at the layer (real-token positions only; padding pass-through), and `L_mean` is the CE with the layer's real-token mean spliced in (control). 1 = SAE recovers all the loss that a mean ablation would have lost; 0 = no better than the dataset mean.

| Layer | NMSE (range across 8 stages) | frac_rec (range) |
|---|---|---|
| 6  | 0.0004 – 0.0005 | 0.981 – 0.989 |
| 12 | 0.0007 – 0.0009 | 0.960 – 0.967 |
| 18 | 0.0034 – 0.0038 | 0.937 – 0.965 |
| 23 | 0.288 – 0.305   | 0.946 – 0.959 |

Layer 23 has noticeably higher NMSE than other layers, but mean ablation at L23 destroys the model (L_mean ≈ 15–20 nats vs `L_base` ≈ 2.4–3.2 nats), so frac_rec stays at 0.95+ — the unrecovered variance lives in low-importance directions for the LM head. Layer 18 frac_rec drifts down ~3pp across PPO training (0.965 at step10 → 0.937 at step200) while NMSE stays flat; layers 6 and 12 are stable across the chain.

## Files

- `sae_{stage}_layer{N}.pt` — SAE state_dict + config (`d_model`, `d_sae`, `k`, `source`, `source_kind`, `seed`, `init_from_stage`, `best_epoch`, `best_val_loss`, `selection_metric`).

## Base model

[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) — d_model=896, 24 transformer layers.
