---
license: apache-2.0
library_name: pytorch
language:
  - en
pipeline_tag: feature-extraction
base_model: Qwen/Qwen2.5-0.5B-Instruct
datasets:
  - openai/gsm8k
tags:
  - sparse-autoencoder
  - sae
  - topk-sae
  - interpretability
  - mechanistic-interpretability
  - ppo
  - rlhf
  - reasoning
  - training-dynamics
  - qwen
  - qwen2.5
---

# SAE × RL: Qwen2.5-0.5B on GSM8k (multi-condition)

Warm-start TopK SAEs trained on residual-stream activations of
Qwen2.5-0.5B-Instruct across **four PPO training conditions**. The base
model, training data, and SAE protocol are held fixed across all four;
only the reward function and the KL coefficient vary. Purpose: measure
what RL changes inside the model and which of those changes depend on
the reward signal vs the optimization budget vs the layer architecture.

## Four conditions

| Chain | Reward | KL coef | PPO stages saved | Final PPO step |
|---|---|---:|---|---:|
| **flexible** | last-number-correct (custom) | 0.005 | `instruct_base, ppo_step{10,30,60,100,140,180,200}` | 200 |
| **strict** | verl built-in GSM8k (`#### N` format) | 0.005 | `instruct_base, ppo_step{10,30,50,80,116}` | 116 |
| **kl0p025** | last-number-correct (custom) | **0.025** | `instruct_base, ppo_step{10,30}` | 58 |
| **shuffled** | last-number-correct vs *permuted* ground truth | 0.005 | `instruct_base, ppo_step{10,30}` | (pending) |

Three A/B comparisons run on top of these:

- **flexible vs strict** — same KL coef, different reward shape (form).
- **flexible vs kl0p025** — same reward, 5× the KL coefficient.
- **flexible vs shuffled** — same KL coef, same scoring rule, ground truth
  permuted so the prompt → reward correspondence is broken. Tests whether
  observed reorganization is reward-content-driven or generic PPO pressure.

### What each chain tests

**flexible** is the reference run. The reward is permissive — credit is
given if the last number anywhere in the response matches the answer,
regardless of formatting. Baseline accuracy ≈ 35%; the policy reaches
≈ 70% by step 200. This chain is the longest (8 stages out to step 200)
and is the canonical drift trajectory; everything else is compared to
it. L23 was retrained at K=256 (the other layers stay at K=64) because
L23's NMSE was unacceptably high at K=64; this is the only place the
SAE protocol differs from the other chains.

**strict** is the first A/B partner. The reward function is replaced
with verl's built-in GSM8k scorer, which only credits responses ending
in the canonical `#### N` format. Baseline ≈ 14% (vs flexible's 35%) —
the same model with the same prompts has a ~3× harder reward to satisfy,
so the policy must change more aggressively in early training to hit
nontrivial reward. KL coefficient is held at 0.005 to match flexible.
This chain isolates **what changes when the reward is harder while the
KL budget is identical**. Trained to step 116 (further runs were
deprioritized once the directional findings reproduced).

**kl0p025** is the second A/B partner. The reward function is
identical to flexible, but the KL coefficient is **5× larger** (0.025 vs
0.005). The KL term constrains how far the policy can move from the
reference per step, so a larger coefficient is a stronger leash. This
chain isolates **what changes when the optimization budget tightens
while the reward is identical**. Trained to step 58; SAEs at
`instruct_base, ppo_step10, ppo_step30` are sufficient to compute a
per-step drift rate for comparison against flexible.

**shuffled** is the third A/B partner. The scoring rule is identical to
flexible (last-number match), but the ground truth used by the reward
function is permuted: a deterministic shuffle (seed=0) maps each prompt's
real ground truth to that of a *different* GSM8k question. The reward
magnitude distribution is preserved exactly — every ground-truth value
in the dataset still appears as a target the same number of times — but
the prompt → reward semantic correspondence is broken. KL coefficient is
held at 0.005 to match flexible. This chain isolates **whether the
depth-specific reorganization observed in the other chains is driven by
reward content or by generic PPO pressure** at matched KL budget.
Pre-registered predictions are recorded in the project devlog before
training; the result section will compare against those predictions.

The four-chain design lets each measured effect be classified by which
chains it holds across:
- Holds across **all four** → structural (PPO + this base + this dataset,
  independent of reward shape, reward content, or KL setting in the tested
  range).
- Flips between **flexible and strict** → reward-shape-driven (form of
  the scoring rule matters).
- Flips between **flexible and kl0p025** → KL-budget-driven (optimization
  pressure setting matters).
- Flips between **flexible and shuffled** → reward-content-driven
  (semantic correspondence between prompt and reward target matters).

## Repository layout

```
sae_flexible/                       canonical flexible chain
    sae_{stage}_layer{6,12,18,23}.pt
sae_strict/
    k64/                            full strict chain at k=64 across all layers
        sae_{stage}_layer{6,12,18,23}.pt
    l23_k256/                       L23 retrained at k=256 (matches flexible L23 capacity)
        sae_{stage}_layer23.pt
sae_kl0p025/                        5x KL-coef sweep
    sae_{stage}_layer{6,12,18,23}.pt
sae_shuffled/                       reward-content control (prompt -> gt permuted)
    sae_{stage}_layer{6,12,18,23}.pt
results/                            eval CSVs and analysis outputs (per-chain)
loader.py                           convenience SAE loader
README.md
```

Each stage's SAE is warm-start initialised from the previous stage's
weights, so feature indices align across checkpoints within a chain.
This is what makes per-feature decoder-cosine drift a meaningful
quantity.

## Hyperparameters

### Common across all chains (PPO)

| | |
|---|---|
| Base model | Qwen/Qwen2.5-0.5B-Instruct |
| Advantage estimator | GAE |
| Train / mini / micro batch | 256 / 64 / 8 |
| Actor LR / Critic LR | 1e-6 / 1e-5 |
| Entropy coeff | 0.001 |
| `use_kl_loss` | False (KL is in reward, not loss) |
| Rollout | vLLM, n=8, temperature=1.0 |
| Max prompt / response length | 512 / 512 |
| save_freq / test_freq | 10 / 5 |
| GPUs | 2 |

### Per-chain (PPO)

| Chain | Reward function | KL coef | Total epochs |
|---|---|---:|---:|
| flexible | `reward_gsm8k_flexible.py` (`compute_score`) | 0.005 | 4 |
| strict | verl built-in `openai/gsm8k` scorer | 0.005 | unset (ran to step 116) |
| kl0p025 | `reward_gsm8k_flexible.py` (`compute_score`) | **0.025** | 2 |
| shuffled | `reward_gsm8k_shuffled.py` (`compute_score`) | 0.005 | 2 |

### SAE (TopK with pre-encoder centering, common across all chains)

| | |
|---|---|
| Architecture | TopK SAE + `b_pre` (pre-encoder bias, init = data mean) |
| d_model / d_sae | 896 / 7168 |
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
| Saved checkpoint | best-epoch val MSE |

### K (top-k sparsity) per layer per chain

| Chain | L6 | L12 | L18 | L23 |
|---|---:|---:|---:|---:|
| `sae_flexible/`     | 64 | 64 | 64 | 256 |
| `sae_strict/k64/`   | 64 | 64 | 64 |  64 |
| `sae_strict/l23_k256/` |  — |  — |  — | 256 |
| `sae_kl0p025/`      | 64 | 64 | 64 |  64 |
| `sae_shuffled/`     | 64 | 64 | 64 |  64 |

L23 is at K=256 in `sae_flexible/` and `sae_strict/l23_k256/` — these are
the two chains intended for apples-to-apples L23 comparison. The
`sae_strict/k64/` L23 files are kept for completeness but their L23 NMSE
is much higher (0.28-0.29 vs 0.15-0.18 at K=256) and they should not be
used for cross-chain L23 analysis.

### Activation collection (common)

| | |
|---|---|
| Layers | 6, 12, 18, 23 |
| Max sequence length | 512 |
| Batch size | 16 |
| Tokens per (stage, layer) | 2,000,000 (≈ 445,744 token positions) |
| Dataset | GSM8k train split |
| Train / val split | 356,596 / 89,148 rows, random shuffle, seed=0 |

## Quality (per-chain, NMSE and CE-loss-recovery)

`NMSE = MSE / Var(x)` pooled over val activations (89,148 rows per
(stage, layer)). `frac_rec = (L_mean − L_sae) / (L_mean − L_base)` on
100 GSM8k test prompts: 1 = SAE recovers all the loss a mean ablation
would have lost; 0 = no better than mean. Real-token positions only;
padding pass-through.

### flexible chain (kl=0.005, 8 stages)

| Layer | K | NMSE range | frac_rec range |
|---|---:|---|---|
| 6  |  64 | 0.0004 – 0.0005 | 0.981 – 0.989 |
| 12 |  64 | 0.0007 – 0.0009 | 0.960 – 0.966 |
| 18 |  64 | 0.0034 – 0.0038 | 0.937 – 0.965 |
| 23 | 256 | 0.149  – 0.177  | 0.979 – 0.986 |

### strict chain (kl=0.005, 6 stages)

`sae_strict/k64/`:

| Layer | K | NMSE range | frac_rec range |
|---|---:|---|---|
| 6  |  64 | 0.0003 – 0.0005 | 0.985 – 0.989 |
| 12 |  64 | 0.0007 – 0.0008 | 0.961 – 0.972 |
| 18 |  64 | 0.0032 – 0.0036 | 0.950 – 0.967 |
| 23 |  64 | 0.281  – 0.290  | 0.946 – 0.957 |

`sae_strict/l23_k256/` (L23 only):

| Layer | K | NMSE range | frac_rec range |
|---|---:|---|---|
| 23 | 256 | 0.152 – 0.175 | 0.979 – 0.986 |

### kl0p025 chain (kl=0.025, 3 stages)

| Layer | K | NMSE range | frac_rec range |
|---|---:|---|---|
| 6  |  64 | 0.0004 – 0.0005 | 0.981 – 0.986 |
| 12 |  64 | 0.0007 – 0.0008 | 0.965 – 0.973 |
| 18 |  64 | 0.0033 – 0.0037 | 0.957 – 0.964 |
| 23 |  64 | 0.287  – 0.301  | 0.951 – 0.957 |

L23 here is k=64 (matches strict's k=64 L23, not flexible's k=256).
For apples-to-apples L23 comparison against the flexible chain, use
`sae_strict/l23_k256/` as the strict-chain L23 benchmark. The kl0p025
L23 is appropriate for comparison against `sae_strict/k64/` only.

### shuffled chain (kl=0.005, flexible reward shape with permuted ground truth, 3 stages)

Pending PPO + SAE training. The reward magnitude distribution is identical
to the flexible chain (every ground truth still appears as a target the
same number of times); only the prompt → reward correspondence is broken
via a deterministic permutation seeded at module load. Pre-registered
predictions are recorded in `devlog.md` entry `2026-04-30f`. Update will
follow once the chain trains.

## Cross-chain trends

Decoder-cosine drift (per-step), three observations that hold across
both reward shapes at matched KL=0.005:

- **L23 effective feature count collapses** during PPO (5868 → 4405
  flex, 5855 → 4665 strict). Feature distribution concentrates onto
  fewer directions while every feature stays alive.
- **L6/L12/L18 effective feature count expands** during PPO (+30 to
  +58%). Feature distribution diffuses across more directions.
- **L18 encoder/decoder cosines decouple** at the cold-start transition
  (corr ≈ −0.14 flex, −0.10 strict). Decoder direction is preserved
  while encoder selectivity is rewired — features "die" when their old
  inputs no longer fire them and are "born" when encoder rewiring
  routes new inputs to existing decoder directions.

Cross-chain at different KL coef (flexible vs kl0p025), the prediction
"per-step drift rate scales linearly with KL coef" is **falsified**:
observed kl_high/flex ratios at the 0→10 and 10→30 transitions cluster
in 0.6 – 2.3, never near the predicted 5.0. Drift response is
layer-specific (L6 suppressed by stronger KL, L12 amplified, L18
invariant), not budget-set.

## Loading SAEs

```python
from loader import load_sae, load_chain

# Explicit path
sae, cfg = load_sae("sae_flexible/sae_ppo_step100_layer12.pt", device="cuda")

# Chain / stage / layer (convenience)
sae, cfg = load_chain("flexible", "ppo_step100", 12, root=".", device="cuda")
sae, cfg = load_chain("strict",   "ppo_step116", 23, k=256, root=".")  # → sae_strict/l23_k256/
sae, cfg = load_chain("strict",   "ppo_step116", 23, k=64,  root=".")  # → sae_strict/k64/
sae, cfg = load_chain("kl0p025",  "ppo_step30",  18, root=".")
sae, cfg = load_chain("shuffled", "ppo_step30",  18, root=".")

# Forward
x_hat, z = sae(x)   # x: (N, 896)
```

When splicing SAE reconstruction into the residual stream of the base
model for evaluation, replace only real-token positions:

```python
patched = torch.where(mask.unsqueeze(-1).bool(), sae(h)[0], h)
```

## Files

`sae_{stage}_layer{N}.pt` — SAE state_dict + config (`d_model`, `d_sae`,
`k`, `source`, `source_kind`, `seed`, `init_from_stage`, `best_epoch`,
`best_val_loss`, `selection_metric`).

## Base model

[Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
— d_model=896, 24 transformer layers.

## References

- [verl documentation](https://verl.readthedocs.io)
- [Qwen2.5 model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- TopK SAE: Gao et al. 2024
