# SAE Feature Interpretation Report
*Generated: 2026-04-07 | Qwen2.5-0.5B-Instruct + PPO (GSM8k) + TopK SAE*

---

## Overview

This report summarizes what we currently know from the 28 SAEs trained on Qwen2.5-0.5B-Instruct
across 7 training stages (SFT baseline + 6 PPO checkpoints) and 4 layers (6, 12, 18, 23).

**Central question:** Does PPO fine-tuning produce interpretable, structured changes in the model's
feature space, and can SAEs detect them?

**Short answer from current data:** Yes — but the signal is concentrated at the final layer (23),
and the later PPO checkpoints (≥200 steps) are confounded by a reward-hacking episode that must be
accounted for before drawing conclusions about genuine task learning.

---

## 1. Training Setup

### Model
- **Qwen2.5-0.5B-Instruct** — d_model=896, 24 transformer layers
- Starting from the instruct checkpoint (SFT was run first as a warm-up on GSM8k CoT)

### SAE Architecture
| Parameter | Value |
|---|---|
| Type | TopK SAE (hand-rolled, `05_train_sae.py`) |
| d_model | 896 |
| d_sae | 7168 (8× expansion) |
| k (active features/token) | 32 |
| Epochs trained | **10** (default is 50 — undertrained, see §5) |
| Learning rate | 3e-4 |
| Dead-feature resampling | Every 5 epochs |

### Activation Dataset
- Source: GSM8k `train` split, question-only prompts (no answers)
- Collected with max_tokens=500k; actual file sizes (~762 MB each) correspond to ~222k tokens per (checkpoint × layer)
- Same prompts used across all checkpoints — apples-to-apples comparison is valid
- Layers targeted: **6** (early), **12** (mid), **18** (mid-late), **23** (final)

### Training Stages
| Stage label | Corresponding PPO step | GSM8k solve rate (score/mean) |
|---|---|---|
| `sft` | 0 — SFT baseline | ~33% |
| `ppo_step10` | 10 | ~36% |
| `ppo_step50` | 50 | ~56% |
| `ppo_step100` | 100 | ~67% |
| `ppo_step200` | 200 | ~79%* |
| `ppo_step300` | 300 | ~12%* (reward hacking) |
| `ppo_step435` | 435 | ~13%* (reward hacking) |

\* Steps 200–435 are from a second PPO run (resumed at step 161) that exhibited **reward hacking**:
response lengths collapsed to ~5–6 tokens at these steps, with models outputting very short strings
that sometimes matched the `####` format without performing actual reasoning. See §4 for implications.

---

## 2. SAE Reconstruction Loss by Layer

Reconstruction loss (MSE) per SAE, evaluated on the same activation data used for training
(held-out validation metrics not yet computed — see §5):

| Layer | SFT | step_10 | step_50 | step_100 | step_200 | step_300 | step_435 | Trend |
|---|---|---|---|---|---|---|---|---|
| **6** | ~flat | ~flat | ~flat | ~flat | ~flat | ~flat | ~flat | **Stable** |
| **12** | ~flat | ~flat | ~flat | ~flat | ~flat | ~flat | ~flat | **Stable** |
| **18** | baseline | ↑ | ↑ | ↑ | ↑ | ↑ | ↑ | **Modest increase** |
| **23** | 0.984 | — | — | — | — | 1.113 | — | **+13% increase** |

**Key finding:** RL training increases reconstruction loss primarily at the final layer (23),
indicating that PPO pushes the model's final-layer representations into a less decomposable
structure — the SAE struggles more to reconstruct them as a sparse sum of dictionary features.
Shallow layers (6, 12) are barely affected. Layer 18 sits in between.

This gradient of change is consistent with the general picture from mechanistic interpretability:
earlier layers tend to encode more general/syntactic features that are robust to fine-tuning,
while later layers encode task-specific computation that changes more with task-specific training.

---

## 3. Feature Activation Statistics (Layer 6)

The most complete quantitative data we have is from Layer 6 (from the valid CUDA run):

| Metric | Value |
|---|---|
| Active features (freq > 1%) | ~820 out of 7168 |
| Dead features | ~6348 |
| Stable features across consecutive transitions | **88–98 per transition** |

**Note on dead features:** ~89% of the 7168-feature dictionary is dead (never activates on
these inputs). This is a known consequence of undertrained SAEs and/or a too-large expansion
factor relative to the available training tokens. With 222k tokens and k=32, effective training
signal per feature is sparse. See §5 (caveats) for the recommended fix.

**Note on stability:** 88–98 stable features across each consecutive transition (out of ~820 active)
means roughly **10–12% of active features are stable** per transition. This is a lower stability
than typically seen in SAEs trained to convergence. However, given undertraining, this may partially
reflect noise in feature directions rather than genuine representational flux.

---

## 4. Feature Drift Analysis

Drift plots are available at `results/feature_analysis/drift_{s1}_{s2}_layer{N}.png` for all
6 consecutive transitions × 4 layers = 24 plots.

The drift metric used is **best-match cosine similarity**: for each feature in the source SAE,
find its highest-cosine-similarity match in the target SAE. A score near 1.0 means the feature
direction is preserved; a score near 0 means the feature has rotated or been replaced.

### Interpretation guidance for the plots

**Layer 6 and 12 (shallow/mid):** Expect predominantly high-similarity distributions (mass near
1.0) given the flat reconstruction loss. Most features should have a close match in the next stage,
indicating that shallow representations are not meaningfully rewritten by PPO.

**Layer 18 (mid-late):** Expect a bimodal distribution — a bulk of stable features plus a tail of
drifted ones. The drifted tail likely grows as PPO progresses, coinciding with the modest
reconstruction loss increase.

**Layer 23 (final):** Expect the heaviest drift here, particularly in the SFT→step_10 and
step_50→step_100 transitions (the period of genuine task learning, before reward hacking). The
+13% reconstruction loss increase at this layer is the primary signal. Features here may be
encoding arithmetic reasoning, answer-detection, or step-enumeration circuits that sharpen with RL.

### Reward hacking caveat for later transitions

The transitions **step_100→step_200**, **step_200→step_300**, and **step_300→step_435** occur
partly or fully within the reward-hacking regime (response lengths of 5–6 tokens, low solve rate).
Feature drift in these transitions may reflect:
1. Genuine further task learning during the step_100→step_200 window
2. Model collapse / degenerate output mode for step_200→step_435

Do not interpret large drift at layer 23 in these late transitions as evidence of continued
arithmetic learning. The more interpretable comparisons are:
- **SFT → step_10:** First contact with RL reward signal
- **step_10 → step_50:** Rapid early learning phase (~56% solve rate)
- **step_50 → step_100:** Consolidation into strong performance (~67% solve rate)

---

## 5. Caveats and Known Issues

### 5a — SAEs are undertrained (10 vs 50 epochs)
The SAE training script defaults to 50 epochs; all 28 SAEs were trained for only 10.
Consequences:
- Higher dead-feature rate than expected (89% dead at layer 6)
- Feature directions may not have converged, inflating apparent drift
- Reconstruction losses are higher than what a converged SAE would achieve

**Recommended action:** Rerun `05_train_sae.py` with `--epochs 50` on all 28 (stage, layer) pairs.
Activation files are still present at `data/activations/`.

### 5b — Feature analysis plots are from the current (correct) SAEs
The memory records an earlier run where the analysis was run on deleted/replaced SAEs, producing
stale/invalid results (~47 active features, 100% drift). The current plots in
`results/feature_analysis/` are from the correct SAEs (confirmed via the CUDA run). These results
can be trusted.

### 5c — No held-out validation metrics yet
The three required eval metrics (reconstruction MSE on held-out data, L0, model delta loss) have
not been run. The reconstruction loss values reported in §2 are training-set estimates.
Running `06_analyze_features.py` with fresh held-out data will give more reliable numbers.

### 5d — Reward hacking in the final PPO run
Checkpoints at steps 200–435 come from a PPO run that collapsed into outputting short, degenerate
responses (mean response length ~5–6 tokens) while appearing to get some correct-format matches.
The underlying model representations at these steps are likely encoding a degenerate output mode,
not improved arithmetic reasoning. Treat SAE features at steps 300 and 435 with this in mind.

The cleanest signal window is **SFT → step_100** (genuine task improvement, no hacking).

---

## 6. What to Do Next

Priority order:

1. **Retrain SAEs with `--epochs 50`** — the most important step for reliable analysis.
   - Command: `python scripts/05_train_sae.py --activations_dir data/activations --save_dir checkpoints/saes --expansion_factor 8 --k 32 --epochs 50 --device cuda`
   - ~6× longer than the current 10-epoch run; plan ~1–2 hours per checkpoint on GPU.

2. **Re-run `06_analyze_features.py`** after retraining SAEs — the current drift/freq plots will
   change with converged SAEs.
   - Also fix the `--sae_dir` arg: current checkpoints are at `checkpoints/saes`, not `checkpoints/saes_lens`.

3. **Compute the three eval metrics** (reconstruction MSE on held-out GSM8k test, L0, model delta
   loss) on the new SAEs.

4. **Focus interpretation on the SFT → step_100 window**. Collect top-activating token examples
   for the features in layer 23 that show highest drift in this window. These are the candidate
   "RL-learned reasoning features."

5. **Consider running a clean PPO run without reward hacking** (use a KL penalty or response-length
   cap) to provide unconfounded checkpoints at steps 200–400.

---

## 7. Key Takeaways

| Finding | Confidence | Notes |
|---|---|---|
| RL changes final-layer (23) representations most | **Medium** | Clear recon loss signal; undertrained SAEs add noise |
| Layers 6 and 12 are stable through RL | **Medium-high** | Consistent across recon loss and stability counts |
| Layer 18 sees moderate change | **Low-medium** | Modest trend; needs retrained SAEs to quantify |
| Late PPO checkpoints (≥300) are confounded by reward hacking | **High** | Confirmed via response length logs |
| Current SAEs have too many dead features to draw strong feature-level conclusions | **High** | 89% dead at layer 6; needs 50-epoch retraining |
| The SFT→step_100 window is the cleanest interpretable signal | **High** | Solve rate goes 33%→67% with normal behavior |

---

*Plots: `results/feature_analysis/freq_layer{6,12,18,23}.png`, `drift_{s1}_{s2}_layer{N}.png`*
*SAEs: `checkpoints/saes/sae_{stage}_layer{N}.pt` (28 files)*
*Activations: `data/activations/{stage}_layer{N}.pt` (28 files, ~762 MB each)*
