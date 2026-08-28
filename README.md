# A Null-Step Control for SAE Feature Drift Across Fine-Tuning Checkpoints

Code and analysis-ready result tables for the NeurIPS 2026 Workshop on
Interpretability for Discovery submission of the same title.

Fitting a sparse autoencoder (SAE) at each checkpoint of a fine-tuning run and
tracking latent *i* across the chain mixes two sources of change: the model
changes, and the SAE is retrained. This repository measures the second source
with a **null step** — the same SAE fit, warm-started and retrained with the
same recipe, but on activations from the *frozen* base model. On
`Qwen2.5-0.5B-Instruct` during GSM8K fine-tuning (one SFT trajectory, four PPO
conditions), one null step accounts for 85–94% of the decoder drift at the
first transition, and seven null steps for 84–95% across the full trajectory —
while reconstruction quality stays flat and same-index correspondence is
preserved. A replicated real/null comparison at layer 18 shows the arms still
separate (every within-arm pair is more similar than every real–null pair), so
the null step removes a large fitting effect while retaining a smaller,
reproducible difference tied to model change.

## Quick check (for reviewers)

| Paper item | Where in this repo |
|---|---|
| SFT-chain drift curves, flat fidelity (Fig. 1a, Table 1) | `results/decoder_drift.csv`, `results/sae_checkpoint_metrics.csv`; plotted by `scripts/topk/09_plotting.py` |
| Real vs. null drift shares, controlled rerun (Table 2) | produced by re-running `scripts/topk/05_train_sae.py` on frozen-base activations (see *Null step* below); per-seed raw data ships with the paper artifact, not this repo |
| Validation-epoch and reward-informativeness effects (Table 3) | `scripts/topk/05b_sweep_hyperparams.sh` (checkpoint selection), `results/sae_eval_shuffled.csv`, `results/sae_eval_strict.csv` |
| L18 causal/attribution analyses (Appendices) | `results/l18_*.csv` from `scripts/topk/07*.py`, `08*.py` |
| JumpReLU arm (§3.5, App. J) | `scripts/jumprelu/` |
| GSM8K accuracy per chain/step | `results/task_performance.csv` |

## Layout

```
scripts/
  topk/       numbered TopK SAE pipeline (01-10) + end-to-end shell drivers
  jumprelu/   JumpReLU arm: model, training, l0/seed sweeps (see its README)
  analysis/   notebooks for collation metrics and layer-12 grid-search analysis
  sft/        SFT training, evaluation, and checkpoint export (verl)
results/      analysis-ready CSVs (see below)
```

## Pipeline

Scripts in `scripts/topk/` are numbered in execution order and run from the
repository root. `04` onward assume the checkpoints from earlier steps.

| Script | What it does |
|---|---|
| `01_prepare_data.py` | GSM8K → prompt + reward-model format for PPO |
| `04_collect_activations.py` | cache residual-stream activations per checkpoint |
| `05_train_sae.py` | train TopK SAEs (8× expansion, `d_sae=7168`; `k=64`, `k=256` for the L23 robustness run), warm-started along each chain, `--seed` for replication |
| `05b_sweep_hyperparams.sh` | standardise SAE hyperparameters across layers |
| `06_analysis_sae.py` | decoder/encoder drift, feature lifecycle, cross-chain identity |
| `06b`–`06e` | per-SAE NMSE and loss recovery; L18 decoupling probe; KL-sweep comparison; raw MSE and dead latents |
| `07`–`07e` | causal interventions and attribution at L18 |
| `08`–`08c` | name and validate the L18 load-bearing features |
| `09_plotting.py` | figure pipeline |
| `10_build_checkpoint_metrics.py` | build the canonical checkpoint-metrics table |

`run_dense_pipeline.sh` runs the PPO-side pipeline end to end;
`run_sft_pipeline.sh` runs the pure-SFT chain. `scripts/sft/` produced the SFT
trajectory itself (verl `sft_trainer`, 3 epochs / 348 steps, intermediate
checkpoints every 29 steps).

**Null step.** A null chain is not separate code: it is `05_train_sae.py`
re-run with the identical recipe (optimizer, epoch budget, checkpoint-selection
rule, warm start) on activations collected from the *frozen* base model via
`04_collect_activations.py`. Seeded real/null replications pass different
`--seed` values. The controlled-rerun comparison then measures `1 − cos`
between decoder directions at matched chain positions.

## Results

`results/sae_checkpoint_metrics.csv` is the canonical table: **118 rows**, one
per (training regime, chain, checkpoint, layer), carrying raw MSE, NMSE, loss
recovery, mean L0, dead-latent fraction, and the full SAE configuration. The
chains are `sft` (32 rows) and four PPO conditions — `strict`, `flexible`,
`kl0p025`, `shuffled` — plus the `strict_l23_k256` robustness run.

The remaining CSVs group as:

- **drift and identity** — `decoder_drift.csv`, `top_drifted_features*.csv`,
  `feature_lifecycle*.csv`
- **per-chain evaluation** — `sae_eval*.csv`, `sae_mse_dead*.csv`
- **L18 causal work** — `l18_ablation.csv`, `l18_attribution.csv`,
  `l18_loadbearing_ablation*.csv`, `l18_decoupling_probe.csv`, `l18_topk_*.csv`
- **feature naming** — `named_feature_tests.csv`, `flex_named_feature_*.csv`
- **task performance** — `task_performance.csv`, GSM8K accuracy per chain per step

Two reading notes. Raw MSE is comparable only *within* a layer — use NMSE
across layers. And dead-latent fraction, not mean L0, is the informative
sparsity statistic here, because TopK fixes mean L0 by construction.

## Setup

```bash
pip install -r requirements.txt
```

PPO training additionally needs [verl](https://github.com/volcengine/verl)
installed from source; the SFT drivers expect `VERL_DIR` to point at a local
checkout (see `scripts/sft/train_sft.sh`). Large artifacts — activation
tensors, model checkpoints, SAE weights — are not in this repository; they are
regenerable from the pipeline above.

## License

Code: Apache-2.0, matching the `Qwen2.5-0.5B-Instruct` base model. GSM8K is
MIT-licensed, from `openai/grade-school-math`.
