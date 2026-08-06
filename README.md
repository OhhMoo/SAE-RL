# SAE × RL — sparse autoencoders across SFT and PPO checkpoints

Training and analysis code for fitting TopK sparse autoencoders to the residual stream of
`Qwen2.5-0.5B-Instruct` at a sequence of GSM8K fine-tuning checkpoints, and measuring how
the learned dictionary moves as the model trains.

Five checkpoint chains at layers 6, 12, 18 and 23: one supervised fine-tuning trajectory and
four PPO conditions that vary reward design or KL regularisation, including a label-shuffled
control that takes gradient of comparable magnitude while carrying no task signal.

## Where things live

This repository is **code plus analysis-ready tables**. The heavy artifacts are elsewhere:

| | Where |
|---|---|
| Pipeline code, result CSVs | here |
| SAE weights (all chains, all layers) | Hugging Face `OhhMoo/sae-rl-qwen05b-layers` |
| SFT model checkpoints | Hugging Face `OhhMoo/qwen05b-gsm8k-sft-instruct` |
| Activation tensors, PPO checkpoints | not published — regenerable from `scripts/01`–`04` |
| The write-up | a separate NeurIPS workshop draft, not in this repository |

If you arrived looking for a model card, it is on the Hugging Face repository above. The
model card used to sit in this file by mistake; it described a directory layout (`sae_sft/`,
`loader.py`) that only ever existed there.

## Pipeline

Scripts are numbered in execution order. `04` onward assume the checkpoints from earlier steps.

| | |
|---|---|
| `01_prepare_data.py` | GSM8K → prompt + reward-model format for PPO |
| `04_collect_activations.py` | cache residual-stream activations per checkpoint |
| `05_train_sae.py` | train TopK SAEs (8× expansion, `d_sae=7168`; `k=64`, and `k=256` for the L23 robustness run) |
| `05b_sweep_hyperparams.sh` | standardise SAE hyperparameters across layers |
| `06_analysis_sae.py` | decoder drift, feature lifecycle, cross-chain identity |
| `06b_eval_sae.py` | normalised MSE and delta loss, one row per SAE |
| `06c`–`06e` | L18 encoder/decoder decoupling probe; KL-sweep comparison; raw MSE and dead latents |
| `07_ablation_sae.py`, `07b`–`07e` | causal interventions and attribution at L18 |
| `08_topk_contexts_l18.py`, `08b`, `08c` | name and validate the L18 load-bearing features |
| `09_plotting.py` | figure pipeline |
| `10_build_checkpoint_metrics.py` | build the canonical checkpoint-metrics table |

`run_dense_pipeline.sh` runs the PPO-side pipeline end to end; `run_sft_pipeline.sh` runs the
pure-SFT chain. `scripts/sft/` holds the SFT training, evaluation and upload code that
produced the SFT trajectory.

## Results

`results/sae_checkpoint_metrics.csv` is the canonical table: **118 rows**, one per
(training regime, chain, checkpoint, layer), carrying raw MSE, NMSE, loss recovery, mean
L0, dead-latent fraction and the full SAE configuration.

The remaining CSVs group as:

- **drift and identity** — `decoder_drift.csv`, `top_drifted_features*.csv`,
  `feature_lifecycle*.csv`
- **per-chain evaluation** — `sae_eval*.csv`, `sae_mse_dead*.csv`
- **L18 causal work** — `l18_ablation.csv`, `l18_attribution.csv`,
  `l18_loadbearing_ablation*.csv`, `l18_decoupling_probe.csv`, `l18_topk_*.csv`
- **feature naming** — `named_feature_tests.csv`, `flex_named_feature_*.csv`
- **task performance** — `task_performance.csv`, GSM8K accuracy per chain per step

Two reading notes that have caused confusion before. Raw MSE is comparable only *within* a
layer — use NMSE across layers. And dead-latent fraction, not mean L0, is the sparsity
statistic that carries information here, because TopK fixes mean L0 by construction.

## Setup

```bash
pip install -r requirements.txt   # verl is installed separately: pip install -e ../verl
```

## Layout

```
scripts/     numbered pipeline + sft/ training code
results/     analysis-ready CSVs
_archive/    superseded and bulky material, gitignored (see _archive/README.md)
```

`_archive/` is not published. It holds the exploratory figure set, cached density tensors,
run logs and an earlier ACL-format draft — kept on the working machine, removed from this
page so that what remains is the part that still stands.

## License

Code: Apache-2.0, matching the `Qwen2.5-0.5B-Instruct` base model. GSM8K is MIT-licensed,
from `openai/grade-school-math`.
