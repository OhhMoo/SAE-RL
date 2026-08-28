# JumpReLU SAE arm (scripts/jumprelu/)

Companion arm to the TopK pipeline (`scripts/topk/05_train_sae.py` etc.),
evaluating the same layers with a JumpReLU parameterization instead of hard
top-k selection. In the paper this arm supports Section 3.5 and Appendix J:
the null-step comparison is repeated under a learned-threshold architecture to
check that the retraining-drift effect is not specific to a hard TopK budget.
The arm also produced the layer-depth fidelity measurements behind the paper's
decision to exclude layer 23 from the main comparison (Limitations).

## Files

- `jumprelu_model.py` -- `JumpReLUSAE` (log-threshold, straight-through
  estimator, unit-norm decoder).
- `jumprelu_train.py` -- `train_sae()` and `best_qualifying_checkpoint()`.
  Reports the same three fidelity metrics as the TopK arm (raw MSE, NMSE,
  dead-latent fraction) plus `expl_var` and `hard_l0` (JumpReLU has no fixed
  k, so hard_l0 is measured post hoc rather than fixed by construction).
- `jumprelu_extract.py` -- GSM8K loading and forward-hook activation
  extraction, for layers not in the cached activation dataset (only 6, 12,
  18, 23 are cached; this arm's investigation required layers 15, 20, 21, 22
  as well).
- `05_train_sae_jumprelu.py` -- train one SAE at one (layer, l0, seed).
- `05b_sweep_jumprelu.py` -- two sweep modes:
  - `--mode l0-sweep`: find the best l0 at a layer under a hard_l0 sparsity
    cutoff (mirrors `scripts/topk/05b_sweep_hyperparams.sh`'s standardisation
    goal).
  - `--mode seed-check`: repeat the same config across multiple seeds to
    check whether an observed cross-layer quality difference is reproducible.

Both write CSVs to `results/` with columns matching
`sae_checkpoint_metrics.csv`'s conventions where applicable (nmse, dead_frac
as the load-bearing sparsity statistic, not mean L0 -- consistent with the
main README's reading notes) plus JumpReLU-specific columns (l0_coefficient,
expl_var, l0_ratio).

## Key results this arm produced (instruct_base only; SFT/PPO checkpoint
runs are not yet included -- see Limitations)

- Layer 23's SAE quality (expl_var=0.70 at the same l0 as layer 18) does not
  resolve under hyperparameter tuning alone: swept across a 33x l0 range,
  only 2 of 5 configurations ever reach a sparsity level comparable to layer
  18's, and none reach layer 18's combination of low l0 and high fidelity.
- A 6-layer scan (15/18/20/21/22/23) localises a sharp fidelity cliff to the
  layer 20->21 boundary specifically, not a gradual decline across the back
  half of the network.
- A per-dimension activation-variance check rules out normalization
  conditioning as the cause at both boundaries tested (18 vs 23, and 20 vs
  21): the better-fitting layer in each pair has a *more* uneven raw
  variance profile, not less.
- A seed-variation check (3 seeds, layers 20 vs 21, fixed l0=1e-3) found zero
  overlap in explained variance across seeds -- the quality gap is a
  reproducible effect, not a single training run's artifact.

## Limitations

- All fidelity results above are `instruct_base` only; the null-step
  comparison itself uses checkpoint chains as described in the main README.
- No `frac_rec` (loss-recovery) metric is computed for this arm yet. The TopK
  arm reports this alongside MSE/NMSE; adding it here would require patching
  SAE reconstructions into the live model's forward pass and comparing
  against mean-ablation, which this arm's scripts don't yet do.
- The GSM8K train/val split in `jumprelu_extract.py::load_gsm8k_split`
  reproduces the ~80/20 *ratio* described in the cached-activation dataset's
  README but is not confirmed to reproduce the same underlying rows -- see
  the function's docstring.
- Only layers 20 and 21 have been checked across random seeds. Layers
  15/18/22/23 have single-seed results only.
