# SAE non-normalized MSE + dead-latent proportions

Computed by `scripts/06e_mse_dead_latents.py` from the trained TopK SAE
checkpoints + cached held-out val activations (~89k tokens/SAE, no LLM pass).
Full per-SAE numbers: `results/sae_mse_dead.csv` (86 rows = all 4 reward chains
× PPO stages × layers 6/12/18/23).

**Validation:** the NMSE column recomputed here matches the existing
`sae_eval*.csv` to 4 decimals, so the raw-MSE and dead numbers come from the
same load/forward path.

## Per-layer summary (range = min–max across all stages + reward chains)

| Layer | K   | Raw MSE (per-elem) | Var(x) | NMSE          | Dead latents (/7168) |
|-------|-----|--------------------|--------|---------------|----------------------|
| 6     | 64  | 0.016 – 0.025      | ~46    | 0.0003–0.0005 | 0.2% – 4.5%          |
| 12    | 64  | 0.030 – 0.041      | ~47    | 0.0007–0.0009 | 0.2% – 4.4%          |
| 18    | 64  | 0.155 – 0.178      | ~48    | 0.0032–0.0038 | 0.1% – 3.3%          |
| 23    | 64  | 0.84 – 0.92        | ~3.0   | 0.281–0.301   | 0%                   |
| 23    | 256 | 0.42 – 0.54        | ~2.9   | 0.149–0.177   | 0%                   |

Over all 86 SAEs: dead-latent fraction mean 0.96%, max 4.51% (323/7168); 26/86
have zero dead latents.

## Read-me caveats

- **Raw MSE is scale-dependent — not comparable across layers.** Var(x) at L23
  (~3.0) is ~15× smaller than at L6/12/18 (~46–48), which is why L23's raw MSE
  looks large despite reasonable NMSE. Compare raw MSE *within* a layer; use
  NMSE for cross-layer.
- **Dead** = latent never selected into the top-k on any of the ~89k held-out
  val tokens. d_sae = 7168 (8× expansion).
- Dead counts stay low because training resamples dead features every 10 steps
  (dead_threshold 1e-4) and warm-starts each PPO-stage SAE from the previous
  stage. The 68%-dead (k=64) / 21%-dead (BatchTopK k=220) numbers in the writeup
  are from Jake's standalone grid search (no resampling), **not** these chains.

## Architecture

All 86 SAEs are **non-batch (per-token) TopK** — `torch.topk(z, k, dim=-1)`
applied independently per token (`05_train_sae.py:56`). No BatchTopK variants
were trained (repo-wide grep for batch-topk is empty).
