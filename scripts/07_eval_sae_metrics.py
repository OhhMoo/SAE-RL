"""
Step 7: Compute the three required SAE evaluation metrics on held-out data.

Metrics (per SAE):
  1. Reconstruction MSE   — mean squared error on held-out activations
  2. Mean L0              — average number of active features per token
  3. Model delta loss     — change in model cross-entropy when the SAE is
                            spliced in at inference time (measures whether
                            the reconstruction is faithful enough to not
                            degrade downstream computation)

Usage:
    python scripts/07_eval_sae_metrics.py \
        --sae_dir checkpoints/saes \
        --model_dir checkpoints/ppo_merged \
        --device cuda

Output: results/sae_eval_metrics.csv  (one row per SAE)
"""

import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# TopK SAE (matches 05_train_sae.py)
# ---------------------------------------------------------------------------

class TopKSAE(nn.Module):
    def __init__(self, d_model, d_sae, k):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.d_sae = d_sae
        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)

    def encode(self, x):
        z = self.encoder(x - self.b_pre)
        topk_values, topk_indices = torch.topk(z, self.k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topk_indices, topk_values)
        return z_sparse

    def forward(self, x):
        z_sparse = self.encode(x)
        return self.decoder(z_sparse), z_sparse


def load_sae(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sae = TopKSAE(cfg["d_model"], cfg["d_sae"], cfg["k"])
    sae.load_state_dict(ckpt["state_dict"], strict=False)
    return sae.to(device).eval(), cfg


# ---------------------------------------------------------------------------
# Metrics 1 & 2: reconstruction MSE and L0 from cached activations
# ---------------------------------------------------------------------------

def eval_recon_and_l0(sae: TopKSAE, acts: torch.Tensor, device: str,
                      batch_size: int = 512):
    sae = sae.to(device)
    acts = acts.to(device).float()

    total_mse = 0.0
    total_l0 = 0.0
    n_batches = 0

    with torch.no_grad():
        for i in range(0, len(acts), batch_size):
            batch = acts[i : i + batch_size]
            x_hat, z_sparse = sae(batch)
            total_mse += (batch - x_hat).pow(2).mean().item()
            total_l0  += (z_sparse != 0).float().sum(dim=-1).mean().item()
            n_batches += 1

    return total_mse / n_batches, total_l0 / n_batches


# ---------------------------------------------------------------------------
# Metric 3: model delta loss
# Splice the SAE reconstruction into the residual stream and measure the
# change in the model's language-modelling cross-entropy loss.
# ---------------------------------------------------------------------------

def eval_model_delta_loss(
    sae: TopKSAE,
    model,
    tokenizer,
    layer_idx: int,
    prompts: list[str],
    device: str,
    max_length: int = 256,
    batch_size: int = 4,
    n_prompts: int = 200,
) -> float:
    """
    Returns (loss_with_sae - loss_baseline).  Positive = SAE degrades model;
    near-zero = SAE reconstruction is faithful.
    """
    prompts = prompts[:n_prompts]
    model.eval()
    sae.eval()

    baseline_losses = []
    patched_losses  = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="  delta loss", leave=False):
        batch_prompts = prompts[i : i + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        # ---- Baseline loss (no SAE) ----
        with torch.no_grad():
            out = model(**enc, labels=input_ids)
            baseline_losses.append(out.loss.item())

        # ---- Patched loss (with SAE splice) ----
        hook_outputs = {}

        def fwd_hook(module, inp, out):
            # out may be a tuple (hidden, ...) or just a tensor depending on
            # the transformers version
            is_tuple = isinstance(out, tuple)
            hidden = (out[0] if is_tuple else out).float()
            B, T, D = hidden.shape
            flat = hidden.reshape(B * T, D)
            with torch.no_grad():
                reconstructed = sae.decoder(sae.encode(flat))
            orig_dtype = out[0].dtype if is_tuple else out.dtype
            patched = reconstructed.reshape(B, T, D).to(orig_dtype)
            return (patched,) + out[1:] if is_tuple else patched

        handle = model.model.layers[layer_idx].register_forward_hook(fwd_hook)
        try:
            with torch.no_grad():
                out = model(**enc, labels=input_ids)
                patched_losses.append(out.loss.item())
        finally:
            handle.remove()

    baseline = sum(baseline_losses) / len(baseline_losses)
    patched  = sum(patched_losses)  / len(patched_losses)
    return patched - baseline


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# PPO-without-SFT pipeline: stage 0 is the raw instruct model (baseline
# before any RL updates), and later stages are merged PPO checkpoints.
STAGE_ORDER = ["instruct_base", "ppo_step10", "ppo_step30", "ppo_step50",
               "ppo_step80", "ppo_step100", "ppo_step120", "ppo_step140",
               "ppo_step160", "ppo_step180", "ppo_step200"]

STAGE_TO_MODEL = {
    "instruct_base": "Qwen/Qwen2.5-0.5B-Instruct",
    "ppo_step10":    "checkpoints/ppo_merged/step_10",
    "ppo_step30":    "checkpoints/ppo_merged/step_30",
    "ppo_step50":    "checkpoints/ppo_merged/step_50",
    "ppo_step80":    "checkpoints/ppo_merged/step_80",
    "ppo_step100":   "checkpoints/ppo_merged/step_100",
    "ppo_step120":   "checkpoints/ppo_merged/step_120",
    "ppo_step140":   "checkpoints/ppo_merged/step_140",
    "ppo_step160":   "checkpoints/ppo_merged/step_160",
    "ppo_step180":   "checkpoints/ppo_merged/step_180",
    "ppo_step200":   "checkpoints/ppo_merged/step_200",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_dir",      default="checkpoints/saes")
    parser.add_argument("--activations_dir", default="data/activations")
    parser.add_argument("--model_dir",    default="checkpoints/ppo_merged",
                        help="Root of merged model checkpoints (used to resolve stage paths)")
    parser.add_argument("--output_dir",   default="results")
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--skip_delta",   action="store_true",
                        help="Skip model delta loss (faster, no model loading needed)")
    parser.add_argument("--n_delta_prompts", type=int, default=200,
                        help="Number of GSM8k test prompts for delta loss eval")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, "sae_eval_metrics.csv")

    sae_root = Path(args.sae_dir)
    act_root = Path(args.activations_dir)

    # Collect SAE paths
    sae_paths: dict[tuple, Path] = {}
    for f in sorted(sae_root.glob("sae_*.pt")):
        name = f.stem[len("sae_"):]
        parts = name.rsplit("_layer", 1)
        if len(parts) == 2:
            stage, layer_str = parts
            sae_paths[(stage, int(layer_str))] = f

    # Collect activation paths
    act_paths: dict[tuple, Path] = {}
    for f in sorted(act_root.glob("*.pt")):
        parts = f.stem.rsplit("_layer", 1)
        if len(parts) == 2:
            act_paths[(parts[0], int(parts[1]))] = f

    # Load held-out prompts from GSM8k test split
    print("Loading GSM8k test prompts...")
    test_prompts = [ex["question"]
                    for ex in load_dataset("openai/gsm8k", "main", split="test")]

    rows = []

    # Cache loaded models to avoid reloading for every layer of the same stage
    loaded_model_stage = None
    model = tokenizer = None

    for stage in STAGE_ORDER:
        for layer in sorted({k[1] for k in sae_paths}):
            key = (stage, layer)
            if key not in sae_paths:
                continue

            print(f"\n[{stage}  layer {layer}]")
            sae, cfg = load_sae(str(sae_paths[key]), args.device)

            # --- Metrics 1 & 2 ---
            mse = l0 = None
            if key in act_paths:
                # Use held-out slice: last 20% of cached tokens
                acts_all = torch.load(act_paths[key], weights_only=True)
                split = int(len(acts_all) * 0.8)
                acts_val = acts_all[split:]
                mse, l0 = eval_recon_and_l0(sae, acts_val, args.device)
                print(f"  recon_mse={mse:.6f}  mean_l0={l0:.2f}")
            else:
                print("  (no activation file — skipping MSE/L0)")

            # --- Metric 3: model delta loss ---
            delta = None
            if not args.skip_delta and stage in STAGE_TO_MODEL:
                model_path = STAGE_TO_MODEL[stage]
                if not Path(model_path).exists():
                    print(f"  model path not found ({model_path}) — skipping delta loss")
                else:
                    if loaded_model_stage != stage:
                        print(f"  Loading model: {model_path}")
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path, dtype=torch.float16
                        ).to(args.device)
                        loaded_model_stage = stage

                    delta = eval_model_delta_loss(
                        sae, model, tokenizer, layer,
                        test_prompts, args.device,
                        n_prompts=args.n_delta_prompts,
                    )
                    print(f"  model_delta_loss={delta:+.6f}")

            rows.append({
                "stage": stage,
                "layer": layer,
                "recon_mse": f"{mse:.6f}" if mse is not None else "",
                "mean_l0":   f"{l0:.2f}"   if l0  is not None else "",
                "model_delta_loss": f"{delta:+.6f}" if delta is not None else "",
            })

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stage", "layer", "recon_mse",
                                               "mean_l0", "model_delta_loss"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Results -> {out_csv}")

    # Also print a summary table
    print(f"\n{'stage':<15} {'layer':>5} {'recon_mse':>12} {'mean_l0':>10} {'delta_loss':>12}")
    print("-" * 58)
    for r in rows:
        print(f"{r['stage']:<15} {r['layer']:>5} {r['recon_mse']:>12} "
              f"{r['mean_l0']:>10} {r['model_delta_loss']:>12}")


if __name__ == "__main__":
    main()
