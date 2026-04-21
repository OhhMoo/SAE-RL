"""
Step 7 (maskeval variant): evaluate SAEs with a padding-safe, mean-ablation-
referenced metric.

Differences vs 07_eval_sae_metrics.py:
  - Padding tokens are excluded from the CE loss (labels masked to -100).
  - The forward hook only overwrites real-token positions; padding hidden
    states are passed through unchanged. This matches how the SAE was
    trained (padding was stripped in 04_collect_activations.py).
  - Reports fraction of variance explained (FVE) alongside MSE.
  - Adds a mean-ablation arm and reports fraction of CE loss recovered:
        frac_recovered = (L_mean - L_sae) / (L_mean - L_baseline)
    Bounded in expectation to [0, 1]: 1 = perfect reconstruction, 0 = no
    better than replacing the layer with the dataset mean. Values outside
    this range expose degeneracies (e.g. smoothing that beats the baseline).
  - Accepts Hugging Face model IDs (not just local paths) for the baseline
    stage.

Output: results/sae_eval_metrics_maskeval.csv (one row per SAE).
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
# Metrics 1 & 2: reconstruction MSE, L0, FVE from cached activations
# ---------------------------------------------------------------------------

def eval_recon_and_l0(sae: TopKSAE, acts: torch.Tensor, device: str,
                      batch_size: int = 512):
    sae = sae.to(device)
    acts = acts.to(device).float()

    total_sq_err = 0.0
    total_l0 = 0.0
    n_elems = 0
    n_batches = 0

    # Dataset variance is computed once on the full held-out slice.
    var = acts.var().item()

    with torch.no_grad():
        for i in range(0, len(acts), batch_size):
            batch = acts[i : i + batch_size]
            x_hat, z_sparse = sae(batch)
            total_sq_err += (batch - x_hat).pow(2).sum().item()
            total_l0     += (z_sparse != 0).float().sum(dim=-1).mean().item()
            n_elems      += batch.numel()
            n_batches    += 1

    mse = total_sq_err / n_elems
    l0  = total_l0 / n_batches
    fve = 1.0 - mse / var if var > 0 else float("nan")
    return mse, l0, fve


# ---------------------------------------------------------------------------
# Metric 3: padding-safe model delta loss with mean-ablation reference
# ---------------------------------------------------------------------------

def _masked_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Replace padding positions with -100 so they don't contribute to CE."""
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return labels


def _run_with_replacement(model, enc, layer_idx, replace_fn):
    """Run the model with `replace_fn(hidden, mask) -> new_hidden` spliced at
    `model.model.layers[layer_idx]`. Only real-token positions are overwritten.
    Returns the CE loss (padding-masked labels).
    """
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = _masked_labels(input_ids, attention_mask)

    def fwd_hook(module, inp, out):
        is_tuple = isinstance(out, tuple)
        hidden = (out[0] if is_tuple else out)
        orig_dtype = hidden.dtype
        h_f = hidden.float()
        new_hidden = replace_fn(h_f, attention_mask)
        # Only overwrite real-token positions; leave padding hidden states alone.
        mask = attention_mask.unsqueeze(-1).bool()
        patched = torch.where(mask, new_hidden.to(orig_dtype), hidden)
        return (patched,) + out[1:] if is_tuple else patched

    handle = model.model.layers[layer_idx].register_forward_hook(fwd_hook)
    try:
        with torch.no_grad():
            out = model(**enc, labels=labels)
    finally:
        handle.remove()
    return out.loss.item()


def eval_delta_loss_triplet(
    sae: TopKSAE,
    model,
    tokenizer,
    layer_idx: int,
    prompts: list[str],
    device: str,
    max_length: int = 256,
    batch_size: int = 4,
    n_prompts: int = 200,
):
    """Return (L_baseline, L_sae, L_mean, frac_recovered).

    Loss is CE with padding masked to -100.
    L_sae:  splice sae(x) at `layer_idx` on real-token positions only.
    L_mean: splice the dataset mean vector at the same positions.
    """
    prompts = prompts[:n_prompts]
    model.eval()
    sae.eval()

    # --- Estimate layer mean on real tokens over a small warm-up pass ----
    # We collect the real-token mean of the ORIGINAL layer output.
    mean_accum = torch.zeros(sae.d_model, device=device, dtype=torch.float32)
    count = 0

    captured = {}

    def capture_hook(module, inp, out):
        hidden = (out[0] if isinstance(out, tuple) else out).float()
        captured["h"] = hidden
        return out

    warmup_n = min(32, len(prompts))
    handle = model.model.layers[layer_idx].register_forward_hook(capture_hook)
    try:
        for i in range(0, warmup_n, batch_size):
            batch_prompts = prompts[i : i + batch_size]
            enc = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length).to(device)
            with torch.no_grad():
                model(**enc)
            hidden = captured["h"]  # (B, T, D)
            mask = enc["attention_mask"].bool()
            real = hidden[mask]  # (n_real, D)
            mean_accum += real.sum(dim=0)
            count += real.shape[0]
    finally:
        handle.remove()

    mean_vec = (mean_accum / max(count, 1)).to(device)  # (D,)

    # --- Eval loop ---
    baseline_losses, sae_losses, mean_losses = [], [], []

    for i in tqdm(range(0, len(prompts), batch_size),
                  desc="  delta loss", leave=False):
        batch_prompts = prompts[i : i + batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length).to(device)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels = _masked_labels(input_ids, attention_mask)

        # Baseline
        with torch.no_grad():
            out = model(**enc, labels=labels)
            baseline_losses.append(out.loss.item())

        # SAE splice
        def sae_replace(hidden_f, mask):
            B, T, D = hidden_f.shape
            flat = hidden_f.reshape(B * T, D)
            with torch.no_grad():
                recon = sae.decoder(sae.encode(flat))
            return recon.reshape(B, T, D)

        sae_losses.append(_run_with_replacement(model, enc, layer_idx, sae_replace))

        # Mean ablation
        def mean_replace(hidden_f, mask):
            B, T, D = hidden_f.shape
            return mean_vec.view(1, 1, D).expand(B, T, D).contiguous()

        mean_losses.append(_run_with_replacement(model, enc, layer_idx, mean_replace))

    L_baseline = sum(baseline_losses) / len(baseline_losses)
    L_sae      = sum(sae_losses)      / len(sae_losses)
    L_mean     = sum(mean_losses)     / len(mean_losses)

    denom = L_mean - L_baseline
    frac_recovered = (L_mean - L_sae) / denom if abs(denom) > 1e-8 else float("nan")
    return L_baseline, L_sae, L_mean, frac_recovered


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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


def _resolve_model_path(stage_id: str) -> str | None:
    """Return a path/ID if the model is loadable, else None.

    Local checkpoints must exist on disk; Hugging Face IDs (anything not
    starting with '.', '/', or an existing path) are passed through.
    """
    if stage_id not in STAGE_TO_MODEL:
        return None
    path = STAGE_TO_MODEL[stage_id]
    if "/" in path and not Path(path).parts[0].startswith("checkpoints"):
        # Looks like an HF repo id (e.g. "Qwen/Qwen2.5-0.5B-Instruct")
        return path
    return path if Path(path).exists() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_dir",         default="checkpoints/saes")
    parser.add_argument("--activations_dir", default="data/activations")
    parser.add_argument("--model_dir",       default="checkpoints/ppo_merged")
    parser.add_argument("--output_dir",      default="results")
    parser.add_argument("--output_csv",      default="sae_eval_metrics_maskeval.csv")
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--skip_delta",      action="store_true")
    parser.add_argument("--n_delta_prompts", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, args.output_csv)

    sae_root = Path(args.sae_dir)
    act_root = Path(args.activations_dir)

    sae_paths: dict[tuple, Path] = {}
    for f in sorted(sae_root.glob("sae_*.pt")):
        name = f.stem[len("sae_"):]
        parts = name.rsplit("_layer", 1)
        if len(parts) == 2:
            stage, layer_str = parts
            sae_paths[(stage, int(layer_str))] = f

    act_paths: dict[tuple, Path] = {}
    for f in sorted(act_root.glob("*.pt")):
        parts = f.stem.rsplit("_layer", 1)
        if len(parts) == 2:
            act_paths[(parts[0], int(parts[1]))] = f

    print("Loading GSM8k test prompts...")
    test_prompts = [ex["question"]
                    for ex in load_dataset("openai/gsm8k", "main", split="test")]

    rows = []
    loaded_model_stage = None
    model = tokenizer = None

    for stage in STAGE_ORDER:
        for layer in sorted({k[1] for k in sae_paths}):
            key = (stage, layer)
            if key not in sae_paths:
                continue

            print(f"\n[{stage}  layer {layer}]")
            sae, cfg = load_sae(str(sae_paths[key]), args.device)

            mse = l0 = fve = None
            if key in act_paths:
                acts_all = torch.load(act_paths[key], weights_only=True)
                split = int(len(acts_all) * 0.8)
                acts_val = acts_all[split:]
                mse, l0, fve = eval_recon_and_l0(sae, acts_val, args.device)
                print(f"  recon_mse={mse:.6f}  mean_l0={l0:.2f}  FVE={fve:.4f}")
            else:
                print("  (no activation file — skipping MSE/L0/FVE)")

            L_base = L_sae = L_mean = frac = None
            if not args.skip_delta:
                model_path = _resolve_model_path(stage)
                if model_path is None:
                    print(f"  model path not available for {stage} — skipping delta loss")
                else:
                    if loaded_model_stage != stage:
                        print(f"  Loading model: {model_path}")
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        if tokenizer.pad_token is None:
                            tokenizer.pad_token = tokenizer.eos_token
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path, torch_dtype=torch.float16
                        ).to(args.device)
                        loaded_model_stage = stage

                    L_base, L_sae, L_mean, frac = eval_delta_loss_triplet(
                        sae, model, tokenizer, layer,
                        test_prompts, args.device,
                        n_prompts=args.n_delta_prompts,
                    )
                    print(f"  L_base={L_base:.4f}  L_sae={L_sae:.4f}  "
                          f"L_mean={L_mean:.4f}  frac_recovered={frac:.4f}")

            rows.append({
                "stage": stage,
                "layer": layer,
                "recon_mse": f"{mse:.6f}" if mse is not None else "",
                "mean_l0":   f"{l0:.2f}"  if l0  is not None else "",
                "fve":       f"{fve:.4f}" if fve is not None else "",
                "L_baseline":      f"{L_base:.4f}" if L_base is not None else "",
                "L_sae_patched":   f"{L_sae:.4f}"  if L_sae  is not None else "",
                "L_mean_ablation": f"{L_mean:.4f}" if L_mean is not None else "",
                "frac_loss_recovered": f"{frac:.4f}" if frac is not None else "",
            })

    fieldnames = ["stage", "layer", "recon_mse", "mean_l0", "fve",
                  "L_baseline", "L_sae_patched", "L_mean_ablation",
                  "frac_loss_recovered"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Results -> {out_csv}")

    print(f"\n{'stage':<15} {'layer':>5} {'mse':>10} {'FVE':>8} "
          f"{'L_base':>8} {'L_sae':>8} {'L_mean':>8} {'frac_rec':>9}")
    print("-" * 78)
    for r in rows:
        print(f"{r['stage']:<15} {r['layer']:>5} {r['recon_mse']:>10} "
              f"{r['fve']:>8} {r['L_baseline']:>8} {r['L_sae_patched']:>8} "
              f"{r['L_mean_ablation']:>8} {r['frac_loss_recovered']:>9}")


if __name__ == "__main__":
    main()
