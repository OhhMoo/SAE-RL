"""
Step 6: Evaluate trained SAEs.

For each SAE checkpoint, computes canonical quality metrics:
  * FVE: fraction of variance explained = 1 - SSE / SST.
  * L0:  empirical mean active features per token (equals k for TopK).
  * Dead fraction: features whose firing rate < dead_threshold across the eval set.
  * Feature-density buckets: log10(firing rate) histogram.
  * (Optional) CE delta: patch SAE reconstruction into the LM at that layer,
    measure the increase in next-token cross-entropy vs. the clean forward pass.

Usage:
    python scripts/06_eval_sae.py                           # cheap metrics only
    python scripts/06_eval_sae.py --ce_delta                # + patched-forward CE
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def load_sae(sae_path, device):
    ckpt = torch.load(sae_path, map_location=device, weights_only=True)
    cfg = ckpt["config"]
    sys.path.insert(0, str(Path(__file__).parent))
    from importlib import import_module
    mod = import_module("05_train_sae") if False else None  # 05 starts with digit
    # Inline the module class to avoid import-name issue:
    class TopKSAE(nn.Module):
        def __init__(self, d_model, d_sae, k):
            super().__init__()
            self.k = k
            self.b_pre = nn.Parameter(torch.zeros(d_model))
            self.encoder = nn.Linear(d_model, d_sae, bias=True)
            self.decoder = nn.Linear(d_sae, d_model, bias=True)
        def encode(self, x):
            z = self.encoder(x - self.b_pre)
            v, idx = torch.topk(z, self.k, dim=-1)
            out = torch.zeros_like(z)
            out.scatter_(-1, idx, v)
            return out
        def forward(self, x):
            z = self.encode(x)
            return self.decoder(z), z
    sae = TopKSAE(cfg["d_model"], cfg["d_sae"], cfg["k"]).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae, cfg


def eval_recon(sae, activations, device, batch_size=1024):
    """Compute FVE, MSE, and per-feature firing counts over the activation tensor."""
    d_sae = sae.decoder.weight.shape[1]
    total_mean = activations.float().mean(dim=0).to(device)

    sse = 0.0        # sum of squared residuals (x - xhat)
    sst = 0.0        # sum of squared deviations (x - mean)
    n_elems = 0
    feature_counts = torch.zeros(d_sae, device=device)
    n_tokens = 0
    active_per_token_total = 0

    loader = DataLoader(TensorDataset(activations), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device).float()
            x_hat, z = sae(batch)
            sse += (batch - x_hat).pow(2).sum().item()
            sst += (batch - total_mean).pow(2).sum().item()
            n_elems += batch.numel()
            active_mask = (z != 0)
            feature_counts += active_mask.float().sum(dim=0)
            active_per_token_total += active_mask.float().sum().item()
            n_tokens += batch.shape[0]

    fve = 1.0 - sse / sst
    mse_per_el = sse / n_elems
    l0 = active_per_token_total / n_tokens
    density = (feature_counts / n_tokens).cpu()
    return {
        "fve": fve,
        "mse": mse_per_el,
        "l0": l0,
        "density": density,
    }


def density_report(density, dead_threshold=1e-4):
    """Return dead fraction and log10-density histogram buckets."""
    dead_frac = (density < dead_threshold).float().mean().item()
    # log10 histogram on firing features only
    firing = density[density >= dead_threshold].clamp_min(1e-10)
    logd = firing.log10()
    buckets = {
        "dead(<1e-4)": dead_frac,
        "1e-4..1e-3": ((logd >= -4) & (logd < -3)).float().mean().item() * (1 - dead_frac),
        "1e-3..1e-2": ((logd >= -3) & (logd < -2)).float().mean().item() * (1 - dead_frac),
        "1e-2..1e-1": ((logd >= -2) & (logd < -1)).float().mean().item() * (1 - dead_frac),
        "1e-1..1":    ((logd >= -1) & (logd < 0)).float().mean().item() * (1 - dead_frac),
    }
    return dead_frac, buckets


def eval_ce_delta(model, tokenizer, sae, layer_idx, prompts, device, max_length=512, batch_size=4):
    """Measure CE increase when layer `layer_idx`'s output is replaced by SAE(reconstruction).

    Returns (clean_ce, patched_ce, delta_abs, delta_rel).
    """
    clean_losses = []
    patched_losses = []

    def make_patch_hook():
        def hook(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            shape = hidden.shape  # (B, T, D)
            flat = hidden.reshape(-1, shape[-1]).float()
            with torch.no_grad():
                x_hat, _ = sae(flat)
            patched = x_hat.reshape(shape).to(hidden.dtype)
            if isinstance(out, tuple):
                return (patched,) + out[1:]
            return patched
        return hook

    model.eval()
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100

        with torch.no_grad():
            out = model(**enc, labels=labels)
            clean_losses.append(out.loss.item())

        handle = model.model.layers[layer_idx].register_forward_hook(make_patch_hook())
        try:
            with torch.no_grad():
                out = model(**enc, labels=labels)
                patched_losses.append(out.loss.item())
        finally:
            handle.remove()

    clean_ce = sum(clean_losses) / len(clean_losses)
    patched_ce = sum(patched_losses) / len(patched_losses)
    return {
        "clean_ce": clean_ce,
        "patched_ce": patched_ce,
        "ce_delta": patched_ce - clean_ce,
        "ce_delta_rel": (patched_ce - clean_ce) / clean_ce,
    }


def model_path_for(stage_label, ppo_merged_root):
    """Map a stage label (instruct_base | ppo_stepN) to a model path / HF id."""
    if stage_label == "instruct_base":
        return "Qwen/Qwen2.5-0.5B-Instruct"
    m = re.match(r"ppo_step(\d+)$", stage_label)
    if m:
        return os.path.join(ppo_merged_root, f"step_{m.group(1)}")
    raise ValueError(f"unknown stage label: {stage_label}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_dir", default="checkpoints/saes")
    ap.add_argument("--activations_dir", default="data/activations")
    ap.add_argument("--out_json", default="checkpoints/saes/eval_report.json")
    ap.add_argument("--dead_threshold", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ce_delta", action="store_true",
                    help="Also compute patched-forward CE delta (expensive; loads LMs).")
    ap.add_argument("--ce_prompts", type=int, default=64,
                    help="Number of held-out GSM8k prompts to use for CE delta.")
    ap.add_argument("--ce_max_length", type=int, default=512)
    ap.add_argument("--ce_batch_size", type=int, default=4)
    ap.add_argument("--ppo_merged_root", default="checkpoints/ppo_merged")
    args = ap.parse_args()

    sae_files = sorted(Path(args.sae_dir).glob("sae_*.pt"))
    print(f"Found {len(sae_files)} SAE checkpoints")

    # Parse each name: sae_{stage}_layer{L}.pt
    records = []
    for f in sae_files:
        m = re.match(r"sae_(.+)_layer(\d+)\.pt$", f.name)
        if not m:
            print(f"  [skip] {f.name}")
            continue
        records.append({"path": f, "stage": m.group(1), "layer": int(m.group(2))})
    records.sort(key=lambda r: (r["stage"], r["layer"]))

    # Cheap metrics pass
    results = []
    for r in tqdm(records, desc="Cheap metrics"):
        act_path = Path(args.activations_dir) / f"{r['stage']}_layer{r['layer']}.pt"
        if not act_path.exists():
            print(f"  [skip] no activations for {r['stage']}_layer{r['layer']}")
            continue

        activations = torch.load(act_path, weights_only=True)
        sae, cfg = load_sae(r["path"], args.device)
        metrics = eval_recon(sae, activations, args.device)
        dead_frac, buckets = density_report(metrics["density"], args.dead_threshold)
        rec = {
            "stage": r["stage"],
            "layer": r["layer"],
            "n_tokens": activations.shape[0],
            "d_model": cfg["d_model"],
            "d_sae": cfg["d_sae"],
            "k": cfg["k"],
            "fve": metrics["fve"],
            "mse": metrics["mse"],
            "l0": metrics["l0"],
            "dead_frac": dead_frac,
            "density_buckets": buckets,
        }
        results.append(rec)
        del sae, activations
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # Optional CE delta pass (group by stage so each LM loads once)
    if args.ce_delta:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        prompts = [ex["question"] for ex in ds][: args.ce_prompts]

        stages = sorted({r["stage"] for r in results})
        for stage in stages:
            mpath = model_path_for(stage, args.ppo_merged_root)
            print(f"\n[CE delta] loading model: {mpath}")
            tok = AutoTokenizer.from_pretrained(mpath)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                mpath, torch_dtype=torch.float16
            ).to(args.device)
            for rec in [r for r in results if r["stage"] == stage]:
                sae_path = Path(args.sae_dir) / f"sae_{stage}_layer{rec['layer']}.pt"
                sae, _ = load_sae(sae_path, args.device)
                ce = eval_ce_delta(model, tok, sae, rec["layer"], prompts, args.device,
                                   max_length=args.ce_max_length,
                                   batch_size=args.ce_batch_size)
                rec.update(ce)
                print(f"  {stage}_layer{rec['layer']}: clean={ce['clean_ce']:.4f}  "
                      f"patched={ce['patched_ce']:.4f}  "
                      f"Δ={ce['ce_delta']:+.4f} ({100*ce['ce_delta_rel']:+.1f}%)")
                del sae
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            del model, tok
            if args.device == "cuda":
                torch.cuda.empty_cache()

    # Pretty print summary
    print(f"\n{'stage':<18} {'layer':>5} {'k':>4} {'d_sae':>6} "
          f"{'FVE':>7} {'MSE':>8} {'L0':>5} {'dead%':>6}", end="")
    if args.ce_delta:
        print(f" {'cleanCE':>8} {'patchCE':>8} {'ΔCE':>8} {'Δ%':>6}")
    else:
        print()
    for r in results:
        print(f"{r['stage']:<18} {r['layer']:>5} {r['k']:>4} {r['d_sae']:>6} "
              f"{r['fve']:>7.4f} {r['mse']:>8.4f} {r['l0']:>5.1f} "
              f"{100*r['dead_frac']:>5.1f}%", end="")
        if args.ce_delta and "ce_delta" in r:
            print(f" {r['clean_ce']:>8.4f} {r['patched_ce']:>8.4f} "
                  f"{r['ce_delta']:>+8.4f} {100*r['ce_delta_rel']:>+5.1f}%")
        else:
            print()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    # JSON-safe serialization
    for r in results:
        r["density_buckets"] = {k: float(v) for k, v in r["density_buckets"].items()}
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved report -> {args.out_json}")


if __name__ == "__main__":
    main()
