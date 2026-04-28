"""
Minimal SAE eval — normalized MSE + delta loss, one CSV row per SAE.

  - nmse      = mean (x - x_hat)^2 / Var(x), pooled over all elements of the
                held-out activations. Lower is better. NMSE = 0 ⇔ perfect recon.
  - mean_l0   = average number of nonzero features per token (~ k for TopK).
  - L_base    = CE loss on GSM8k test prompts with the live model untouched.
  - L_sae     = CE loss with sae(x) spliced in at the layer (real-token positions only).
  - L_mean    = CE loss with the layer's real-token mean spliced in (control).
  - frac_rec  = (L_mean - L_sae) / (L_mean - L_base). 1 = SAE recovers all the loss
                a mean ablation would have lost; 0 = SAE no better than the mean.

Inputs: SAE checkpoints in --sae_dir, val activations in --activations_dir
(`{stage}_layer{N}_val.pt`), merged HF models in checkpoints/ppo_merged/step_N,
and the HF id Qwen/Qwen2.5-0.5B-Instruct for the instruct baseline.

Output: results/sae_eval.csv (default) — one row per SAE.

Run from sae_rl/:
    python scripts/eval_sae.py
    python scripts/eval_sae.py --skip_delta            # NMSE/L0 only, fast
    python scripts/eval_sae.py --n_delta_prompts 50    # quicker delta loss
"""

import argparse
import csv
import re
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# TopK SAE — must match scripts/05_train_sae.py
# ---------------------------------------------------------------------------

class TopKSAE(nn.Module):
    def __init__(self, d_model, d_sae, k):
        super().__init__()
        self.k, self.d_model, self.d_sae = k, d_model, d_sae
        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)

    def encode(self, x):
        z = self.encoder(x - self.b_pre)
        topk_v, topk_i = torch.topk(z, self.k, dim=-1)
        z_sparse = torch.zeros_like(z).scatter_(-1, topk_i, topk_v)
        return z_sparse

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z), z


def load_sae(path: Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sae = TopKSAE(cfg["d_model"], cfg["d_sae"], cfg["k"])
    sae.load_state_dict(ckpt["state_dict"], strict=False)
    return sae.to(device).eval(), cfg


# ---------------------------------------------------------------------------
# NMSE + L0 on cached val activations
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_nmse_l0(sae, acts: torch.Tensor, device: str, batch_size: int = 512):
    acts = acts.to(device).float()
    var = acts.var().item()
    sq_err = 0.0
    l0_sum = 0.0
    n_elems = 0
    n_batches = 0
    for i in range(0, len(acts), batch_size):
        b = acts[i:i + batch_size]
        x_hat, z = sae(b)
        sq_err += (b - x_hat).pow(2).sum().item()
        l0_sum += (z != 0).float().sum(dim=-1).mean().item()
        n_elems += b.numel()
        n_batches += 1
    mse = sq_err / n_elems
    return mse / var if var > 0 else float("nan"), l0_sum / n_batches


# ---------------------------------------------------------------------------
# Delta loss with mean-ablation reference (padding-safe)
# ---------------------------------------------------------------------------

def _masked_labels(input_ids, attention_mask):
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return labels


def _run_with_replacement(model, enc, layer_idx, replace_fn):
    labels = _masked_labels(enc["input_ids"], enc["attention_mask"])
    mask = enc["attention_mask"].unsqueeze(-1).bool()

    def hook(module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = (out[0] if is_tuple else out)
        new_h = replace_fn(h.float()).to(h.dtype)
        patched = torch.where(mask, new_h, h)
        return (patched,) + out[1:] if is_tuple else patched

    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(**enc, labels=labels)
    finally:
        handle.remove()
    return out.loss.item()


@torch.no_grad()
def eval_delta_loss(sae, model, tokenizer, layer_idx, prompts, device,
                    n_prompts=200, batch_size=4, max_length=256):
    prompts = prompts[:n_prompts]
    model.eval()

    # Estimate the layer's real-token mean over a small warm-up.
    mean_accum = torch.zeros(sae.d_model, device=device, dtype=torch.float32)
    count = 0
    captured = {}

    def capture(module, inp, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).float()
        return out

    handle = model.model.layers[layer_idx].register_forward_hook(capture)
    try:
        for i in range(0, min(32, len(prompts)), batch_size):
            enc = tokenizer(prompts[i:i + batch_size], return_tensors="pt",
                            padding=True, truncation=True,
                            max_length=max_length).to(device)
            model(**enc)
            real = captured["h"][enc["attention_mask"].bool()]
            mean_accum += real.sum(dim=0)
            count += real.shape[0]
    finally:
        handle.remove()
    mean_vec = (mean_accum / max(count, 1)).to(device)

    L_base, L_sae, L_mean = [], [], []
    for i in tqdm(range(0, len(prompts), batch_size),
                  desc="  delta", leave=False):
        enc = tokenizer(prompts[i:i + batch_size], return_tensors="pt",
                        padding=True, truncation=True,
                        max_length=max_length).to(device)
        labels = _masked_labels(enc["input_ids"], enc["attention_mask"])
        L_base.append(model(**enc, labels=labels).loss.item())

        def sae_replace(h_f):
            B, T, D = h_f.shape
            recon, _ = sae(h_f.reshape(B * T, D))
            return recon.reshape(B, T, D)

        def mean_replace(h_f):
            B, T, D = h_f.shape
            return mean_vec.view(1, 1, D).expand(B, T, D).contiguous()

        L_sae.append(_run_with_replacement(model, enc, layer_idx, sae_replace))
        L_mean.append(_run_with_replacement(model, enc, layer_idx, mean_replace))

    L_base = sum(L_base) / len(L_base)
    L_sae  = sum(L_sae)  / len(L_sae)
    L_mean = sum(L_mean) / len(L_mean)
    denom = L_mean - L_base
    frac = (L_mean - L_sae) / denom if abs(denom) > 1e-8 else float("nan")
    return L_base, L_sae, L_mean, frac


# ---------------------------------------------------------------------------
# Stage → model resolver
# ---------------------------------------------------------------------------

def resolve_model_path(stage: str, merged_root: Path):
    if stage == "instruct_base":
        return "Qwen/Qwen2.5-0.5B-Instruct"
    m = re.match(r"ppo_step(\d+)$", stage)
    if not m:
        return None
    p = merged_root / f"step_{m.group(1)}"
    return str(p) if p.exists() else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STAGE_ORDER = ["instruct_base", "ppo_step10", "ppo_step30", "ppo_step60",
               "ppo_step100", "ppo_step140", "ppo_step180", "ppo_step200"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_dir",         default="checkpoints/saes")
    ap.add_argument("--activations_dir", default="data/activations")
    ap.add_argument("--merged_dir",      default="checkpoints/ppo_merged")
    ap.add_argument("--output_csv",      default="results/sae_eval.csv")
    ap.add_argument("--device",          default="cuda")
    ap.add_argument("--skip_delta",      action="store_true")
    ap.add_argument("--n_delta_prompts", type=int, default=200)
    args = ap.parse_args()

    sae_dir = Path(args.sae_dir)
    act_dir = Path(args.activations_dir)
    merged_root = Path(args.merged_dir)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    saes = {}
    for f in sorted(sae_dir.glob("sae_*.pt")):
        name = f.stem[len("sae_"):]
        parts = name.rsplit("_layer", 1)
        if len(parts) == 2 and parts[1].isdigit():
            saes[(parts[0], int(parts[1]))] = f
    if not saes:
        print(f"No SAEs found in {sae_dir}")
        return
    stages_present = [s for s in STAGE_ORDER if any(k[0] == s for k in saes)]
    layers_present = sorted({k[1] for k in saes})
    print(f"{len(saes)} SAEs across stages={stages_present} layers={layers_present}")

    test_prompts = None
    if not args.skip_delta:
        print("Loading GSM8k test prompts...")
        test_prompts = [ex["question"] for ex in
                        load_dataset("openai/gsm8k", "main", split="test")]

    rows = []
    loaded_stage = None
    model = tokenizer = None

    for stage in stages_present:
        for layer in layers_present:
            key = (stage, layer)
            if key not in saes:
                continue
            print(f"\n[{stage}  layer {layer}]")
            sae, cfg = load_sae(saes[key], args.device)

            val_path = act_dir / f"{stage}_layer{layer}_val.pt"
            nmse = mean_l0 = None
            if val_path.exists():
                acts = torch.load(val_path, weights_only=True)
                nmse, mean_l0 = eval_nmse_l0(sae, acts, args.device)
                print(f"  nmse={nmse:.4f}  mean_l0={mean_l0:.2f}")
            else:
                print(f"  [warn] no val activations at {val_path}")

            L_base = L_sae = L_mean = frac = None
            if not args.skip_delta:
                model_path = resolve_model_path(stage, merged_root)
                if model_path is None:
                    print(f"  [warn] no merged model for {stage}; skipping delta")
                else:
                    if loaded_stage != stage:
                        del model
                        torch.cuda.empty_cache() if args.device == "cuda" else None
                        print(f"  loading model: {model_path}")
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        if tokenizer.pad_token is None:
                            tokenizer.pad_token = tokenizer.eos_token
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path, torch_dtype=torch.float16
                        ).to(args.device)
                        loaded_stage = stage
                    L_base, L_sae, L_mean, frac = eval_delta_loss(
                        sae, model, tokenizer, layer, test_prompts,
                        args.device, n_prompts=args.n_delta_prompts,
                    )
                    print(f"  L_base={L_base:.4f}  L_sae={L_sae:.4f}  "
                          f"L_mean={L_mean:.4f}  frac_rec={frac:.4f}")

            rows.append({
                "stage": stage,
                "layer": layer,
                "k": cfg["k"],
                "d_sae": cfg["d_sae"],
                "nmse":      f"{nmse:.4f}"   if nmse     is not None else "",
                "mean_l0":   f"{mean_l0:.2f}" if mean_l0 is not None else "",
                "L_base":    f"{L_base:.4f}" if L_base   is not None else "",
                "L_sae":     f"{L_sae:.4f}"  if L_sae    is not None else "",
                "L_mean":    f"{L_mean:.4f}" if L_mean   is not None else "",
                "frac_rec":  f"{frac:.4f}"   if frac     is not None else "",
            })

            del sae
            if args.device == "cuda":
                torch.cuda.empty_cache()

    fieldnames = ["stage", "layer", "k", "d_sae", "nmse", "mean_l0",
                  "L_base", "L_sae", "L_mean", "frac_rec"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nDone. Results -> {out_csv}")

    print(f"\n{'stage':<15} {'layer':>5} {'nmse':>8} {'L0':>6} "
          f"{'L_base':>8} {'L_sae':>8} {'frac':>7}")
    print("-" * 65)
    for r in rows:
        print(f"{r['stage']:<15} {r['layer']:>5} {r['nmse']:>8} "
              f"{r['mean_l0']:>6} {r['L_base']:>8} {r['L_sae']:>8} "
              f"{r['frac_rec']:>7}")


if __name__ == "__main__":
    main()
