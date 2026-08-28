"""
06e: Non-normalized (raw) reconstruction MSE + dead-latent proportion per SAE.

The minimal eval (06b_eval_sae.py) records only NMSE (= MSE / Var(x)) and never
computes dead latents. This script recovers both directly from the trained SAE
checkpoints and the cached val activations — no LLM forward pass required, so it
is fast (seconds for all chains).

Definitions
-----------
  mse        raw mean per-element squared reconstruction error over the held-out
             val activations: sum (x - x_hat)^2 / x.numel().  (NOT divided by var)
  var        Var(x) over all elements (unbiased), matching 06b's NMSE denominator.
  nmse       mse / var  -- recomputed here purely to cross-check against the
             existing sae_eval*.csv files (should match to rounding).
  mean_l0    avg number of nonzero latents per token (~ k for TopK).
  dead_frac  fraction of the d_sae latents that NEVER fire on any val token
             (never selected into the top-k). dead = (z == 0 for all tokens).
  n_dead     dead_frac * d_sae (integer count of never-firing latents).

Note: raw MSE magnitude is scale-dependent (the residual stream grows with depth,
so L23 has a far larger MSE than L6 at equal NMSE); always read it alongside var.

Run from sae_rl/:
    python scripts/06e_mse_dead_latents.py
"""

import argparse
import csv
import re
from pathlib import Path

import torch
import torch.nn as nn


# TopK SAE — must match scripts/05_train_sae.py / 06b_eval_sae.py
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


@torch.no_grad()
def eval_one(sae, acts: torch.Tensor, device: str, batch_size: int = 4096):
    acts = acts.to(device).float()
    var = acts.var().item()                       # matches 06b NMSE denominator
    d_sae = sae.d_sae
    sq_err = 0.0
    l0_sum = 0.0
    n_batches = 0
    fired = torch.zeros(d_sae, dtype=torch.bool, device=device)
    for i in range(0, len(acts), batch_size):
        b = acts[i:i + batch_size]
        x_hat, z = sae(b)
        sq_err += (b - x_hat).pow(2).sum().item()
        l0_sum += (z != 0).float().sum(dim=-1).mean().item()
        fired |= (z != 0).any(dim=0)
        n_batches += 1
    mse = sq_err / acts.numel()
    nmse = mse / var if var > 0 else float("nan")
    mean_l0 = l0_sum / n_batches
    n_dead = int((~fired).sum().item())
    dead_frac = n_dead / d_sae
    return dict(mse=mse, var=var, nmse=nmse, mean_l0=mean_l0,
                d_sae=d_sae, n_dead=n_dead, dead_frac=dead_frac,
                n_tokens=len(acts))


# (chain label, SAE checkpoint dir, primary val-activation dir,
#  optional fallback val-activation dir)
#
# SFT checkpoints use ``data/activations_sft`` for their fine-tuned stages,
# while the shared instruct-base SAE retains the original activation cache.
# Keeping the fallback explicit lets the output contain a true step-0 baseline
# for the SFT trajectory instead of silently dropping it.
CHAINS = [
    ("flexible",         "checkpoints/saes",                "data/activations",          None),
    ("strict",           "checkpoints/saes_strict",         "data/activations_strict",   None),
    ("strict_l23_k256",  "checkpoints/saes_strict_l23_k256", "data/activations_strict",   None),
    ("kl0p025",          "checkpoints/saes_kl0p025",        "data/activations_kl0p025",  None),
    ("shuffled",         "checkpoints/saes_shuffled",       "data/activations_shuffled", None),
    ("sft",              "checkpoints/saes_sft",            "data/activations_sft",      "data/activations"),
]

STAGE_ORDER = ["instruct_base", "ppo_step10", "ppo_step30", "ppo_step50",
               "ppo_step60", "ppo_step80", "ppo_step100", "ppo_step116",
               "ppo_step140", "ppo_step180", "ppo_step200"]


def stage_key(stage):
    if stage == "instruct_base":
        return -1
    m = re.match(r"(?:ppo|sft)_step(\d+)$", stage)
    return int(m.group(1)) if m else 9999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_csv", default="results/sae_mse_dead.csv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--chains", nargs="+", choices=[chain[0] for chain in CHAINS],
        help="Evaluate only the named chains. Use this in a sparse checkout, "
             "for example: --chains sft --output_csv results/sae_mse_dead_sft.csv",
    )
    args = ap.parse_args()

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    selected_chains = set(args.chains) if args.chains else None
    for chain, sae_dir, act_dir, fallback_act_dir in CHAINS:
        if selected_chains is not None and chain not in selected_chains:
            continue
        sae_dir, act_dir = Path(sae_dir), Path(act_dir)
        fallback_act_dir = Path(fallback_act_dir) if fallback_act_dir else None
        if not sae_dir.exists():
            print(f"[skip] {chain}: no {sae_dir}")
            continue
        files = sorted(sae_dir.glob("sae_*.pt"))
        for f in files:
            name = f.stem[len("sae_"):]
            parts = name.rsplit("_layer", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                continue
            stage, layer = parts[0], int(parts[1])
            val_path = act_dir / f"{stage}_layer{layer}_val.pt"
            if not val_path.exists() and fallback_act_dir:
                fallback = fallback_act_dir / f"{stage}_layer{layer}_val.pt"
                if fallback.exists():
                    val_path = fallback
            if not val_path.exists():
                print(f"[warn] {chain} {stage} L{layer}: missing {val_path}")
                continue
            sae, cfg = load_sae(f, args.device)
            acts = torch.load(val_path, weights_only=True)
            m = eval_one(sae, acts, args.device)
            rows.append(dict(chain=chain, stage=stage, layer=layer,
                             k=cfg["k"], **m))
            print(f"{chain:<16} {stage:<14} L{layer:<2} k={cfg['k']:<3} "
                  f"mse={m['mse']:.4f} var={m['var']:.3f} nmse={m['nmse']:.4f} "
                  f"L0={m['mean_l0']:.1f} dead={m['dead_frac']*100:.1f}% "
                  f"({m['n_dead']}/{m['d_sae']})")
            del sae, acts
            if args.device == "cuda":
                torch.cuda.empty_cache()

    rows.sort(key=lambda r: (r["chain"], r["layer"], stage_key(r["stage"])))
    fields = ["chain", "stage", "layer", "k", "d_sae", "n_tokens",
              "mse", "var", "nmse", "mean_l0", "n_dead", "dead_frac"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "chain": r["chain"], "stage": r["stage"], "layer": r["layer"],
                "k": r["k"], "d_sae": r["d_sae"], "n_tokens": r["n_tokens"],
                "mse": f"{r['mse']:.6f}", "var": f"{r['var']:.6f}",
                "nmse": f"{r['nmse']:.6f}", "mean_l0": f"{r['mean_l0']:.2f}",
                "n_dead": r["n_dead"], "dead_frac": f"{r['dead_frac']:.6f}",
            })
    print(f"\nWrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
