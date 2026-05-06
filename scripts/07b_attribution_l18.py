"""
07b_attribution_l18.py — Per-feature attribution sweep at L18 step30.

Today's v3 ablation (07_ablation_sae.py → results/l18_ablation.csv) showed
that the L18 reweighted quadrant is the *reorganization* zone but not the
*load-bearing* zone: per unit firing mass, ablating reweighted features at
step30 hurts CE only ~14% as much (flex) / 1.7% as much (strict) as
ablating density-matched non-reweighted features.

That's a negative result on the original mechanism story. To turn it into
a positive one we need to find which features actually carry the L18
behaviour and ask whether they cluster in a different (dec_cos, enc_cos)
quadrant than the reweighted set. This script answers that.

Method (per chain ∈ {flexible, strict, shuffled}):
  1. Load merged ppo_step30 model + ppo_step30 L18 SAE.
  2. Compute baseline ce_full_recon on n_prompts GSM8k test prompts.
  3. For each feature i ∈ [0, d_sae): with the SAE-splice hook active,
     zero only feature i in z, measure ce_i, record delta_ce_i.
  4. Join with per-feature (dec_cos, enc_cos) at the cold-start transition
     (instruct_base → ppo_step10, same as 06c_probe_l18_decoupling.py and
     07_ablation_sae.py:reweighted_indices) and density at step30.
  5. Assign each feature a quadrant by median-split on (dec_cos, enc_cos)
     using the same convention as 06c.

Output: results/l18_attribution.csv with columns
    chain, feature_idx, delta_ce, density, dec_cos, enc_cos, quadrant

Run from sae_rl/:
    python scripts/07b_attribution_l18.py
    python scripts/07b_attribution_l18.py --n_prompts 100
    python scripts/07b_attribution_l18.py --chains flexible --n_prompts 20

Wall-clock budget: ~7168 features × n_prompts CE-eval. With n_prompts=50,
batch_size=4, fp16, on a 0.5B model: ~50-100 ms per per-feature pass.
Single chain ≈ 6-12 min, three chains ≈ 20-40 min.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
LAYER = 18
INSTRUCT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
STAGE_BASE = "instruct_base"
STAGE_COLD = "ppo_step10"
STAGE_TARGET = "ppo_step30"

CHAINS = {
    "flexible": {
        "sae_dir":    ROOT / "checkpoints" / "saes",
        "merged_dir": ROOT / "checkpoints" / "ppo_merged",
    },
    "strict": {
        "sae_dir":    ROOT / "checkpoints" / "saes_strict",
        "merged_dir": ROOT / "checkpoints" / "ppo_merged_strict",
    },
    "shuffled": {
        "sae_dir":    ROOT / "checkpoints" / "saes_shuffled",
        "merged_dir": ROOT / "checkpoints" / "ppo_merged_shuffled",
    },
}

DEAD_THRESHOLD = 1e-4  # matches 06_analysis_sae.py


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


def load_dec_enc(path: Path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    return sd["decoder.weight"].float(), sd["encoder.weight"].float()


# ---------------------------------------------------------------------------
# Quadrant geometry — mirrors 06c_probe_l18_decoupling.py
# ---------------------------------------------------------------------------

def per_feature_decoder_cos(W_a, W_b):
    a = nn.functional.normalize(W_a, dim=0)
    b = nn.functional.normalize(W_b, dim=0)
    return (a * b).sum(dim=0)


def per_feature_encoder_cos(W_a, W_b):
    a = nn.functional.normalize(W_a, dim=1)
    b = nn.functional.normalize(W_b, dim=1)
    return (a * b).sum(dim=1)


def quadrant_geometry(chain: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (dec_cos, enc_cos, quadrant) per feature.

    Quadrant codes (mirroring 06c):
      "stable"      = high dec, high enc
      "reweighted"  = high dec, low  enc  ← already known to be reorg zone
      "redirected"  = low  dec, high enc
      "rotated"     = low  dec, low  enc
    Median-split on each axis at the cold-start transition base→step10.
    """
    sae_dir = CHAINS[chain]["sae_dir"]
    Wd_a, We_a = load_dec_enc(sae_dir / f"sae_{STAGE_BASE}_layer{LAYER}.pt")
    Wd_b, We_b = load_dec_enc(sae_dir / f"sae_{STAGE_COLD}_layer{LAYER}.pt")
    dcos = per_feature_decoder_cos(Wd_a, Wd_b).numpy()
    ecos = per_feature_encoder_cos(We_a, We_b).numpy()
    dec_med = float(np.median(dcos))
    enc_med = float(np.median(ecos))
    high_d = dcos >= dec_med
    high_e = ecos >= enc_med
    quad = np.empty(len(dcos), dtype=object)
    quad[ high_d &  high_e] = "stable"
    quad[ high_d & ~high_e] = "reweighted"
    quad[~high_d &  high_e] = "redirected"
    quad[~high_d & ~high_e] = "rotated"
    return dcos, ecos, quad


def load_density(chain: str, stage: str, layer: int = LAYER) -> np.ndarray:
    cache = ROOT / "results" / "cache" / f"density_{chain}_L{layer}_{stage}.pt"
    if not cache.exists():
        raise FileNotFoundError(
            f"missing density cache {cache}; run "
            f"`scripts/06_analysis_sae.py feature_lifecycle` first to populate."
        )
    return torch.load(cache, weights_only=False)["density"].numpy()


# ---------------------------------------------------------------------------
# Forward hook with mutable single-feature mask
# ---------------------------------------------------------------------------

class SAESpliceMaskHook:
    """SAE-splice at `layer_idx`, with a mutable per-feature keep_mask the
    caller can update in place between forward passes (avoids re-registering
    the hook 7168 times).

    If `keep_mask is None` → full reconstruction (no ablation).
    Otherwise z is multiplied by keep_mask elementwise before decoding.
    """

    def __init__(self, model, layer_idx: int, sae: TopKSAE):
        self.model = model
        self.layer_idx = layer_idx
        self.sae = sae
        self.attn_mask: torch.Tensor | None = None
        self.keep_mask: torch.Tensor | None = None  # shape (d_sae,)
        self._handle = None

    def _hook(self, module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = (out[0] if is_tuple else out)
        B, T, D = h.shape
        h_f = h.float().reshape(B * T, D)
        z = self.sae.encode(h_f)
        if self.keep_mask is not None:
            z = z * self.keep_mask.unsqueeze(0)
        recon = self.sae.decoder(z).reshape(B, T, D).to(h.dtype)
        if self.attn_mask is not None:
            real = self.attn_mask.unsqueeze(-1).bool()
            patched = torch.where(real, recon, h)
        else:
            patched = recon
        return (patched,) + out[1:] if is_tuple else patched

    def __enter__(self):
        self._handle = self.model.model.layers[self.layer_idx].register_forward_hook(
            self._hook
        )
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---------------------------------------------------------------------------
# CE eval — pretokenize once for speed across 7168 sweeps
# ---------------------------------------------------------------------------

def _masked_labels(input_ids, attention_mask):
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return labels


def pretokenize(tokenizer, prompts: list[str], device: str, batch_size: int,
                max_length: int = 384):
    """Tokenize once and return a list of (input_ids, attention_mask, labels)
    batches, all on `device`. Avoids re-tokenizing on every per-feature pass."""
    batches = []
    for i in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[i:i + batch_size], return_tensors="pt",
                        padding=True, truncation=True,
                        max_length=max_length).to(device)
        labels = _masked_labels(enc["input_ids"], enc["attention_mask"])
        batches.append((enc["input_ids"], enc["attention_mask"], labels))
    return batches


@torch.no_grad()
def eval_ce_pretok(model, batches, hook: SAESpliceMaskHook | None) -> float:
    losses = []
    for input_ids, attn_mask, labels in batches:
        if hook is not None:
            hook.attn_mask = attn_mask
        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        losses.append(out.loss.item())
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Per-chain attribution sweep
# ---------------------------------------------------------------------------

def run_chain(chain: str, n_prompts: int, batch_size: int, prompts: list[str],
              device: str) -> list[dict]:
    cfg = CHAINS[chain]
    step = STAGE_TARGET[len("ppo_step"):]
    model_path = cfg["merged_dir"] / f"step_{step}"
    sae_path = cfg["sae_dir"] / f"sae_{STAGE_TARGET}_layer{LAYER}.pt"
    if not model_path.exists():
        print(f"[skip] {chain}: missing merged model {model_path}")
        return []
    if not sae_path.exists():
        print(f"[skip] {chain}: missing SAE {sae_path}")
        return []

    print(f"\n=== {chain} / {STAGE_TARGET} ===")
    print(f"  model = {model_path}")
    print(f"  sae   = {sae_path.name}")

    dcos, ecos, quad = quadrant_geometry(chain)
    density = load_density(chain, STAGE_TARGET)
    d_sae = len(dcos)
    assert len(density) == d_sae, (len(density), d_sae)

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.float16
    ).to(device).eval()
    sae, _cfg = load_sae(sae_path, device)

    batches = pretokenize(tokenizer, prompts, device, batch_size)
    print(f"  pretokenized {n_prompts} prompts → {len(batches)} batches")

    # Persistent keep-mask of ones; we toggle one slot off, eval, then back on.
    keep = torch.ones(d_sae, dtype=torch.float32, device=device)

    rows: list[dict] = []
    with SAESpliceMaskHook(model, LAYER, sae) as hook:
        # Baseline: full reconstruction (no ablation).
        hook.keep_mask = None
        ce_full = eval_ce_pretok(model, batches, hook)
        print(f"  ce_full_recon = {ce_full:.4f}")

        # Per-feature sweep.
        hook.keep_mask = keep
        alive_count = int((density > DEAD_THRESHOLD).sum())
        print(f"  sweeping {d_sae} features ({alive_count} alive at step30)...")
        for i in tqdm(range(d_sae), desc=f"  {chain}", leave=False):
            keep[i] = 0.0
            ce_i = eval_ce_pretok(model, batches, hook)
            keep[i] = 1.0
            rows.append({
                "chain": chain,
                "feature_idx": int(i),
                "delta_ce": float(ce_i - ce_full),
                "ce": float(ce_i),
                "density": float(density[i]),
                "dec_cos": float(dcos[i]),
                "enc_cos": float(ecos[i]),
                "quadrant": str(quad[i]),
            })

    # Per-chain summary diagnostics (printed; full data lands in CSV).
    deltas = np.array([r["delta_ce"] for r in rows])
    densities = np.array([r["density"] for r in rows])
    quadrants = np.array([r["quadrant"] for r in rows])

    dead_mask = densities <= DEAD_THRESHOLD
    if dead_mask.any():
        d_dead = deltas[dead_mask]
        n_within = int(np.sum(np.abs(d_dead) < 1e-3))
        print(f"  sanity: dead-feature delta_ce in [{d_dead.min():+.5f}, "
              f"{d_dead.max():+.5f}], {n_within}/{len(d_dead)} within ±1e-3")

    print("  per-quadrant alive-feature stats (mean / max delta_ce):")
    for q in ("stable", "reweighted", "redirected", "rotated"):
        m = (quadrants == q) & ~dead_mask
        if m.any():
            print(f"    {q:<10s} n_alive={int(m.sum())}  "
                  f"mean={deltas[m].mean():+.5f}  max={deltas[m].max():+.5f}")

    top10 = np.argsort(-deltas)[:10]
    print("  top-10 features by delta_ce:")
    for j in top10:
        r = rows[j]
        print(f"    idx={r['feature_idx']:5d}  delta={r['delta_ce']:+.4f}  "
              f"dens={r['density']:.4f}  quad={r['quadrant']}")

    del model, sae, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", nargs="+", default=list(CHAINS.keys()),
                    help="Subset of chains to run.")
    ap.add_argument("--n_prompts", type=int, default=50,
                    help="GSM8k test prompts per per-feature CE eval.")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--output_csv", default="results/l18_attribution.csv")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print("=== L18 per-feature attribution sweep ===")
    print(f"chains   : {args.chains}")
    print(f"layer    : L{LAYER}")
    print(f"stage    : {STAGE_TARGET}")
    print(f"prompts  : {args.n_prompts} GSM8k test")
    print(f"output   : {args.output_csv}\n")

    print("Loading GSM8k test prompts (question + answer)...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    prompts_full = [f"{ex['question']} {ex['answer']}" for ex in ds]
    prompts = prompts_full[:args.n_prompts]

    all_rows: list[dict] = []
    for chain in args.chains:
        if chain not in CHAINS:
            print(f"[skip] unknown chain {chain}")
            continue
        all_rows.extend(run_chain(chain, args.n_prompts, args.batch_size,
                                   prompts, args.device))

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["chain", "feature_idx", "delta_ce", "ce", "density",
                  "dec_cos", "enc_cos", "quadrant"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({
                "chain": r["chain"],
                "feature_idx": r["feature_idx"],
                "delta_ce": f"{r['delta_ce']:.6f}",
                "ce": f"{r['ce']:.6f}",
                "density": f"{r['density']:.6f}",
                "dec_cos": f"{r['dec_cos']:.6f}",
                "enc_cos": f"{r['enc_cos']:.6f}",
                "quadrant": r["quadrant"],
            })
    print(f"\nDone. {len(all_rows)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
