"""
07e_loadbearing_ablation_position.py — BOS / non-BOS decomposition of 07d Δ_load.

08 (top-K activating contexts) showed that 32/50 of shuffled's load-bearing
features never fire on a non-BOS token in 100 GSM8k prompts; their entire
07b Δce comes from position-0 anomalies. This script causally validates
that observation: same load-bearing K per chain (K=50, density_cap=0.5),
same prompts, same SAE — but the keep_mask is gated by position, ablating
features only at the BOS position (position-0 of each sequence) or only at
non-BOS positions.

Predicted (if the BOS-anomaly story is right):
  strict   : Δ_load_nonbos / Δ_load_all >> Δ_load_bos / Δ_load_all
  shuffled : the reverse — most of Δ_load is recovered by BOS-only ablation
  flex     : intermediate

Cells per chain at ppo_step30 (n_prompts=100, GSM8k test q+a):
  A. full_recon                — SAE recon spliced at L18, no ablation
  B. ablate_load all positions — recovers 07d's Δ_load (sanity)
  C. ablate_load BOS-only      — only zero z at first-real-token of each seq
  D. ablate_load non-BOS       — zero z everywhere except first-real-token
  E. ablate_rand all positions — sanity, recovers 07d's Δ_rand
  F. ablate_rand BOS-only      — control: is BOS-only ablation generically painful?
  G. ablate_rand non-BOS       — control completes the matched decomposition

Run from sae_rl/:
    python scripts/07e_loadbearing_ablation_position.py
    python scripts/07e_loadbearing_ablation_position.py --chains shuffled
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


ROOT = Path(__file__).resolve().parents[2]
LAYER = 18
STAGE_TARGET = "ppo_step30"
DEAD_THRESHOLD = 1e-4

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


# ---------------------------------------------------------------------------
# TopK SAE — copied from 07d/05_train_sae.
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
# Index sets (reuse 07d's logic so K is byte-identical).
# ---------------------------------------------------------------------------

def load_dec_enc(path: Path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    return sd["decoder.weight"].float(), sd["encoder.weight"].float()


def per_feature_decoder_cos(W_a, W_b):
    a = nn.functional.normalize(W_a, dim=0); b = nn.functional.normalize(W_b, dim=0)
    return (a * b).sum(dim=0)


def per_feature_encoder_cos(W_a, W_b):
    a = nn.functional.normalize(W_a, dim=1); b = nn.functional.normalize(W_b, dim=1)
    return (a * b).sum(dim=1)


def reweighted_indices(chain: str) -> np.ndarray:
    sae_dir = CHAINS[chain]["sae_dir"]
    Wd_a, We_a = load_dec_enc(sae_dir / f"sae_instruct_base_layer{LAYER}.pt")
    Wd_b, We_b = load_dec_enc(sae_dir / f"sae_ppo_step10_layer{LAYER}.pt")
    dcos = per_feature_decoder_cos(Wd_a, Wd_b).numpy()
    ecos = per_feature_encoder_cos(We_a, We_b).numpy()
    return np.where((dcos >= np.median(dcos)) & (ecos < np.median(ecos)))[0]


def load_attribution(chain: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = ROOT / "results" / "l18_attribution.csv"
    rows = [r for r in csv.DictReader(open(path)) if r["chain"] == chain]
    rows.sort(key=lambda r: int(r["feature_idx"]))
    delta = np.array([float(r["delta_ce"]) for r in rows])
    density = np.array([float(r["density"]) for r in rows])
    return delta, density, density > DEAD_THRESHOLD


def loadbearing_indices(delta, density, alive, k, density_cap):
    eligible = alive & (density < density_cap) & (delta > 0)
    pool = np.where(eligible)[0]
    if len(pool) < k:
        raise ValueError(f"only {len(pool)} eligible features; reduce k or raise cap")
    order = np.argsort(-delta[pool])
    return pool[order[:k]]


def density_match(anchor_idx, pool_idx, density, seed) -> np.ndarray:
    """Greedy nearest-density pairing — same algorithm as 07d v3."""
    pool_density = density[pool_idx]
    order = np.argsort(pool_density, kind="stable")
    pool_sorted_idx = pool_idx[order]
    pool_sorted_density = pool_density[order]
    available = np.ones(len(pool_sorted_idx), dtype=bool)
    picked: list[int] = []
    g = np.random.default_rng(seed)
    perm = g.permutation(len(anchor_idx))
    for i in perm:
        target = float(density[anchor_idx[i]])
        avail_local = np.where(available)[0]
        if len(avail_local) == 0:
            break
        nearest = avail_local[np.argmin(np.abs(pool_sorted_density[avail_local] - target))]
        picked.append(int(pool_sorted_idx[nearest]))
        available[nearest] = False
    return np.array(sorted(picked), dtype=np.int64)


# ---------------------------------------------------------------------------
# Position-aware splice hook.
# ---------------------------------------------------------------------------

class SAESplicePosHook:
    """Like 07d.SAESpliceHook but the keep_mask can be gated by position
    via a string mode. Position-mask is built per batch from the
    attention_mask so it's robust to left/right padding.

    position_mode ∈ {"all", "bos_only", "non_bos"}:
      * "all"      — keep_mask applied at every position (matches 07d)
      * "bos_only" — keep_mask applied ONLY at the first-real-token of each
                     sequence; other positions get full SAE recon (no ablation)
      * "non_bos"  — keep_mask applied at every real token EXCEPT the first;
                     first-real-token gets full SAE recon
    """

    def __init__(self, model, layer_idx, sae, ablate_idx, position_mode="all"):
        self.model, self.layer_idx, self.sae = model, layer_idx, sae
        self.position_mode = position_mode
        self.attn_mask: torch.Tensor | None = None
        self._handle = None
        if ablate_idx is None or len(ablate_idx) == 0:
            self.keep_mask = None
        else:
            keep = torch.ones(sae.d_sae, dtype=torch.float32,
                              device=sae.encoder.weight.device)
            keep[torch.tensor(ablate_idx, dtype=torch.long, device=keep.device)] = 0.0
            self.keep_mask = keep

    def _position_mask(self, attn: torch.Tensor) -> torch.Tensor | None:
        """Returns (B, T) bool: True where the keep_mask should be applied.
        Returns None for mode="all" — caller skips the gating branch."""
        if self.position_mode == "all":
            return None
        cs = attn.cumsum(dim=1)
        is_first_real = (cs == 1) & (attn == 1)              # (B, T) bool
        if self.position_mode == "bos_only":
            return is_first_real
        if self.position_mode == "non_bos":
            return (attn == 1) & (~is_first_real)
        raise ValueError(f"unknown position_mode={self.position_mode}")

    def _hook(self, module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        B, T, D = h.shape
        h_f = h.float().reshape(B * T, D)
        z = self.sae.encode(h_f)                             # (B*T, d_sae)
        if self.keep_mask is not None:
            pos_mask = self._position_mask(self.attn_mask) if self.attn_mask is not None else None
            if pos_mask is None:
                z = z * self.keep_mask.unsqueeze(0)          # ablate everywhere
            else:
                # Apply keep_mask only at gated positions; identity elsewhere.
                pos_mask = pos_mask.to(z.device)
                z = z.reshape(B, T, -1)
                eff = torch.where(
                    pos_mask.unsqueeze(-1),                  # (B, T, 1)
                    self.keep_mask.view(1, 1, -1).expand(B, T, -1),
                    torch.ones_like(self.keep_mask).view(1, 1, -1).expand(B, T, -1),
                )                                            # (B, T, d_sae)
                z = (z * eff).reshape(B * T, -1)
        recon = self.sae.decoder(z).reshape(B, T, D).to(h.dtype)
        if self.attn_mask is not None:
            real = self.attn_mask.unsqueeze(-1).bool()
            patched = torch.where(real, recon, h)
        else:
            patched = recon
        return (patched,) + out[1:] if is_tuple else patched

    def __enter__(self):
        self._handle = self.model.model.layers[self.layer_idx].register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---------------------------------------------------------------------------
# CE eval (mirrors 07d).
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_ce(model, tokenizer, prompts, hook, device, batch_size, max_length=384):
    losses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="    ce", leave=False):
        enc = tokenizer(prompts[i:i + batch_size], return_tensors="pt",
                        padding=True, truncation=True,
                        max_length=max_length).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        if hook is not None:
            hook.attn_mask = enc["attention_mask"]
        out = model(**enc, labels=labels)
        losses.append(out.loss.item())
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Per-chain driver.
# ---------------------------------------------------------------------------

def run_chain(chain: str, args, prompts: list[str]) -> dict | None:
    cfg = CHAINS[chain]
    step = STAGE_TARGET[len("ppo_step"):]
    merged = cfg["merged_dir"] / f"step_{step}"
    sae_path = cfg["sae_dir"] / f"sae_{STAGE_TARGET}_layer{LAYER}.pt"
    if not merged.exists() or not sae_path.exists():
        print(f"[skip] {chain}: missing merged or SAE")
        return None

    delta, density, alive = load_attribution(chain)
    rw_set = reweighted_indices(chain)
    is_rw = np.zeros(len(density), dtype=bool); is_rw[rw_set] = True
    load_idx = loadbearing_indices(delta, density, alive,
                                   k=args.k, density_cap=args.density_cap)
    is_load = np.zeros(len(density), dtype=bool); is_load[load_idx] = True

    rand_pool = np.where(alive & ~is_rw & ~is_load)[0]
    rand_idx = density_match(load_idx, rand_pool, density, args.seed + 1)

    print(f"\n--- {chain} ---")
    print(f"  load_idx K={len(load_idx)}  density_max={float(density[load_idx].max()):.3f}")
    print(f"  rand_idx K={len(rand_idx)}  density_match_ratio="
          f"{float(density[rand_idx].sum() / density[load_idx].sum()):.3f}")
    if chain == "strict":
        assert 1622 in set(int(x) for x in load_idx), \
            "strict load_idx drifted from 07d (1622 missing) — selection bug"

    tokenizer = AutoTokenizer.from_pretrained(str(merged))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(merged), torch_dtype=torch.float16
    ).to(args.device).eval()
    sae, _ = load_sae(sae_path, args.device)

    def cell(label, ablate_idx, mode):
        with SAESplicePosHook(model, LAYER, sae, ablate_idx, position_mode=mode) as hook:
            ce = eval_ce(model, tokenizer, prompts, hook, args.device, args.batch_size)
        print(f"  {label:<26s} CE={ce:.4f}")
        return ce

    ce_full = cell("full_recon", None, "all")
    ce_load_all = cell("ablate_load all", load_idx, "all")
    ce_load_bos = cell("ablate_load BOS-only", load_idx, "bos_only")
    ce_load_non = cell("ablate_load non-BOS", load_idx, "non_bos")
    ce_rand_all = cell("ablate_rand all", rand_idx, "all")
    ce_rand_bos = cell("ablate_rand BOS-only", rand_idx, "bos_only")
    ce_rand_non = cell("ablate_rand non-BOS", rand_idx, "non_bos")

    del model, sae, tokenizer
    if args.device == "cuda":
        torch.cuda.empty_cache()

    d_load_all = ce_load_all - ce_full
    d_load_bos = ce_load_bos - ce_full
    d_load_non = ce_load_non - ce_full
    d_rand_all = ce_rand_all - ce_full
    d_rand_bos = ce_rand_bos - ce_full
    d_rand_non = ce_rand_non - ce_full

    def share(num, denom):
        return float(num / denom) if abs(denom) > 1e-6 else float("nan")

    return {
        "chain": chain,
        "stage": STAGE_TARGET,
        "k": len(load_idx),
        "ce_full":      f"{ce_full:.4f}",
        "ce_load_all":  f"{ce_load_all:.4f}",
        "ce_load_bos":  f"{ce_load_bos:.4f}",
        "ce_load_non":  f"{ce_load_non:.4f}",
        "ce_rand_all":  f"{ce_rand_all:.4f}",
        "ce_rand_bos":  f"{ce_rand_bos:.4f}",
        "ce_rand_non":  f"{ce_rand_non:.4f}",
        "delta_load_all": f"{d_load_all:+.4f}",
        "delta_load_bos": f"{d_load_bos:+.4f}",
        "delta_load_non": f"{d_load_non:+.4f}",
        "delta_rand_all": f"{d_rand_all:+.4f}",
        "delta_rand_bos": f"{d_rand_bos:+.4f}",
        "delta_rand_non": f"{d_rand_non:+.4f}",
        "bos_share_load":   f"{share(d_load_bos, d_load_all):.3f}",
        "nonbos_share_load":f"{share(d_load_non, d_load_all):.3f}",
        "bos_share_rand":   f"{share(d_rand_bos, d_rand_all):.3f}",
        "nonbos_share_rand":f"{share(d_rand_non, d_rand_all):.3f}",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", nargs="+", default=list(CHAINS.keys()))
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--density_cap", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output_csv", default="results/l18_loadbearing_ablation_position.csv")
    args = ap.parse_args()

    print("=== L18 load-bearing position-conditional ablation (07e) ===")
    print(f"chains       : {args.chains}")
    print(f"K            : {args.k}     density_cap : {args.density_cap}")
    print(f"prompts      : {args.n_prompts} GSM8k test (q + a)")
    print(f"output       : {args.output_csv}\n")

    print("Loading GSM8k test prompts...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    prompts = [f"{ex['question']} {ex['answer']}" for ex in ds][:args.n_prompts]

    rows: list[dict] = []
    for chain in args.chains:
        if chain not in CHAINS:
            print(f"[skip] unknown chain {chain}"); continue
        r = run_chain(chain, args, prompts)
        if r is not None:
            rows.append(r)

    if not rows:
        print("no rows produced"); return

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {out_csv}")

    # Console summary table.
    print(f"\n{'chain':<10s} {'Δ_load_all':>11s} {'Δ_load_bos':>11s} {'Δ_load_non':>11s} "
          f"{'bos_share':>10s} {'non_share':>10s} | "
          f"{'Δ_rand_all':>11s} {'Δ_rand_bos':>11s} {'Δ_rand_non':>11s} "
          f"{'r_bos_sh':>9s} {'r_non_sh':>9s}")
    print("-" * 130)
    for r in rows:
        print(f"{r['chain']:<10s} "
              f"{r['delta_load_all']:>11s} {r['delta_load_bos']:>11s} {r['delta_load_non']:>11s} "
              f"{r['bos_share_load']:>10s} {r['nonbos_share_load']:>10s} | "
              f"{r['delta_rand_all']:>11s} {r['delta_rand_bos']:>11s} {r['delta_rand_non']:>11s} "
              f"{r['bos_share_rand']:>9s} {r['nonbos_share_rand']:>9s}")


if __name__ == "__main__":
    main()
