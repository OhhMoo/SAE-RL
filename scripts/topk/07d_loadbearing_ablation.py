"""
07d_loadbearing_ablation.py — Round-trip causal test for the L18 load-bearing set.

The 07b attribution sweep gave per-feature delta_ce at L18 step30. 07c summarised
the quadrant share — load-bearing concentrates in `stable` (62-84% per chain) but
the top of the list is dominated by 1-3 high-density anchor features (density
0.85-0.93) whose single-feature delta_ce is mostly a recon-offset artifact.

This script defines a clean load-bearing set:

    loadbearing-K = top-K features by delta_ce, restricted to alive
                    (density > 1e-4) and below an anchor cap (default 0.5).

and joint-ablates it on GSM8k, against two density-matched controls:

    reweighted-matched-K = K features from the reweighted quadrant,
                           greedy nearest-density paired to loadbearing-K
    random-matched-K     = K features from alive non-loadbearing non-reweighted,
                           greedy nearest-density paired to loadbearing-K

Density-matching uses the same algorithm as 07's v3 control.

Conditions per (chain, stage in {instruct_base, ppo_step30}):
    1. full_recon                — SAE recon spliced at L18
    2. ablate_loadbearing        — recon with loadbearing-K zeroed in z
    3. ablate_reweighted_matched — recon with reweighted-matched-K zeroed
    4. ablate_random_matched     — recon with random-matched-K zeroed

Headline metrics (per chain, step30):
    delta_load     = ce_ablate_loadbearing - ce_full_recon
    delta_rw       = ce_ablate_reweighted_matched - ce_full_recon
    delta_rand     = ce_ablate_random_matched - ce_full_recon
    specificity_lr = delta_load - delta_rw     (load-bearing > reweighted-matched?)
    specificity_lc = delta_load - delta_rand   (load-bearing > random-matched?)

    incremental_load = (delta_load at step30) - (delta_load at instruct_base)
        — were these features load-bearing only after PPO, or already at base?

Pre-registered success criterion (set before running):
  At step30, in flex+strict, delta_load >= 2x max(delta_rw, delta_rand).
  Shuffled is not pre-registered (its load-bearing concentration in `stable` is
  driven by the high-density anchor 3335/6277, which the cap excludes — the
  resulting K may shrink heavily and the test isn't meaningful).

Run from sae_rl/:
    python scripts/07d_loadbearing_ablation.py
    python scripts/07d_loadbearing_ablation.py --k 50 --density_cap 0.5
    python scripts/07d_loadbearing_ablation.py --chains flexible strict
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
INSTRUCT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
STAGE_BASE = "instruct_base"
STAGE_COLD = "ppo_step10"
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
# Set definitions
# ---------------------------------------------------------------------------

def per_feature_decoder_cos(W_a, W_b):
    a = nn.functional.normalize(W_a, dim=0)
    b = nn.functional.normalize(W_b, dim=0)
    return (a * b).sum(dim=0)


def per_feature_encoder_cos(W_a, W_b):
    a = nn.functional.normalize(W_a, dim=1)
    b = nn.functional.normalize(W_b, dim=1)
    return (a * b).sum(dim=1)


def reweighted_indices(chain: str) -> np.ndarray:
    """High dec_cos × low enc_cos at the cold-start transition (median-split).
    Mirrors 06c_probe_l18_decoupling.py and 07_ablation_sae.py."""
    sae_dir = CHAINS[chain]["sae_dir"]
    Wd_a, We_a = load_dec_enc(sae_dir / f"sae_{STAGE_BASE}_layer{LAYER}.pt")
    Wd_b, We_b = load_dec_enc(sae_dir / f"sae_{STAGE_COLD}_layer{LAYER}.pt")
    dcos = per_feature_decoder_cos(Wd_a, Wd_b).numpy()
    ecos = per_feature_encoder_cos(We_a, We_b).numpy()
    return np.where((dcos >= np.median(dcos)) & (ecos < np.median(ecos)))[0]


def load_attribution(chain: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (delta_ce, density, alive_mask) per d_sae index, ordered by feature_idx."""
    path = ROOT / "results" / "l18_attribution.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run 07b_attribution_l18.py first")
    rows: list[dict] = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["chain"] == chain:
                rows.append(r)
    rows.sort(key=lambda r: int(r["feature_idx"]))
    delta = np.array([float(r["delta_ce"]) for r in rows])
    density = np.array([float(r["density"]) for r in rows])
    alive = density > DEAD_THRESHOLD
    return delta, density, alive


def loadbearing_indices(delta: np.ndarray, density: np.ndarray, alive: np.ndarray,
                        k: int, density_cap: float) -> np.ndarray:
    """Top-K features by delta_ce among alive features below the anchor cap."""
    eligible = alive & (density < density_cap) & (delta > 0)
    pool = np.where(eligible)[0]
    if len(pool) < k:
        raise ValueError(f"only {len(pool)} eligible features; reduce k or raise cap")
    order = np.argsort(-delta[pool])
    return pool[order[:k]]


def density_match(anchor_idx: np.ndarray, pool_idx: np.ndarray,
                  density: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    """Greedy nearest-density pairing. For each anchor (in random order), pick
    the closest-density unused pool feature. Returns (matched_idx, diag)."""
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

    matched = np.array(sorted(picked), dtype=np.int64)
    anchor_total = float(density[anchor_idx].sum())
    matched_total = float(density[matched].sum()) if len(matched) else 0.0
    return matched, {
        "n_anchor": int(len(anchor_idx)),
        "n_pool": int(len(pool_idx)),
        "n_matched": int(len(matched)),
        "anchor_summed_density": anchor_total,
        "matched_summed_density": matched_total,
        "density_match_ratio": (matched_total / anchor_total
                                if anchor_total > 0 else float("nan")),
    }


# ---------------------------------------------------------------------------
# Forward hook (matches 07_ablation_sae.SAESpliceHook)
# ---------------------------------------------------------------------------

class SAESpliceHook:
    def __init__(self, model, layer_idx: int, sae: TopKSAE,
                 ablate_idx: np.ndarray | None):
        self.model = model
        self.layer_idx = layer_idx
        self.sae = sae
        self.attn_mask: torch.Tensor | None = None
        self._handle = None
        if ablate_idx is None or len(ablate_idx) == 0:
            self.keep_mask = None
        else:
            keep = torch.ones(sae.d_sae, dtype=torch.float32,
                              device=sae.encoder.weight.device)
            keep[torch.tensor(ablate_idx, dtype=torch.long, device=keep.device)] = 0.0
            self.keep_mask = keep

    def _hook(self, module, inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
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
        self._handle = self.model.model.layers[self.layer_idx].register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---------------------------------------------------------------------------
# CE eval
# ---------------------------------------------------------------------------

def _masked_labels(input_ids, attention_mask):
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return labels


@torch.no_grad()
def eval_ce(model, tokenizer, prompts, hook, device, batch_size, max_length=384):
    losses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="    ce", leave=False):
        enc = tokenizer(prompts[i:i + batch_size], return_tensors="pt",
                        padding=True, truncation=True,
                        max_length=max_length).to(device)
        labels = _masked_labels(enc["input_ids"], enc["attention_mask"])
        if hook is not None:
            hook.attn_mask = enc["attention_mask"]
        out = model(**enc, labels=labels)
        losses.append(out.loss.item())
    return float(np.mean(losses))


# ---------------------------------------------------------------------------
# Per-cell driver
# ---------------------------------------------------------------------------

def run_cell(label: str, model_path: str, sae_path: Path, prompts: list[str],
             load_idx: np.ndarray, rw_idx: np.ndarray, rand_idx: np.ndarray,
             device: str, batch_size: int) -> dict[str, float]:
    print(f"\n[{label}]  model={model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to(device).eval()
    sae, _ = load_sae(sae_path, device)

    out: dict[str, float] = {}
    with SAESpliceHook(model, LAYER, sae, ablate_idx=None) as hook:
        out["full_recon"] = eval_ce(model, tokenizer, prompts, hook, device, batch_size)
    print(f"  full_recon                 CE={out['full_recon']:.4f}")
    with SAESpliceHook(model, LAYER, sae, ablate_idx=load_idx) as hook:
        out["ablate_loadbearing"] = eval_ce(model, tokenizer, prompts, hook, device, batch_size)
    print(f"  ablate_loadbearing         CE={out['ablate_loadbearing']:.4f}  (n={len(load_idx)})")
    with SAESpliceHook(model, LAYER, sae, ablate_idx=rw_idx) as hook:
        out["ablate_reweighted_matched"] = eval_ce(model, tokenizer, prompts, hook, device, batch_size)
    print(f"  ablate_reweighted_matched  CE={out['ablate_reweighted_matched']:.4f}  (n={len(rw_idx)})")
    with SAESpliceHook(model, LAYER, sae, ablate_idx=rand_idx) as hook:
        out["ablate_random_matched"] = eval_ce(model, tokenizer, prompts, hook, device, batch_size)
    print(f"  ablate_random_matched      CE={out['ablate_random_matched']:.4f}  (n={len(rand_idx)})")

    del model, sae, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", nargs="+", default=list(CHAINS.keys()))
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--k", type=int, default=50,
                    help="Top-K load-bearing features (after density cap).")
    ap.add_argument("--density_cap", type=float, default=0.5,
                    help="Exclude features with density >= cap (anchor filter).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output_csv", default="results/l18_loadbearing_ablation.csv")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print("=== L18 load-bearing round-trip ablation ===")
    print(f"chains       : {args.chains}")
    print(f"K            : {args.k}")
    print(f"density_cap  : {args.density_cap}  (anchor filter)")
    print(f"prompts      : {args.n_prompts} GSM8k test (question + answer)")
    print(f"output       : {args.output_csv}\n")

    print("Loading GSM8k test prompts...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    prompts = [f"{ex['question']} {ex['answer']}" for ex in ds][:args.n_prompts]

    rows: list[dict] = []

    for chain in args.chains:
        if chain not in CHAINS:
            print(f"[skip] unknown chain {chain}")
            continue
        cfg = CHAINS[chain]

        # Build the three index sets (defined by step30 attribution).
        delta, density, alive = load_attribution(chain)
        rw_set = reweighted_indices(chain)
        is_rw = np.zeros(len(density), dtype=bool); is_rw[rw_set] = True

        load_idx = loadbearing_indices(delta, density, alive,
                                       k=args.k, density_cap=args.density_cap)
        is_load = np.zeros(len(density), dtype=bool); is_load[load_idx] = True

        rw_pool = rw_set[alive[rw_set] & ~is_load[rw_set]]
        rand_pool = np.where(alive & ~is_rw & ~is_load)[0]

        rw_match_idx, rw_diag = density_match(load_idx, rw_pool, density, args.seed)
        rand_match_idx, rand_diag = density_match(load_idx, rand_pool, density, args.seed + 1)

        load_density_sum = float(density[load_idx].sum())
        quads_for_chain: dict[int, str] = {}
        with open(ROOT / "results" / "l18_attribution.csv") as f:
            for r in csv.DictReader(f):
                if r["chain"] == chain:
                    quads_for_chain[int(r["feature_idx"])] = r["quadrant"]
        load_quads = [quads_for_chain[int(i)] for i in load_idx]
        quad_counts = {q: load_quads.count(q) for q in
                       ("stable", "reweighted", "redirected", "rotated")}

        print(f"\n--- {chain} ---")
        print(f"  loadbearing-K  : K={len(load_idx)}  "
              f"density_sum={load_density_sum:.3f}  "
              f"density_max={float(density[load_idx].max()):.3f}  "
              f"delta_ce_min={float(delta[load_idx].min()):+.4f}  "
              f"delta_ce_max={float(delta[load_idx].max()):+.4f}")
        print(f"  load quadrants : {quad_counts}")
        print(f"  reweighted-pool: |pool|={len(rw_pool)}  matched={rw_diag['n_matched']}  "
              f"density_match={rw_diag['density_match_ratio']:.3f}")
        print(f"  random-pool    : |pool|={len(rand_pool)}  matched={rand_diag['n_matched']}  "
              f"density_match={rand_diag['density_match_ratio']:.3f}")

        for stage in (STAGE_BASE, STAGE_TARGET):
            if stage == STAGE_BASE:
                model_path = INSTRUCT_MODEL
            else:
                step = stage[len("ppo_step"):]
                merged = cfg["merged_dir"] / f"step_{step}"
                if not merged.exists():
                    print(f"[skip] {chain} {stage}: missing merged model")
                    continue
                model_path = str(merged)
            sae_path = cfg["sae_dir"] / f"sae_{stage}_layer{LAYER}.pt"
            if not sae_path.exists():
                print(f"[skip] {chain} {stage}: missing SAE {sae_path.name}")
                continue

            ce = run_cell(f"{chain}/{stage}", model_path, sae_path, prompts,
                          load_idx, rw_match_idx, rand_match_idx,
                          args.device, args.batch_size)

            rows.append({
                "chain": chain,
                "stage": stage,
                "k": len(load_idx),
                "load_quad_stable":     quad_counts["stable"],
                "load_quad_reweighted": quad_counts["reweighted"],
                "load_quad_redirected": quad_counts["redirected"],
                "load_quad_rotated":    quad_counts["rotated"],
                "load_density_sum":  f"{load_density_sum:.4f}",
                "rw_density_match":  f"{rw_diag['density_match_ratio']:.4f}",
                "rand_density_match": f"{rand_diag['density_match_ratio']:.4f}",
                "ce_full_recon":              f"{ce['full_recon']:.4f}",
                "ce_ablate_loadbearing":      f"{ce['ablate_loadbearing']:.4f}",
                "ce_ablate_reweighted_matched": f"{ce['ablate_reweighted_matched']:.4f}",
                "ce_ablate_random_matched":   f"{ce['ablate_random_matched']:.4f}",
                "delta_load":  f"{ce['ablate_loadbearing'] - ce['full_recon']:.4f}",
                "delta_rw":    f"{ce['ablate_reweighted_matched'] - ce['full_recon']:.4f}",
                "delta_rand":  f"{ce['ablate_random_matched'] - ce['full_recon']:.4f}",
                "specificity_load_minus_rw":   f"{ce['ablate_loadbearing'] - ce['ablate_reweighted_matched']:.4f}",
                "specificity_load_minus_rand": f"{ce['ablate_loadbearing'] - ce['ablate_random_matched']:.4f}",
            })

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nDone. Results -> {out_csv}")

    # Console summary
    print(f"\n{'chain':<10s} {'stage':<14s} "
          f"{'full':>7s} {'load':>7s} {'rw':>7s} {'rand':>7s} "
          f"{'d_load':>8s} {'d_rw':>8s} {'d_rand':>8s}")
    print("-" * 90)
    for r in rows:
        print(f"{r['chain']:<10s} {r['stage']:<14s} "
              f"{r['ce_full_recon']:>7s} "
              f"{r['ce_ablate_loadbearing']:>7s} "
              f"{r['ce_ablate_reweighted_matched']:>7s} "
              f"{r['ce_ablate_random_matched']:>7s} "
              f"{r['delta_load']:>8s} {r['delta_rw']:>8s} {r['delta_rand']:>8s}")


if __name__ == "__main__":
    main()
