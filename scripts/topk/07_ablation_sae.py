"""
07_ablation_sae.py — Causal intervention on L18 reweighted-quadrant features.

The Apr 30e probe identified the L18 reweighted quadrant (high decoder cosine,
low encoder cosine across the cold-start transition) as the active reorganization
zone of L18: PPO preserves residual-stream directions while rewiring input
selectivity. The May 2 shuffled scoring confirmed the mechanism is reward-
invariant (P3 |corr|=0.015, P4 ratio=4.14x — both clean PPO-pressure-band hits).

This script runs the causal validation reviewers will ask for: ablate the
reweighted-quadrant features in the SAE-spliced forward pass and measure GSM8k
CE-loss degradation against three controls.

Design (per devlog 2026-05-02):

  Conditions per (chain, stage):
    1. unpatched         — model alone, no hook                (upper bound)
    2. full_recon        — full SAE recon spliced at L18, no masking
    3. ablate_reweighted — SAE recon with reweighted-quadrant indices zeroed
    4. ablate_random     — SAE recon with same |set| zeroed, **density-rank
                            stratified** matched control: drawn from
                            non-reweighted features so the per-decile density
                            distribution matches the reweighted set at the
                            *eval stage*. Avoids the v1 bias where uniform
                            sampling hit more high-density features than the
                            (low-density-skewed) reweighted set.

  Stages per chain:
    a) instruct_base     — Qwen/Qwen2.5-0.5B-Instruct + chain's instruct_base SAE
    b) ppo_step30        — chain's merged ppo_step30 model + chain's ppo_step30 SAE

  Chains: flexible, strict, shuffled.

  Reweighted indices defined per chain by the median-split on (dec_cos, enc_cos)
  at the cold-start transition (instruct_base → ppo_step10), matching
  06c_probe_l18_decoupling.py exactly. Random control is recomputed per
  (chain, stage) using `results/cache/density_<chain>_L18_<stage>.pt` (populated
  by 06_analysis_sae.py's feature_lifecycle subcommand — must exist).

  Headline metric:
    specificity = ce_ablate_reweighted - ce_ablate_random

  Cross-stage differencing:
    incremental = (ce_ablate_reweighted_step30 - ce_full_recon_step30)
                - (ce_ablate_reweighted_base   - ce_full_recon_base)
    isolates the behavioural cost of *PPO-acquired* L18 structure.

CE is computed on (question + " " + answer) pairs from GSM8k test, so the
answer tokens (where the math content lives) contribute to the loss.

Output: results/l18_ablation.csv

Run from sae_rl/:
    python scripts/07_ablation_sae.py
    python scripts/07_ablation_sae.py --n_prompts 100   # tighter CIs
    python scripts/07_ablation_sae.py --chains flexible shuffled
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
# Reweighted-quadrant indices — mirrors 06c_probe_l18_decoupling.py
# ---------------------------------------------------------------------------

def per_feature_decoder_cos(W_a, W_b):
    a = nn.functional.normalize(W_a, dim=0)
    b = nn.functional.normalize(W_b, dim=0)
    return (a * b).sum(dim=0)


def per_feature_encoder_cos(W_a, W_b):
    a = nn.functional.normalize(W_a, dim=1)
    b = nn.functional.normalize(W_b, dim=1)
    return (a * b).sum(dim=1)


def reweighted_indices(chain: str) -> torch.Tensor:
    """High dec_cos × low enc_cos at the cold-start transition (median-split)."""
    sae_dir = CHAINS[chain]["sae_dir"]
    Wd_a, We_a = load_dec_enc(sae_dir / f"sae_{STAGE_BASE}_layer{LAYER}.pt")
    Wd_b, We_b = load_dec_enc(sae_dir / f"sae_{STAGE_COLD}_layer{LAYER}.pt")
    dcos = per_feature_decoder_cos(Wd_a, Wd_b).numpy()
    ecos = per_feature_encoder_cos(We_a, We_b).numpy()
    dec_med = float(np.median(dcos))
    enc_med = float(np.median(ecos))
    mask = (dcos >= dec_med) & (ecos < enc_med)
    return torch.tensor(np.where(mask)[0], dtype=torch.long)


def load_density(chain: str, stage: str, layer: int = LAYER) -> np.ndarray:
    cache = ROOT / "results" / "cache" / f"density_{chain}_L{layer}_{stage}.pt"
    if not cache.exists():
        raise FileNotFoundError(
            f"missing density cache {cache}; run "
            f"`scripts/06_analysis_sae.py feature_lifecycle` first to populate."
        )
    return torch.load(cache, weights_only=False)["density"].numpy()


DEAD_THRESHOLD = 1e-4  # matches DEAD_THRESHOLD in 06_analysis_sae.py


def density_matched_indices(reweighted: torch.Tensor, density: np.ndarray,
                            seed: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Greedy nearest-density pairing on the alive subset of both sides.

    Why filter to alive: dead features (density < DEAD_THRESHOLD) have z=0 by
    TopK construction, so zeroing them in the splice is a no-op — they cannot
    contribute to the ablation effect. Including them just creates pairing
    headaches: the reweighted set typically has more dead features than the
    non-reweighted pool, so dead rw features end up paired with low-density
    *alive* non-rw features and inflate the control's density mass.

    Returns (rw_alive_idx, ctrl_idx, diag). The caller should ablate
    rw_alive_idx, not the full reweighted set.
    """
    d_sae = len(density)
    rw_arr_full = np.array(reweighted.tolist(), dtype=np.int64)
    is_rw = np.zeros(d_sae, dtype=bool)
    is_rw[rw_arr_full] = True

    alive_mask = density > DEAD_THRESHOLD
    rw_alive_idx = rw_arr_full[alive_mask[rw_arr_full]]
    non_rw_alive = np.where(alive_mask & ~is_rw)[0]

    non_rw_density = density[non_rw_alive]
    order = np.argsort(non_rw_density, kind="stable")
    non_rw_sorted_idx = non_rw_alive[order]
    non_rw_sorted_density = non_rw_density[order]

    n_pool = len(non_rw_sorted_idx)
    available = np.ones(n_pool, dtype=bool)
    picked: list[int] = []

    g = np.random.default_rng(seed)
    perm = g.permutation(len(rw_alive_idx))
    for i in perm:
        target = float(density[rw_alive_idx[i]])
        avail_local = np.where(available)[0]
        if len(avail_local) == 0:
            break
        avail_density = non_rw_sorted_density[avail_local]
        nearest = avail_local[np.argmin(np.abs(avail_density - target))]
        picked.append(int(non_rw_sorted_idx[nearest]))
        available[nearest] = False

    rw_total = float(density[rw_alive_idx].sum())
    ctrl_total = float(density[np.array(picked)].sum()) if picked else 0.0
    diag = {
        "n_rw_full": int(len(rw_arr_full)),
        "n_rw_alive": int(len(rw_alive_idx)),
        "n_rw_dead_dropped": int(len(rw_arr_full) - len(rw_alive_idx)),
        "n_non_rw_alive_pool": int(n_pool),
        "rw_summed_density": rw_total,
        "ctrl_summed_density": ctrl_total,
        "density_match_ratio": (ctrl_total / rw_total
                                if rw_total > 0 else float("nan")),
    }
    return (torch.tensor(sorted(rw_alive_idx.tolist()), dtype=torch.long),
            torch.tensor(sorted(picked), dtype=torch.long),
            diag)


# ---------------------------------------------------------------------------
# Forward hook: SAE recon (optionally with masked features) at one layer
# ---------------------------------------------------------------------------

class SAESpliceHook:
    """
    Splices SAE reconstruction at `layer_idx` of `model`, optionally zeroing
    a chosen index set in the post-encode z. Real-token positions only
    (matches eval_sae.py's _run_with_replacement).

    Caller sets `attn_mask` per batch; the hook reads it on each forward.
    """

    def __init__(self, model, layer_idx: int, sae: TopKSAE,
                 ablate_idx: torch.Tensor | None):
        self.model = model
        self.layer_idx = layer_idx
        self.sae = sae
        self.attn_mask: torch.Tensor | None = None
        self._handle = None

        if ablate_idx is None:
            self.keep_mask = None
        else:
            keep = torch.ones(sae.d_sae, dtype=torch.float32,
                              device=sae.encoder.weight.device)
            keep[ablate_idx.to(keep.device)] = 0.0
            self.keep_mask = keep  # shape (d_sae,)

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
# CE loss eval
# ---------------------------------------------------------------------------

def _masked_labels(input_ids, attention_mask):
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return labels


@torch.no_grad()
def eval_ce(model, tokenizer, prompts: list[str], hook: SAESpliceHook | None,
            device: str, batch_size: int = 4, max_length: int = 384) -> float:
    losses = []
    for i in tqdm(range(0, len(prompts), batch_size),
                  desc="    ce", leave=False):
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

def run_cell(chain: str, stage: str, model_path: str, sae_path: Path,
             prompts: list[str], rw_idx: torch.Tensor, rand_idx: torch.Tensor,
             device: str, batch_size: int) -> dict[str, float]:
    """Run all 4 conditions for one (chain, stage) cell. Returns CE per condition."""
    print(f"\n[{chain} / {stage}]  model={model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to(device).eval()
    sae, _cfg = load_sae(sae_path, device)

    out: dict[str, float] = {}

    out["unpatched"] = eval_ce(model, tokenizer, prompts, None, device, batch_size)
    print(f"  unpatched         CE={out['unpatched']:.4f}")

    with SAESpliceHook(model, LAYER, sae, ablate_idx=None) as hook:
        out["full_recon"] = eval_ce(model, tokenizer, prompts, hook, device,
                                    batch_size)
    print(f"  full_recon        CE={out['full_recon']:.4f}")

    with SAESpliceHook(model, LAYER, sae, ablate_idx=rw_idx) as hook:
        out["ablate_reweighted"] = eval_ce(model, tokenizer, prompts, hook,
                                           device, batch_size)
    print(f"  ablate_reweighted CE={out['ablate_reweighted']:.4f}  "
          f"(n_ablated={len(rw_idx)})")

    with SAESpliceHook(model, LAYER, sae, ablate_idx=rand_idx) as hook:
        out["ablate_random"] = eval_ce(model, tokenizer, prompts, hook,
                                       device, batch_size)
    print(f"  ablate_random     CE={out['ablate_random']:.4f}  "
          f"(n_ablated={len(rand_idx)})")

    del model, sae, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", nargs="+", default=list(CHAINS.keys()),
                    help="Subset of chains to run.")
    ap.add_argument("--n_prompts", type=int, default=100,
                    help="GSM8k test prompts per condition.")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for matched-random index draw (per chain × stage).")
    ap.add_argument("--output_csv", default="results/l18_ablation.csv")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print("=== L18 reweighted-quadrant ablation ===")
    print(f"chains      : {args.chains}")
    print(f"layer       : L{LAYER}")
    print(f"stages      : {STAGE_BASE} (Qwen instruct), {STAGE_TARGET}")
    print(f"prompts     : {args.n_prompts} GSM8k test")
    print(f"output      : {args.output_csv}\n")

    print("Loading GSM8k test prompts (question + answer)...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    prompts_full = [f"{ex['question']} {ex['answer']}" for ex in ds]
    prompts = prompts_full[:args.n_prompts]

    rows: list[dict] = []

    for chain in args.chains:
        if chain not in CHAINS:
            print(f"[skip] unknown chain {chain}")
            continue
        cfg = CHAINS[chain]

        rw_idx = reweighted_indices(chain)
        Wd, _We = load_dec_enc(cfg["sae_dir"] / f"sae_{STAGE_BASE}_layer{LAYER}.pt")
        d_sae = Wd.shape[1]
        print(f"\n--- {chain} ---")
        print(f"  reweighted |Q| = {len(rw_idx)} / {d_sae}")

        for stage in (STAGE_BASE, STAGE_TARGET):
            if stage == STAGE_BASE:
                model_path = INSTRUCT_MODEL
            else:
                step = stage[len("ppo_step"):]
                model_path = str(cfg["merged_dir"] / f"step_{step}")
                if not (cfg["merged_dir"] / f"step_{step}").exists():
                    print(f"[skip] {chain} {stage}: missing merged model")
                    continue
            sae_path = cfg["sae_dir"] / f"sae_{stage}_layer{LAYER}.pt"
            if not sae_path.exists():
                print(f"[skip] {chain} {stage}: missing SAE {sae_path.name}")
                continue

            density = load_density(chain, stage)
            rw_alive_idx, rand_idx, diag = density_matched_indices(
                rw_idx, density, seed=args.seed,
            )
            print(f"  [{stage}] alive subset: "
                  f"rw {diag['n_rw_alive']}/{diag['n_rw_full']} alive "
                  f"(dropped {diag['n_rw_dead_dropped']} dead), "
                  f"non-rw alive pool={diag['n_non_rw_alive_pool']}")
            print(f"  [{stage}] density-matched control: "
                  f"|ctrl|={len(rand_idx)} "
                  f"density_match={diag['density_match_ratio']:.4f} "
                  f"(rw={diag['rw_summed_density']:.3f}, "
                  f"ctrl={diag['ctrl_summed_density']:.3f})")

            ce = run_cell(chain, stage, model_path, sae_path, prompts,
                          rw_alive_idx, rand_idx, args.device, args.batch_size)
            rows.append({
                "chain": chain,
                "stage": stage,
                "n_rw_full":   diag["n_rw_full"],
                "n_rw_alive":  diag["n_rw_alive"],
                "n_random":    len(rand_idx),
                "rw_summed_density":   f"{diag['rw_summed_density']:.4f}",
                "ctrl_summed_density": f"{diag['ctrl_summed_density']:.4f}",
                "density_match_ratio": f"{diag['density_match_ratio']:.4f}",
                "ce_unpatched":         f"{ce['unpatched']:.4f}",
                "ce_full_recon":        f"{ce['full_recon']:.4f}",
                "ce_ablate_reweighted": f"{ce['ablate_reweighted']:.4f}",
                "ce_ablate_random":     f"{ce['ablate_random']:.4f}",
                "delta_rw_vs_full":     f"{ce['ablate_reweighted'] - ce['full_recon']:.4f}",
                "delta_rand_vs_full":   f"{ce['ablate_random']     - ce['full_recon']:.4f}",
                "specificity_rw_minus_rand":
                    f"{ce['ablate_reweighted'] - ce['ablate_random']:.4f}",
            })

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["chain", "stage", "n_rw_full", "n_rw_alive", "n_random",
                  "rw_summed_density", "ctrl_summed_density",
                  "density_match_ratio",
                  "ce_unpatched", "ce_full_recon",
                  "ce_ablate_reweighted", "ce_ablate_random",
                  "delta_rw_vs_full", "delta_rand_vs_full",
                  "specificity_rw_minus_rand"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nDone. Results -> {out_csv}")

    # Print readable summary
    print(f"\n{'chain':<10s} {'stage':<14s} "
          f"{'full':>7s} {'rw':>7s} {'rand':>7s} "
          f"{'rw-full':>8s} {'rand-full':>10s} {'rw-rand':>8s}")
    print("-" * 80)
    for r in rows:
        print(f"{r['chain']:<10s} {r['stage']:<14s} "
              f"{r['ce_full_recon']:>7s} "
              f"{r['ce_ablate_reweighted']:>7s} "
              f"{r['ce_ablate_random']:>7s} "
              f"{r['delta_rw_vs_full']:>8s} "
              f"{r['delta_rand_vs_full']:>10s} "
              f"{r['specificity_rw_minus_rand']:>8s}")

    print("\nCross-stage incremental (specificity at step30 vs at instruct_base):")
    by_chain: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_chain.setdefault(r["chain"], {})[r["stage"]] = r
    for chain, by_stage in by_chain.items():
        if STAGE_BASE in by_stage and STAGE_TARGET in by_stage:
            base = float(by_stage[STAGE_BASE]["specificity_rw_minus_rand"])
            tgt  = float(by_stage[STAGE_TARGET]["specificity_rw_minus_rand"])
            print(f"  {chain:<10s} base={base:+.4f}  step30={tgt:+.4f}  "
                  f"increment={tgt - base:+.4f}")


if __name__ == "__main__":
    main()
