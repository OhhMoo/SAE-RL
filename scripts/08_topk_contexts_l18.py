"""
08_topk_contexts_l18.py — Name the L18 load-bearing circuit.

07d located the L18 load-bearing K=50 set per chain (causal: ~1.3-1.8 nats CE
hit on GSM8k, vs density-matched controls at ~0.05). This script asks what
those features actually fire on, so we can claim a *named* L18 circuit, not
just located.

Per chain (flexible / strict / shuffled), for each of the 50 load-bearing
features at ppo_step30:
  - encode 100 GSM8k test (q + a) prompts at L18 with the chain's SAE
  - rank tokens by feature activation
  - dump top-K activating contexts (default top_rank=20) with ±W tokens
  - flag math-content tokens with a deterministic regex
  - summarize per-chain mean math_frac across the 50

Hypothesis (devlog 2026-05-03 pre-reg): flex+strict load-bearing features fire
on math-content tokens; shuffled's K fires on generic CoT scaffolding.

Run from sae_rl/:
    python scripts/08_topk_contexts_l18.py
    python scripts/08_topk_contexts_l18.py --chains strict --top_rank 30
"""

import argparse
import csv
import re
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


# Digits, common ASCII operators, unicode × ÷, decimal/comma separators, and
# `<` `>` because GSM8k answers wrap arithmetic in `<<expr=val>>` markup —
# a feature that fires on " <<" is firing on a calculation-block opener.
MATH_RE = re.compile(r"[0-9+\-*/=×÷.,<>]")


def is_math_token(s: str) -> int:
    return int(bool(MATH_RE.search(s)))


# ---------------------------------------------------------------------------
# TopK SAE — must match scripts/05_train_sae.py (copied from 07d).
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
# Selection (copied from 07d so the K-set matches exactly).
# ---------------------------------------------------------------------------

def load_attribution(chain: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    """Returns (delta_ce, density, alive_mask, quadrant_by_idx)."""
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
    quad = {int(r["feature_idx"]): r["quadrant"] for r in rows}
    return delta, density, alive, quad


def loadbearing_indices(delta: np.ndarray, density: np.ndarray, alive: np.ndarray,
                        k: int, density_cap: float) -> np.ndarray:
    eligible = alive & (density < density_cap) & (delta > 0)
    pool = np.where(eligible)[0]
    if len(pool) < k:
        raise ValueError(f"only {len(pool)} eligible features; reduce k or raise cap")
    order = np.argsort(-delta[pool])
    return pool[order[:k]]


# ---------------------------------------------------------------------------
# Forward hook: capture per-token z for selected features only.
# Does NOT modify the residual stream (residual is returned unchanged).
# ---------------------------------------------------------------------------

class CaptureZHook:
    def __init__(self, model, layer_idx: int, sae: TopKSAE, feat_idx: np.ndarray):
        self.model, self.layer_idx, self.sae = model, layer_idx, sae
        device = sae.encoder.weight.device
        self.feat_idx_t = torch.tensor(feat_idx, dtype=torch.long, device=device)
        self.captured: list[torch.Tensor] = []  # per-batch fp16 cpu tensors of shape (B,T,K)
        self._handle = None

    def _hook(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        B, T, D = h.shape
        z = self.sae.encode(h.float().reshape(B * T, D))            # (B*T, d_sae)
        z_sel = z.index_select(1, self.feat_idx_t).reshape(B, T, -1)  # (B, T, K)
        self.captured.append(z_sel.detach().to("cpu", torch.float16))
        return out  # explicit pass-through; residual unchanged

    def __enter__(self):
        self._handle = self.model.model.layers[self.layer_idx].register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ---------------------------------------------------------------------------
# Per-chain driver
# ---------------------------------------------------------------------------

@torch.no_grad()
def sanity_check_no_residual_modification(model, tokenizer, prompts, sae, feat_idx,
                                          device, batch_size, max_length):
    """One batch: CE under CaptureZHook must equal bare-model CE within float noise."""
    enc = tokenizer(prompts[:batch_size], return_tensors="pt", padding=True,
                    truncation=True, max_length=max_length).to(device)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    ce_bare = model(**enc, labels=labels).loss.item()
    with CaptureZHook(model, LAYER, sae, feat_idx):
        ce_hooked = model(**enc, labels=labels).loss.item()
    assert abs(ce_bare - ce_hooked) < 1e-3, (
        f"CaptureZHook modifies the residual stream: bare={ce_bare:.6f} hooked={ce_hooked:.6f}"
    )
    print(f"  sanity: bare CE {ce_bare:.4f} == hooked CE {ce_hooked:.4f}")


def decode_context(input_ids: list[int], attn: list[int], t: int,
                   W: int, tokenizer) -> tuple[str, str]:
    n = len(input_ids)
    pre_ids = [input_ids[k] for k in range(max(0, t - W), t) if attn[k] == 1]
    post_ids = [input_ids[k] for k in range(t + 1, min(n, t + 1 + W)) if attn[k] == 1]
    return tokenizer.decode(pre_ids), tokenizer.decode(post_ids)


def run_chain(chain: str, args, prompts: list[str]) -> tuple[list[dict], list[dict]]:
    cfg = CHAINS[chain]
    if args.stage == STAGE_BASE:
        model_path = INSTRUCT_MODEL
    else:
        step = args.stage[len("ppo_step"):]
        merged = cfg["merged_dir"] / f"step_{step}"
        if not merged.exists():
            print(f"[skip] {chain} {args.stage}: missing merged model at {merged}")
            return [], []
        model_path = str(merged)
    sae_path = cfg["sae_dir"] / f"sae_{args.stage}_layer{LAYER}.pt"
    if not sae_path.exists():
        print(f"[skip] {chain} {args.stage}: missing SAE {sae_path.name}")
        return [], []

    print(f"\n--- {chain} ---")
    print(f"  model = {model_path}")
    print(f"  sae   = {sae_path.name}")

    delta, density, alive, quad = load_attribution(chain)
    load_idx = loadbearing_indices(delta, density, alive, k=args.k,
                                   density_cap=args.density_cap)
    rank_of = {int(fid): r for r, fid in enumerate(load_idx)}
    print(f"  load_idx: K={len(load_idx)}  density_max={float(density[load_idx].max()):.3f}  "
          f"delta_min={float(delta[load_idx].min()):+.4f}  delta_max={float(delta[load_idx].max()):+.4f}")
    if chain == "strict" and args.stage == "ppo_step30":
        if 1622 not in set(int(x) for x in load_idx):
            print("  WARN: feature 1622 not in strict load_idx — selection may have drifted from 07d.")
        else:
            print(f"  sanity: strict feature 1622 present at loadbearing_rank={rank_of[1622]}.")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to(args.device).eval()
    sae, _ = load_sae(sae_path, args.device)

    sanity_check_no_residual_modification(
        model, tokenizer, prompts, sae, load_idx, args.device,
        args.batch_size, args.max_length,
    )

    # Forward sweep — capture per-token z for the K selected features.
    per_batch_inputs: list[torch.Tensor] = []   # list of (B,T) int input_ids cpu
    per_batch_attn: list[torch.Tensor] = []     # list of (B,T) int attn cpu
    with torch.no_grad(), CaptureZHook(model, LAYER, sae, load_idx) as hook:
        for i in tqdm(range(0, len(prompts), args.batch_size),
                      desc=f"  forward ({chain})", leave=False):
            enc = tokenizer(prompts[i:i + args.batch_size], return_tensors="pt",
                            padding=True, truncation=True,
                            max_length=args.max_length).to(args.device)
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            per_batch_inputs.append(enc["input_ids"].to("cpu"))
            per_batch_attn.append(enc["attention_mask"].to("cpu"))

    captured = hook.captured  # list of (B,T,K) cpu fp16
    K = len(load_idx)

    # Build a flat index of (prompt_idx, batch_idx, within_batch_idx).
    # prompt_idx = batch_idx * args.batch_size + within_batch_idx, but the last
    # batch may be smaller — recompute via running offset.
    prompt_offsets = []
    off = 0
    for ids in per_batch_inputs:
        prompt_offsets.append(off)
        off += ids.shape[0]
    n_prompts = off

    # Build per-prompt id/attn lists for context decoding.
    prompt_ids: list[list[int]] = [None] * n_prompts
    prompt_attn: list[list[int]] = [None] * n_prompts
    for b, (ids, attn) in enumerate(zip(per_batch_inputs, per_batch_attn)):
        for j in range(ids.shape[0]):
            p = prompt_offsets[b] + j
            prompt_ids[p] = ids[j].tolist()
            prompt_attn[p] = attn[j].tolist()

    # Per feature: gather (prompt_idx, token_pos, activation) over real tokens.
    long_rows: list[dict] = []
    summary_rows: list[dict] = []
    for j in range(K):
        fid = int(load_idx[j])
        prompt_idx_arr_chunks = []
        token_pos_arr_chunks = []
        act_arr_chunks = []
        for b, (z_b, attn_b) in enumerate(zip(captured, per_batch_attn)):
            B_i, T_i = attn_b.shape
            mask = attn_b.numpy().astype(bool)        # (B_i, T_i)
            acts = z_b[..., j].numpy().astype(np.float32)  # (B_i, T_i)
            # Coordinates of real tokens in this batch (skip BOS-anomaly positions).
            bb, tt = np.nonzero(mask)
            keep = tt >= args.min_token_pos
            bb, tt = bb[keep], tt[keep]
            prompt_idx_arr_chunks.append(prompt_offsets[b] + bb)
            token_pos_arr_chunks.append(tt)
            act_arr_chunks.append(acts[bb, tt])
        p_arr = np.concatenate(prompt_idx_arr_chunks) if prompt_idx_arr_chunks else np.array([], dtype=np.int64)
        t_arr = np.concatenate(token_pos_arr_chunks) if token_pos_arr_chunks else np.array([], dtype=np.int64)
        a_arr = np.concatenate(act_arr_chunks) if act_arr_chunks else np.array([], dtype=np.float32)

        if len(a_arr) == 0:
            order = np.array([], dtype=np.int64)
        else:
            order = np.argsort(-a_arr, kind="stable")
        top = order[:args.top_rank]

        fired = a_arr > 0
        n_prompts_fired = int(len(np.unique(p_arr[fired]))) if fired.any() else 0
        max_act = float(a_arr.max()) if len(a_arr) else 0.0
        mean_act_fired = float(a_arr[fired].mean()) if fired.any() else 0.0

        math_flags: list[int] = []
        for rank, idx in enumerate(top):
            p = int(p_arr[idx])
            t = int(t_arr[idx])
            act = float(a_arr[idx])
            ids = prompt_ids[p]
            attn = prompt_attn[p]
            tok_str = tokenizer.decode([ids[t]])
            ctx_pre, ctx_post = decode_context(ids, attn, t, args.context_window, tokenizer)
            mtok = is_math_token(tok_str)
            math_flags.append(mtok)
            long_rows.append({
                "chain": chain,
                "feature_idx": fid,
                "loadbearing_rank": rank_of[fid],
                "delta_ce": f"{float(delta[fid]):+.4f}",
                "density":  f"{float(density[fid]):.4f}",
                "quadrant": quad.get(fid, ""),
                "top_rank": rank,
                "prompt_idx": p,
                "token_pos": t,
                "token_str": tok_str,
                "context_pre": ctx_pre,
                "context_post": ctx_post,
                "activation": f"{act:.4f}",
                "is_math_token": mtok,
            })
        # Pad to top_rank if fewer real tokens than top_rank existed (shouldn't happen with 100 prompts).
        while len(math_flags) < args.top_rank:
            long_rows.append({
                "chain": chain, "feature_idx": fid,
                "loadbearing_rank": rank_of[fid],
                "delta_ce": f"{float(delta[fid]):+.4f}",
                "density":  f"{float(density[fid]):.4f}",
                "quadrant": quad.get(fid, ""),
                "top_rank": len(math_flags),
                "prompt_idx": -1, "token_pos": -1,
                "token_str": "", "context_pre": "", "context_post": "",
                "activation": "0.0000", "is_math_token": 0,
            })
            math_flags.append(0)

        math_frac = float(np.mean(math_flags)) if math_flags else 0.0
        summary_rows.append({
            "chain": chain,
            "feature_idx": fid,
            "loadbearing_rank": rank_of[fid],
            "delta_ce": f"{float(delta[fid]):+.4f}",
            "density":  f"{float(density[fid]):.4f}",
            "quadrant": quad.get(fid, ""),
            "max_act": f"{max_act:.4f}",
            "mean_act_fired": f"{mean_act_fired:.4f}",
            "n_prompts_fired": n_prompts_fired,
            "math_frac_topk": f"{math_frac:.4f}",
        })

    del model, sae, tokenizer
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return long_rows, summary_rows


# ---------------------------------------------------------------------------
# Console highlights
# ---------------------------------------------------------------------------

def print_chain_highlights(chain: str, long_rows: list[dict], summary_rows: list[dict]):
    if not summary_rows:
        return
    math_fracs = np.array([float(r["math_frac_topk"]) for r in summary_rows])
    nf_zero = int(sum(1 for r in summary_rows if int(r["n_prompts_fired"]) == 0))
    print(f"\n[{chain}] mean math_frac_topk = {math_fracs.mean():.3f}  "
          f"(min={math_fracs.min():.3f}, max={math_fracs.max():.3f})  "
          f"n_features_never_fired={nf_zero}")
    # Top-3 by delta_ce → top-1 context.
    by_delta = sorted(summary_rows, key=lambda r: -float(r["delta_ce"]))[:3]
    for s in by_delta:
        fid = s["feature_idx"]
        # Find rank=0 row for this feature.
        for r in long_rows:
            if r["chain"] == chain and r["feature_idx"] == fid and r["top_rank"] == 0:
                print(f"  fid={fid:>5}  Δce={s['delta_ce']}  d={s['density']}  q={s['quadrant']:<10}  "
                      f"top1: '{r['token_str']}'  pre='{r['context_pre'][-40:]}'  post='{r['context_post'][:40]}'")
                break
    if chain == "strict":
        rows_1622 = [r for r in long_rows if r["feature_idx"] == 1622 and r["top_rank"] < 3]
        if rows_1622:
            print(f"  -- strict feature 1622 (07d-flagged PPO-acquired) top-3 contexts:")
            for r in rows_1622:
                print(f"     rank={r['top_rank']}  tok='{r['token_str']}'  "
                      f"pre='{r['context_pre'][-40:]}' post='{r['context_post'][:40]}'  "
                      f"math={r['is_math_token']}  act={r['activation']}")


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
    ap.add_argument("--top_rank", type=int, default=20)
    ap.add_argument("--context_window", type=int, default=8)
    ap.add_argument("--min_token_pos", type=int, default=1,
                    help="Exclude positions < min_token_pos from top-K extraction. "
                         "Default 1 skips the BOS-anomaly first-token firings that dominate "
                         "raw activation rankings across most SAE features.")
    ap.add_argument("--stage", default="ppo_step30")
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_long", default="results/l18_topk_contexts.csv")
    ap.add_argument("--out_summary", default="results/l18_topk_summary.csv")
    ap.add_argument("--out_chain_summary", default="results/l18_topk_chain_summary.csv")
    args = ap.parse_args()

    print("=== L18 top-K activating contexts (load-bearing K=50) ===")
    print(f"chains       : {args.chains}")
    print(f"stage        : {args.stage}")
    print(f"K            : {args.k}     density_cap : {args.density_cap}")
    print(f"top_rank     : {args.top_rank}     context_window : {args.context_window}")
    print(f"prompts      : {args.n_prompts} GSM8k test (q + a)")
    print(f"out (long)   : {args.out_long}")
    print(f"out (summary): {args.out_summary}")
    print(f"out (chain)  : {args.out_chain_summary}\n")

    print("Loading GSM8k test prompts...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    prompts = [f"{ex['question']} {ex['answer']}" for ex in ds][:args.n_prompts]
    print(f"  loaded {len(prompts)} prompts")

    all_long: list[dict] = []
    all_summary: list[dict] = []
    chain_summary: list[dict] = []
    for chain in args.chains:
        if chain not in CHAINS:
            print(f"[skip] unknown chain {chain}")
            continue
        long_rows, summary_rows = run_chain(chain, args, prompts)
        if not summary_rows:
            continue
        print_chain_highlights(chain, long_rows, summary_rows)
        all_long.extend(long_rows)
        all_summary.extend(summary_rows)
        math_fracs = np.array([float(r["math_frac_topk"]) for r in summary_rows])
        max_acts = np.array([float(r["max_act"]) for r in summary_rows])
        n_fired = np.array([int(r["n_prompts_fired"]) for r in summary_rows])
        fired_mask = n_fired > 0
        mf_fired = float(math_fracs[fired_mask].mean()) if fired_mask.any() else 0.0
        chain_summary.append({
            "chain": chain,
            "n_features_fired": int(fired_mask.sum()),
            "n_features_never_fired": int((~fired_mask).sum()),
            "mean_math_frac_topk_all": f"{math_fracs.mean():.4f}",
            "mean_math_frac_topk_fired_only": f"{mf_fired:.4f}",
            "mean_max_act": f"{max_acts.mean():.4f}",
            "mean_n_prompts_fired": f"{n_fired.mean():.2f}",
        })

    # Write CSVs.
    out_long = Path(args.out_long); out_long.parent.mkdir(parents=True, exist_ok=True)
    out_summary = Path(args.out_summary); out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_chain = Path(args.out_chain_summary); out_chain.parent.mkdir(parents=True, exist_ok=True)

    if all_long:
        with open(out_long, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_long[0].keys()))
            w.writeheader(); w.writerows(all_long)
        print(f"\nWrote {len(all_long)} rows -> {out_long}")
    if all_summary:
        with open(out_summary, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_summary[0].keys()))
            w.writeheader(); w.writerows(all_summary)
        print(f"Wrote {len(all_summary)} rows -> {out_summary}")
    if chain_summary:
        with open(out_chain, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(chain_summary[0].keys()))
            w.writeheader(); w.writerows(chain_summary)
        print(f"Wrote {len(chain_summary)} rows -> {out_chain}")

    # Final cross-chain summary.
    print("\n=== chain-level summary ===")
    print(f"{'chain':<10s} {'fired':>6s} {'mf_all':>8s} {'mf_fired':>9s} "
          f"{'max_act':>9s} {'mean_fired':>11s}")
    for r in chain_summary:
        print(f"{r['chain']:<10s} {r['n_features_fired']:>6d} "
              f"{r['mean_math_frac_topk_all']:>8s} "
              f"{r['mean_math_frac_topk_fired_only']:>9s} "
              f"{r['mean_max_act']:>9s} {r['mean_n_prompts_fired']:>11s}")


if __name__ == "__main__":
    main()
