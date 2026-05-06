"""
08b_named_feature_tests.py — Quantitative validation of strict's named L18 features.

08 surfaced two named features in strict's load-bearing K=50:
  - fid=1622: hypothesized "leading digit of multi-digit numeral".
  - fid=6557: hypothesized "calc-block opener" (predicts <<expr=val>> emission).

This script formally tests both hypotheses with pre-registered lexical predicates
on top-20 activating contexts, plus a counterfactual control of n=5 non-load-bearing
features per named target at matched density.

Pre-registered thresholds (devlog 2026-05-04):
  - fid=1622 leading_digit: top-20 next-token first non-whitespace char is digit, ≥80%.
  - fid=6557 calc_block_opener: top-20 next ≤5 tokens contain BOTH `=` and `>>`, ≥80%.
  - controls: same predicates on matched-density non-load-bearing strict features,
              ≤30% pass rate (selective).

Reads:
  - results/l18_topk_contexts.csv   (08 output, top-20 contexts of load-bearing K)
  - results/l18_attribution.csv     (07b output, for control selection + density)
Writes:
  - results/named_feature_tests.csv     (per-context predicate flags)
  - results/named_feature_summary.csv   (per-feature × predicate pass rates + verdict)

Run from sae_rl/:
    python scripts/08b_named_feature_tests.py
    python scripts/08b_named_feature_tests.py --skip_controls   # CSV-only, no GPU
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
DEAD_THRESHOLD = 1e-4

CHAIN = "strict"
SAE_DIR = ROOT / "checkpoints" / "saes_strict"
MERGED_DIR = ROOT / "checkpoints" / "ppo_merged_strict"

NAMED_FEATURES = {
    1622: "leading_digit",
    6557: "calc_block_opener",
}
PREDICATE_HORIZON = {
    "leading_digit": 1,
    "calc_block_opener": 5,
}
PASS_THRESHOLD = 0.80
CONTROL_THRESHOLD = 0.30


# --- TopKSAE (must match 05_train_sae.py / 08) -----------------------------

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


def load_sae(path: Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sae = TopKSAE(cfg["d_model"], cfg["d_sae"], cfg["k"])
    sae.load_state_dict(ckpt["state_dict"], strict=False)
    return sae.to(device).eval()


# --- Predicates -------------------------------------------------------------

def leading_digit(next_tokens: list[str]) -> bool:
    if not next_tokens:
        return False
    s = next_tokens[0].lstrip()
    return bool(s) and s[0].isdigit()


def calc_block_opener(next_tokens: list[str]) -> bool:
    joined = "".join(next_tokens)
    return ("=" in joined) and (">>" in joined)


PREDICATE_FN = {
    "leading_digit": leading_digit,
    "calc_block_opener": calc_block_opener,
}


# --- IO ---------------------------------------------------------------------

def load_topk_contexts(path: Path, chain: str, fid: int, top_rank: int) -> list[dict]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["chain"] != chain or int(r["feature_idx"]) != fid:
                continue
            if int(r["top_rank"]) >= top_rank:
                continue
            if int(r["prompt_idx"]) < 0:
                continue  # padding row from 08 (feature fired fewer than top_rank times)
            rows.append(r)
    return rows


def load_attribution(chain: str):
    rows = []
    with open(ROOT / "results" / "l18_attribution.csv") as f:
        for r in csv.DictReader(f):
            if r["chain"] == chain:
                rows.append(r)
    rows.sort(key=lambda r: int(r["feature_idx"]))
    delta = np.array([float(r["delta_ce"]) for r in rows])
    density = np.array([float(r["density"]) for r in rows])
    alive = density > DEAD_THRESHOLD
    return delta, density, alive


def loadbearing_indices(delta, density, alive, k=50, density_cap=0.5) -> np.ndarray:
    eligible = alive & (density < density_cap) & (delta > 0)
    pool = np.where(eligible)[0]
    order = np.argsort(-delta[pool])
    return pool[order[:k]]


def matched_density_controls(delta, density, alive, target_fid: int, n: int,
                              exclude: set, rng: np.random.Generator) -> np.ndarray:
    target_d = float(density[target_fid])
    for band in (0.20, 0.50):
        lo, hi = (1 - band) * target_d, (1 + band) * target_d
        eligible = alive & (density >= lo) & (density <= hi)
        pool = np.array([i for i in np.where(eligible)[0] if int(i) not in exclude])
        if len(pool) >= n:
            return rng.choice(pool, size=n, replace=False)
    raise RuntimeError(f"insufficient matched-density pool for fid={target_fid}")


# --- Forward pass to mine top-K firings of control features -----------------

class CaptureZHook:
    """Captures per-token z over selected SAE features without modifying residual."""
    def __init__(self, model, layer_idx: int, sae: TopKSAE, feat_idx: np.ndarray):
        self.model, self.layer_idx, self.sae = model, layer_idx, sae
        device = sae.encoder.weight.device
        self.feat_idx_t = torch.tensor(feat_idx, dtype=torch.long, device=device)
        self.captured: list[torch.Tensor] = []
        self._handle = None

    def _hook(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        B, T, D = h.shape
        z = self.sae.encode(h.float().reshape(B * T, D))
        z_sel = z.index_select(1, self.feat_idx_t).reshape(B, T, -1)
        self.captured.append(z_sel.detach().to("cpu", torch.float16))
        return out

    def __enter__(self):
        self._handle = self.model.model.layers[self.layer_idx].register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@torch.no_grad()
def mine_topk_for_features(feat_idx: np.ndarray, top_rank: int, args,
                            tokenizer, model, sae, prompts: list[str]
                            ) -> dict[int, list[tuple[int, int]]]:
    """Return {fid: [(prompt_idx, token_pos), ...] sorted by activation desc, len ≤ top_rank}."""
    per_attn = []
    p_off = []
    off = 0
    with CaptureZHook(model, LAYER, sae, feat_idx) as hook:
        for i in tqdm(range(0, len(prompts), args.batch_size),
                       desc="control forward", leave=False):
            enc = tokenizer(prompts[i:i + args.batch_size], return_tensors="pt",
                            padding=True, truncation=True,
                            max_length=args.max_length).to(args.device)
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            per_attn.append(enc["attention_mask"].to("cpu"))
            p_off.append(off)
            off += enc["input_ids"].shape[0]
    captured = hook.captured

    out: dict[int, list[tuple[int, int]]] = {}
    for j, fid in enumerate(feat_idx):
        p_chunks, t_chunks, a_chunks = [], [], []
        for b, (z_b, attn_b) in enumerate(zip(captured, per_attn)):
            mask = attn_b.numpy().astype(bool)
            acts = z_b[..., j].numpy().astype(np.float32)
            bb, tt = np.nonzero(mask)
            keep = tt >= args.min_token_pos
            bb, tt = bb[keep], tt[keep]
            p_chunks.append(p_off[b] + bb)
            t_chunks.append(tt)
            a_chunks.append(acts[bb, tt])
        p_arr = np.concatenate(p_chunks) if p_chunks else np.array([], dtype=np.int64)
        t_arr = np.concatenate(t_chunks) if t_chunks else np.array([], dtype=np.int64)
        a_arr = np.concatenate(a_chunks) if a_chunks else np.array([], dtype=np.float32)
        if len(a_arr) == 0:
            out[int(fid)] = []
            continue
        order = np.argsort(-a_arr, kind="stable")[:top_rank]
        out[int(fid)] = [(int(p_arr[i]), int(t_arr[i])) for i in order]
    return out


# --- Test loop --------------------------------------------------------------

def get_next_tokens(input_ids: list[int], attn: list[int], t: int,
                     horizon: int, tokenizer) -> list[str]:
    """Decoded strings of next `horizon` real (non-pad) tokens after position t."""
    out = []
    for k in range(t + 1, min(len(input_ids), t + 1 + horizon)):
        if attn[k] == 0:
            break
        out.append(tokenizer.decode([input_ids[k]]))
    return out


def evaluate(contexts: list[tuple[int, int]], flat_ids, flat_attn,
              predicate_name: str, tokenizer) -> list[bool]:
    horizon = PREDICATE_HORIZON[predicate_name]
    fn = PREDICATE_FN[predicate_name]
    return [fn(get_next_tokens(flat_ids[p], flat_attn[p], t, horizon, tokenizer))
            for p, t in contexts]


def tokenize_in_batches(prompts, batch_size, max_length, tokenizer):
    """Replays the per-batch padding from script 08 so token_pos values align."""
    flat_ids, flat_attn = [], []
    for i in range(0, len(prompts), batch_size):
        enc = tokenizer(prompts[i:i + batch_size], return_tensors="pt",
                        padding=True, truncation=True, max_length=max_length)
        for j in range(enc["input_ids"].shape[0]):
            flat_ids.append(enc["input_ids"][j].tolist())
            flat_attn.append(enc["attention_mask"][j].tolist())
    return flat_ids, flat_attn


def assert_padding_alignment(flat_ids, contexts_csv: Path, chain: str, tokenizer):
    """Decode token_str from (prompt_idx, token_pos) for the first real CSV row and
    verify it matches the stored token_str. Catches batch_size / padding_side drift
    between this script and the run of 08 that produced the contexts CSV."""
    with open(contexts_csv) as f:
        for r in csv.DictReader(f):
            if r["chain"] != chain:
                continue
            p, t = int(r["prompt_idx"]), int(r["token_pos"])
            if p < 0 or p >= len(flat_ids) or t >= len(flat_ids[p]):
                continue
            stored = r["token_str"]
            recovered = tokenizer.decode([flat_ids[p][t]])
            if stored != recovered:
                raise AssertionError(
                    f"\n  Padding/tokenization drift detected:\n"
                    f"    CSV {contexts_csv.name} row: chain={chain} fid={r['feature_idx']} "
                    f"top_rank={r['top_rank']} prompt_idx={p} token_pos={t}\n"
                    f"    stored token_str   = {stored!r}\n"
                    f"    recovered decode() = {recovered!r}\n"
                    f"  Likely cause: --batch_size, --max_length, or tokenizer padding_side "
                    f"differs from the run of 08 that produced this CSV.\n"
                    f"  Fix: rerun with matching --batch_size (default 4) and --max_length "
                    f"(default 384), or regenerate the contexts CSV with the new settings."
                )
            print(f"  alignment OK: chain={chain} (p={p}, t={t}) token_str={stored!r} matches.")
            return
    raise RuntimeError(f"no real (non-padding) rows for chain={chain} in {contexts_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_rank", type=int, default=20)
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--min_token_pos", type=int, default=1)
    ap.add_argument("--n_controls", type=int, default=5)
    ap.add_argument("--stage", default="ppo_step30")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_controls", action="store_true",
                    help="CSV-only mode: skip the GPU forward pass for controls.")
    ap.add_argument("--contexts_csv", default="results/l18_topk_contexts.csv")
    ap.add_argument("--out_long", default="results/named_feature_tests.csv")
    ap.add_argument("--out_summary", default="results/named_feature_summary.csv")
    args = ap.parse_args()

    print("=== Named feature tests (strict L18 load-bearing) ===")
    print(f"named features    : {NAMED_FEATURES}")
    print(f"top_rank          : {args.top_rank}")
    print(f"pass threshold    : ≥{PASS_THRESHOLD:.0%} (loadbearing → confirm)")
    print(f"control threshold : ≤{CONTROL_THRESHOLD:.0%} (control → selective)")
    print(f"controls          : {args.n_controls} matched-density per named feature")
    print(f"skip_controls     : {args.skip_controls}\n")

    rng = np.random.default_rng(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    ds = load_dataset("openai/gsm8k", "main", split="test")
    prompts = [f"{ex['question']} {ex['answer']}" for ex in ds][:args.n_prompts]
    flat_ids, flat_attn = tokenize_in_batches(prompts, args.batch_size,
                                                args.max_length, tokenizer)
    assert_padding_alignment(flat_ids, Path(args.contexts_csv), CHAIN, tokenizer)

    long_rows: list[dict] = []
    summary_rows: list[dict] = []

    # --- 1. Load-bearing named features (CSV-driven) ---
    for fid, pred_name in NAMED_FEATURES.items():
        ctx_rows = load_topk_contexts(Path(args.contexts_csv), CHAIN, fid, args.top_rank)
        if not ctx_rows:
            print(f"[skip] fid={fid}: no contexts in {args.contexts_csv}")
            continue
        contexts = [(int(r["prompt_idx"]), int(r["token_pos"])) for r in ctx_rows]
        flags = evaluate(contexts, flat_ids, flat_attn, pred_name, tokenizer)
        n, n_pass = len(flags), int(sum(flags))
        rate = n_pass / n if n else 0.0
        verdict = "PASS" if rate >= PASS_THRESHOLD else "FAIL"
        for r, flag in zip(ctx_rows, flags):
            long_rows.append({
                "feature_idx": fid,
                "feature_name": pred_name,
                "role": "loadbearing",
                "predicate": pred_name,
                "top_rank": int(r["top_rank"]),
                "prompt_idx": int(r["prompt_idx"]),
                "token_pos": int(r["token_pos"]),
                "token_str": r["token_str"],
                "predicate_pass": int(flag),
                "activation": r["activation"],
            })
        summary_rows.append({
            "feature_idx": fid,
            "feature_name": pred_name,
            "role": "loadbearing",
            "predicate": pred_name,
            "n_contexts": n,
            "n_pass": n_pass,
            "pass_rate": f"{rate:.4f}",
            "threshold": f"{PASS_THRESHOLD:.2f}",
            "verdict": verdict,
        })
        print(f"  fid={fid:>5} ({pred_name}): {n_pass}/{n} = {rate:.0%}  [{verdict}]")

    # --- 2. Matched-density non-load-bearing controls ---
    if not args.skip_controls:
        delta, density, alive = load_attribution(CHAIN)
        load_idx = loadbearing_indices(delta, density, alive)
        load_set = set(int(i) for i in load_idx)

        controls_per_named = {}
        all_controls: set[int] = set()
        for fid in NAMED_FEATURES:
            ctrls = matched_density_controls(delta, density, alive, fid,
                                              args.n_controls, load_set, rng)
            controls_per_named[fid] = [int(c) for c in ctrls]
            all_controls.update(int(c) for c in ctrls)
        controls_arr = np.array(sorted(all_controls))
        print(f"\nControls (n={len(controls_arr)} unique, ±20% density-matched): "
              f"{controls_arr.tolist()}")
        for fid, ctrls in controls_per_named.items():
            print(f"  for fid={fid} (d={float(density[fid]):.4f}): {ctrls}")

        sae_path = SAE_DIR / f"sae_{args.stage}_layer{LAYER}.pt"
        merged = MERGED_DIR / f"step_{args.stage[len('ppo_step'):]}"
        model_path = str(merged) if merged.exists() else INSTRUCT_MODEL
        print(f"  loading model {model_path}\n  loading sae   {sae_path.name}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to(args.device).eval()
        sae = load_sae(sae_path, args.device)

        topk_per_control = mine_topk_for_features(controls_arr, args.top_rank, args,
                                                    tokenizer, model, sae, prompts)
        del model, sae
        if args.device == "cuda":
            torch.cuda.empty_cache()

        for fid, pred_name in NAMED_FEATURES.items():
            for c_fid in controls_per_named[fid]:
                contexts = topk_per_control.get(c_fid, [])
                if not contexts:
                    summary_rows.append({
                        "feature_idx": c_fid,
                        "feature_name": f"control_for_{fid}",
                        "role": "control",
                        "predicate": pred_name,
                        "n_contexts": 0,
                        "n_pass": 0,
                        "pass_rate": "0.0000",
                        "threshold": f"{CONTROL_THRESHOLD:.2f}",
                        "verdict": "NO_FIRINGS",
                    })
                    continue
                flags = evaluate(contexts, flat_ids, flat_attn, pred_name, tokenizer)
                n, n_pass = len(flags), int(sum(flags))
                rate = n_pass / n if n else 0.0
                verdict = "SELECTIVE" if rate <= CONTROL_THRESHOLD else "LEAKY"
                summary_rows.append({
                    "feature_idx": c_fid,
                    "feature_name": f"control_for_{fid}",
                    "role": "control",
                    "predicate": pred_name,
                    "n_contexts": n,
                    "n_pass": n_pass,
                    "pass_rate": f"{rate:.4f}",
                    "threshold": f"{CONTROL_THRESHOLD:.2f}",
                    "verdict": verdict,
                })
                for k, ((p, t), flag) in enumerate(zip(contexts, flags)):
                    long_rows.append({
                        "feature_idx": c_fid,
                        "feature_name": f"control_for_{fid}",
                        "role": "control",
                        "predicate": pred_name,
                        "top_rank": k,
                        "prompt_idx": p,
                        "token_pos": t,
                        "token_str": tokenizer.decode([flat_ids[p][t]]),
                        "predicate_pass": int(flag),
                        "activation": "",
                    })

        print("\n=== control aggregate ===")
        for pred_name in set(NAMED_FEATURES.values()):
            ctrl = [float(r["pass_rate"]) for r in summary_rows
                    if r["role"] == "control" and r["predicate"] == pred_name
                    and r["verdict"] != "NO_FIRINGS"]
            if not ctrl:
                continue
            mean, worst = float(np.mean(ctrl)), float(np.max(ctrl))
            v = "SELECTIVE" if worst <= CONTROL_THRESHOLD else "LEAKY"
            print(f"  {pred_name:<20s}: mean={mean:.0%}  worst={worst:.0%}  [{v}]")

    # --- 3. Write CSVs ---
    out_long = Path(args.out_long); out_long.parent.mkdir(parents=True, exist_ok=True)
    out_summary = Path(args.out_summary); out_summary.parent.mkdir(parents=True, exist_ok=True)
    if long_rows:
        with open(out_long, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
            w.writeheader(); w.writerows(long_rows)
        print(f"\nWrote {len(long_rows)} rows -> {out_long}")
    if summary_rows:
        with open(out_summary, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader(); w.writerows(summary_rows)
        print(f"Wrote {len(summary_rows)} rows -> {out_summary}")


if __name__ == "__main__":
    main()
