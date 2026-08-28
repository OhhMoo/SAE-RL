"""
08c_flex_named_features.py — Cross-reward gradient at named-feature granularity.

Devlog 2026-05-04 #3: tests whether flex's top-3 load-bearing L18 features (by δce)
are strict's named primitives at lower magnitude, or qualitatively distinct.

Applies BOTH 08b predicates (leading_digit, calc_block_opener) to:
  (1) Flex's top-N load-bearing features by δce, AFTER filtering by firing breadth
      (n_prompts_fired ≥ --min_n_fired). The unfiltered v1 of this script picked
      two near-silent features (5505 n=1, 1193 n=0) and one non-arithmetic
      broad feature (6265). Filter excludes flukes/ghosts before ranking.
  (2) Cross-chain probe features (default fid=1622) — same fid as strict's named
      features, tested in flex's chain. Same feature, same predicate → direct
      strict-vs-flex δce attenuation comparison at the SAME identity.

Reads:
  - results/l18_topk_contexts.csv          (08 output)
  - results/l18_topk_summary.csv           (08 output, for n_prompts_fired filter)
  - results/l18_attribution.csv            (07b output)
  - results/named_feature_summary.csv      (08b output, optional, for contrast print)
Writes:
  - results/flex_named_feature_tests.csv   (per-context predicate flags)
  - results/flex_named_feature_summary.csv (per-feature × predicate pass rates)

No GPU needed. Run from sae_rl/:
    python scripts/08c_flex_named_features.py
    python scripts/08c_flex_named_features.py --min_n_fired 20 --n_top 3
    python scripts/08c_flex_named_features.py --cross_chain_fids 1622,6557
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
INSTRUCT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEAD_THRESHOLD = 1e-4
CHAIN = "flexible"

PREDICATE_HORIZON = {
    "leading_digit": 1,
    "calc_block_opener": 5,
}


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


def load_n_fired(chain: str, d_sae: int) -> np.ndarray:
    """n_prompts_fired per fid from 08's topk_summary; 0 for fids not in summary."""
    out = np.zeros(d_sae, dtype=np.int64)
    path = ROOT / "results" / "l18_topk_summary.csv"
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["chain"] != chain:
                continue
            out[int(r["feature_idx"])] = int(r["n_prompts_fired"])
    return out


def loadbearing_top_n(delta, density, alive, n_fired, n=3, k=50,
                       density_cap=0.5, min_n_fired=10):
    """Load-bearing K=50 (07d-identical) → filter by n_fired ≥ min_n_fired → top-n by δce.

    The K=50 selection is preserved (same as 07d/08); the firing-breadth filter
    only changes which n features within K we surface for predicate testing.
    """
    eligible = alive & (density < density_cap) & (delta > 0)
    pool = np.where(eligible)[0]
    order = np.argsort(-delta[pool])
    K = pool[order[:k]]
    K_fired = K[n_fired[K] >= min_n_fired]
    K_fired_sorted = K_fired[np.argsort(-delta[K_fired])]
    return K_fired_sorted[:n], K   # also return full K for diagnostics


def load_topk_contexts(path: Path, chain: str, fid: int, top_rank: int):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["chain"] != chain or int(r["feature_idx"]) != fid:
                continue
            if int(r["top_rank"]) >= top_rank:
                continue
            if int(r["prompt_idx"]) < 0:
                continue
            rows.append(r)
    return rows


def tokenize_in_batches(prompts, batch_size, max_length, tokenizer):
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
    verify it matches the stored token_str. Catches batch_size / padding_side drift."""
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
                    f"  Fix: rerun with matching --batch_size and --max_length, or "
                    f"regenerate the contexts CSV with the new settings."
                )
            print(f"  alignment OK: chain={chain} (p={p}, t={t}) token_str={stored!r} matches.")
            return
    raise RuntimeError(f"no real (non-padding) rows for chain={chain} in {contexts_csv}")


def get_next_tokens(input_ids, attn, t, horizon, tokenizer):
    out = []
    for k in range(t + 1, min(len(input_ids), t + 1 + horizon)):
        if attn[k] == 0:
            break
        out.append(tokenizer.decode([input_ids[k]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_top", type=int, default=3)
    ap.add_argument("--top_rank", type=int, default=20)
    ap.add_argument("--n_prompts", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--min_n_fired", type=int, default=10,
                    help="Filter load-bearing K to features firing on ≥N prompts before top-n by δce.")
    ap.add_argument("--cross_chain_fids", default="1622",
                    help="Comma-separated fids to test in flex regardless of load-bearing rank "
                         "(direct strict-vs-flex same-fid comparison). Default: 1622.")
    ap.add_argument("--contexts_csv", default="results/l18_topk_contexts.csv")
    ap.add_argument("--strict_summary", default="results/named_feature_summary.csv")
    ap.add_argument("--out_long", default="results/flex_named_feature_tests.csv")
    ap.add_argument("--out_summary", default="results/flex_named_feature_summary.csv")
    args = ap.parse_args()

    print(f"=== Flex named-feature contrast "
          f"(top-{args.n_top} load-bearing by δce, n_fired ≥ {args.min_n_fired}) ===")
    delta, density, alive = load_attribution(CHAIN)
    n_fired = load_n_fired(CHAIN, d_sae=delta.shape[0])
    top_n, K = loadbearing_top_n(delta, density, alive, n_fired,
                                   n=args.n_top, min_n_fired=args.min_n_fired)
    n_K_fired = int((n_fired[K] >= args.min_n_fired).sum())
    n_K_zero = int((n_fired[K] == 0).sum())
    print(f"load-bearing K={len(K)}: {n_K_fired}/{len(K)} pass n_fired filter; "
          f"{n_K_zero}/{len(K)} never fire (BOS-only ghosts).")
    print("flex top features (after firing-breadth filter):")
    for fid in top_n:
        print(f"  fid={int(fid):>5}  δce={float(delta[fid]):+.4f}  "
              f"density={float(density[fid]):.4f}  n_fired={int(n_fired[fid])}/100")

    cross_chain = [int(s) for s in args.cross_chain_fids.split(",") if s.strip()]
    test_fids = [(int(f), "loadbearing") for f in top_n]
    for cfid in cross_chain:
        if cfid in [int(f) for f in top_n]:
            continue   # already in top-n by δce
        in_K = cfid in [int(i) for i in K]
        role = "cross_chain_in_K" if in_K else "cross_chain_outside_K"
        test_fids.append((cfid, role))
        print(f"cross-chain probe: fid={cfid:>5}  δce={float(delta[cfid]):+.4f}  "
              f"density={float(density[cfid]):.4f}  n_fired={int(n_fired[cfid])}/100  [{role}]")

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

    print()
    for fid, role in test_fids:
        ctx_rows = load_topk_contexts(Path(args.contexts_csv), CHAIN, fid, args.top_rank)
        if not ctx_rows:
            print(f"[skip] fid={fid}: no contexts in {args.contexts_csv}")
            continue
        contexts = [(int(r["prompt_idx"]), int(r["token_pos"])) for r in ctx_rows]
        max_act = max(float(r["activation"]) for r in ctx_rows)
        for pred_name, fn in PREDICATE_FN.items():
            horizon = PREDICATE_HORIZON[pred_name]
            flags = []
            for r, (p, t) in zip(ctx_rows, contexts):
                nxt = get_next_tokens(flat_ids[p], flat_attn[p], t, horizon, tokenizer)
                flag = fn(nxt)
                flags.append(flag)
                long_rows.append({
                    "feature_idx": fid,
                    "role": role,
                    "delta_ce": f"{float(delta[fid]):+.4f}",
                    "n_fired": int(n_fired[fid]),
                    "predicate": pred_name,
                    "top_rank": int(r["top_rank"]),
                    "prompt_idx": int(r["prompt_idx"]),
                    "token_pos": int(r["token_pos"]),
                    "token_str": r["token_str"],
                    "predicate_pass": int(flag),
                    "activation": r["activation"],
                })
            n, n_pass = len(flags), int(sum(flags))
            rate = n_pass / n if n else 0.0
            summary_rows.append({
                "chain": CHAIN,
                "feature_idx": fid,
                "role": role,
                "delta_ce": f"{float(delta[fid]):+.4f}",
                "density": f"{float(density[fid]):.4f}",
                "max_act": f"{max_act:.4f}",
                "n_fired": int(n_fired[fid]),
                "predicate": pred_name,
                "n_contexts": n,
                "n_pass": n_pass,
                "pass_rate": f"{rate:.4f}",
            })
            print(f"  fid={fid:>5} ({role:<20s}) {pred_name:<20s}: "
                  f"{n_pass}/{n} = {rate:.0%}  max_act={max_act:.2f}")

    # --- Optional: contrast against strict's named-feature pass rates from 08b ---
    strict_path = Path(args.strict_summary)
    if strict_path.exists():
        print("\n=== Strict (08b) vs Flex (this run) — pass-rate contrast ===")
        strict_rates = {}
        strict_max_acts = {}
        # 08b summary doesn't carry max_act; fall back to topk_summary if needed.
        with open(strict_path) as f:
            for r in csv.DictReader(f):
                if r["role"] == "loadbearing":
                    strict_rates[r["predicate"]] = float(r["pass_rate"])
        topk_summary = ROOT / "results" / "l18_topk_summary.csv"
        if topk_summary.exists():
            with open(topk_summary) as f:
                for r in csv.DictReader(f):
                    if r["chain"] == "strict" and int(r["feature_idx"]) in (1622, 6557):
                        strict_max_acts[int(r["feature_idx"])] = float(r["max_act"])
        for pred_name in PREDICATE_FN:
            flex_rates = [float(r["pass_rate"]) for r in summary_rows
                          if r["predicate"] == pred_name]
            flex_best = max(flex_rates) if flex_rates else 0.0
            s = strict_rates.get(pred_name, float("nan"))
            print(f"  {pred_name:<20s}  strict={s:.0%}  flex_best={flex_best:.0%}")
        if strict_max_acts:
            flex_acts = {int(r["feature_idx"]): float(r["max_act"]) for r in summary_rows}
            print(f"  max_act (strict)  : {strict_max_acts}")
            print(f"  max_act (flex top): {flex_acts}")
    else:
        print(f"\n(Skipped contrast: {strict_path} not found — run 08b first.)")

    # --- Write CSVs ---
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
