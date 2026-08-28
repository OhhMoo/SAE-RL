"""
07c_attribution_summary.py — Quadrant summary of L18 per-feature attribution.

Reads `results/l18_attribution.csv` (produced by 07b_attribution_l18.py) and
asks the question that motivated the sweep: do the load-bearing features
at L18 step30 cluster in a different (dec_cos, enc_cos) quadrant than the
"reweighted" reorganization zone identified by 06c?

Per chain, prints:
  - n_alive per quadrant
  - sum(delta_ce) per quadrant (raw load-bearing weight)
  - mean delta_ce per alive feature  (per-feature average impact)
  - sum(delta_ce) / sum(density)     (per-unit-firing-mass impact;
                                       directly comparable to the
                                       per-unit ratios in l18_ablation.csv)
  - top-K features per quadrant
  - share of total load-bearing weight (sum of positive delta_ce across
    alive features) carried by each quadrant

Run from sae_rl/:
    python scripts/07c_attribution_summary.py
    python scripts/07c_attribution_summary.py --top_k 5
    python scripts/07c_attribution_summary.py --input results/l18_attribution.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
QUADRANTS = ["stable", "reweighted", "redirected", "rotated"]
DEAD_THRESHOLD = 1e-4


def load_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def per_chain_summary(rows: list[dict], top_k: int) -> None:
    by_chain: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_chain[r["chain"]].append(r)

    for chain, chain_rows in by_chain.items():
        delta = np.array([float(r["delta_ce"]) for r in chain_rows])
        density = np.array([float(r["density"]) for r in chain_rows])
        quad = np.array([r["quadrant"] for r in chain_rows])
        alive = density > DEAD_THRESHOLD

        positive_total = float(delta[alive & (delta > 0)].sum())

        print(f"\n=== {chain} / ppo_step30 ===")
        print(f"  d_sae = {len(chain_rows)}    n_alive = {int(alive.sum())}    "
              f"sum_positive_delta_ce(alive) = {positive_total:+.4f}")
        header = (f"  {'quadrant':<10s} {'n_alive':>8s} "
                  f"{'sum_d':>9s} {'mean_d':>10s} "
                  f"{'sum_d/sum_dens':>15s} {'pos_share':>10s}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for q in QUADRANTS:
            m = (quad == q) & alive
            if not m.any():
                continue
            sum_d = float(delta[m].sum())
            mean_d = float(delta[m].mean())
            sum_dens = float(density[m].sum())
            ratio = sum_d / sum_dens if sum_dens > 0 else float("nan")
            pos_q = float(delta[m & (delta > 0)].sum())
            share = pos_q / positive_total if positive_total > 0 else float("nan")
            print(f"  {q:<10s} {int(m.sum()):>8d} "
                  f"{sum_d:>+9.4f} {mean_d:>+10.5f} "
                  f"{ratio:>+15.5f} {share:>9.1%}")

        print(f"  top-{top_k} features overall (by delta_ce):")
        order = np.argsort(-delta)
        for i in order[:top_k]:
            r = chain_rows[i]
            print(f"    idx={int(r['feature_idx']):5d}  "
                  f"delta={float(r['delta_ce']):+.4f}  "
                  f"dens={float(r['density']):.4f}  "
                  f"quad={r['quadrant']}")


def cross_chain_table(rows: list[dict]) -> None:
    """Quadrant share of positive delta_ce per chain — paper-table candidate."""
    by_chain: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_chain[r["chain"]].append(r)

    print("\n=== Cross-chain: share of positive load-bearing weight by quadrant ===")
    print(f"  {'chain':<10s} " + " ".join(f"{q:>11s}" for q in QUADRANTS))
    for chain, chain_rows in by_chain.items():
        delta = np.array([float(r["delta_ce"]) for r in chain_rows])
        density = np.array([float(r["density"]) for r in chain_rows])
        quad = np.array([r["quadrant"] for r in chain_rows])
        alive = density > DEAD_THRESHOLD
        positive_total = float(delta[alive & (delta > 0)].sum())
        if positive_total <= 0:
            print(f"  {chain:<10s} (no positive delta_ce — sweep noisy?)")
            continue
        shares = []
        for q in QUADRANTS:
            m = (quad == q) & alive & (delta > 0)
            shares.append(float(delta[m].sum()) / positive_total)
        print(f"  {chain:<10s} " + " ".join(f"{s:>10.1%} " for s in shares))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/l18_attribution.csv")
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    path = (ROOT / args.input) if not Path(args.input).is_absolute() else Path(args.input)
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run 07b_attribution_l18.py first.")

    rows = load_rows(path)
    print(f"Loaded {len(rows)} rows from {path}")
    per_chain_summary(rows, args.top_k)
    cross_chain_table(rows)


if __name__ == "__main__":
    main()
