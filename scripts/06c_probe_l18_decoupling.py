#!/usr/bin/env python3
"""L18 encoder/decoder decoupling probe.

At L18 the cold-start transition (instruct_base -> ppo_step10) shows a
striking property: per-feature decoder cosine and encoder cosine are
uncorrelated (Pearson r ~ -0.14 flex, -0.10 strict). At every other layer
they are positively correlated. This script asks: do L18 features split
into distinct subpopulations differentiated by which of {encoder direction,
decoder direction} changed? And does the "low decoder, high encoder"
subpopulation (encoder selectivity preserved, decoder direction shifted)
align with the "born without dying" features in the lifecycle plot?

Each feature is placed into one of four quadrants by median split on
(dec_cos, enc_cos):
  stable      = high dec, high enc  (no change to either)
  rotated     = low  dec, low  enc  (both directions changed)
  reweighted  = high dec, low  enc  (decoder same, encoder shifted)
  redirected  = low  dec, high enc  (encoder same, decoder shifted)

Cross-tabulated with lifecycle status (alive_base x alive_step10 -> stayed_
alive / stayed_dead / born / died).

Outputs:
  results/l18_decoupling_probe.csv   (one row per chain x quadrant)
  figures/fig_l18_decoupling_probe.png

Run from sae_rl/:
  python scripts/probe_l18_decoupling.py
"""
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
LAYER = 18
DEAD_THRESHOLD = 1e-4

PALETTE = {
    "blush":    "#FAE8EB",
    "rose":     "#F6CACA",
    "dusty":    "#E4C2C6",
    "lilac":    "#CD9FCC",
    "twilight": "#0A014F",
}

CHAINS = {
    "flexible": ROOT / "checkpoints" / "saes",
    "strict":   ROOT / "checkpoints" / "saes_strict",
    "shuffled": ROOT / "checkpoints" / "saes_shuffled",
}
STAGE_A = "instruct_base"
STAGE_B = "ppo_step10"
CACHE_DIR = ROOT / "results" / "cache"
QUADS = ["stable", "reweighted", "redirected", "rotated"]


def load_sae_weights(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    return sd["decoder.weight"].float(), sd["encoder.weight"].float()


def per_feature_decoder_cosine(W_a, W_b):
    a = nn.functional.normalize(W_a, dim=0)
    b = nn.functional.normalize(W_b, dim=0)
    return (a * b).sum(dim=0)


def per_feature_encoder_cosine(W_a, W_b):
    a = nn.functional.normalize(W_a, dim=1)
    b = nn.functional.normalize(W_b, dim=1)
    return (a * b).sum(dim=1)


def load_density(chain, layer, stage):
    p = CACHE_DIR / f"density_{chain}_L{layer}_{stage}.pt"
    return torch.load(p, map_location="cpu", weights_only=False)["density"].numpy()


def assign_quadrants(dcos, ecos):
    dec_med = float(np.median(dcos))
    enc_med = float(np.median(ecos))
    quad = np.empty(len(dcos), dtype=object)
    quad[(dcos >= dec_med) & (ecos >= enc_med)] = "stable"
    quad[(dcos < dec_med) & (ecos < enc_med)] = "rotated"
    quad[(dcos >= dec_med) & (ecos < enc_med)] = "reweighted"
    quad[(dcos < dec_med) & (ecos >= enc_med)] = "redirected"
    return quad, dec_med, enc_med


def lifecycle_status(d_a, d_b):
    alive_a = d_a > DEAD_THRESHOLD
    alive_b = d_b > DEAD_THRESHOLD
    status = np.empty(len(d_a), dtype=object)
    status[alive_a & alive_b]   = "stayed_alive"
    status[~alive_a & ~alive_b] = "stayed_dead"
    status[~alive_a & alive_b]  = "born"
    status[alive_a & ~alive_b]  = "died"
    return status


# ---------------------------------------------------------------------------

rows_csv = []
fig, axes = plt.subplots(2, len(CHAINS), figsize=(5.5 * len(CHAINS), 9), squeeze=False)

for col, (chain, sae_dir) in enumerate(CHAINS.items()):
    dec_a, enc_a = load_sae_weights(sae_dir / f"sae_{STAGE_A}_layer{LAYER}.pt")
    dec_b, enc_b = load_sae_weights(sae_dir / f"sae_{STAGE_B}_layer{LAYER}.pt")
    dcos = per_feature_decoder_cosine(dec_a, dec_b).numpy()
    ecos = per_feature_encoder_cosine(enc_a, enc_b).numpy()
    d_a = load_density(chain, LAYER, STAGE_A)
    d_b = load_density(chain, LAYER, STAGE_B)
    status = lifecycle_status(d_a, d_b)
    quad, dec_med, enc_med = assign_quadrants(dcos, ecos)
    corr = float(np.corrcoef(dcos, ecos)[0, 1])

    print(f"\n=== {chain} L18  {STAGE_A} -> {STAGE_B} ===")
    print(f"  per-feature corr(dec_cos, enc_cos) = {corr:+.3f}")
    print(f"  median dec_cos = {dec_med:.4f}   median enc_cos = {enc_med:.4f}")
    print(f"\n  {'quadrant':<12s}{'n':>6s}{'dec_cos':>10s}{'enc_cos':>10s}"
          f"{'born':>7s}{'died':>7s}{'alive':>8s}{'dead':>7s}"
          f"{'born_frac':>11s}")
    for q in QUADS:
        m = quad == q
        n = int(m.sum())
        if n == 0:
            continue
        n_born  = int((status[m] == "born").sum())
        n_died  = int((status[m] == "died").sum())
        n_alive = int((status[m] == "stayed_alive").sum())
        n_dead  = int((status[m] == "stayed_dead").sum())
        born_frac = n_born / n
        print(f"  {q:<12s}{n:>6d}{dcos[m].mean():>10.4f}{ecos[m].mean():>10.4f}"
              f"{n_born:>7d}{n_died:>7d}{n_alive:>8d}{n_dead:>7d}"
              f"{born_frac:>11.3f}")
        rows_csv.append({
            "chain": chain, "layer": LAYER, "quadrant": q, "n": n,
            "mean_dec_cos": float(dcos[m].mean()),
            "mean_enc_cos": float(ecos[m].mean()),
            "mean_density_base": float(d_a[m].mean()),
            "mean_density_step10": float(d_b[m].mean()),
            "n_born": n_born, "n_died": n_died,
            "n_stayed_alive": n_alive, "n_stayed_dead": n_dead,
            "born_frac": born_frac,
            "dec_med_thresh": dec_med, "enc_med_thresh": enc_med,
            "corr_dec_enc": corr,
        })

    # Top row: scatter colored by lifecycle status
    ax = axes[0, col]
    for st, color, label in (
        ("stayed_alive", PALETTE["dusty"],    "stayed alive"),
        ("stayed_dead",  PALETTE["rose"],     "stayed dead"),
        ("born",         PALETTE["lilac"],    "born"),
        ("died",         PALETTE["twilight"], "died"),
    ):
        m = status == st
        if m.any():
            ax.scatter(dcos[m], ecos[m], s=5, alpha=0.55,
                       color=color, label=f"{label} (n={int(m.sum())})")
    ax.axvline(dec_med, color="grey", linestyle="--", linewidth=0.6)
    ax.axhline(enc_med, color="grey", linestyle="--", linewidth=0.6)
    ax.set_title(f"{chain} L18  cold-start  corr={corr:+.2f}")
    ax.set_xlabel("decoder cosine")
    ax.set_ylabel("encoder cosine")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.2)

    # Bottom row: born fraction per quadrant
    ax = axes[1, col]
    quad_n = {q: int((quad == q).sum()) for q in QUADS}
    quad_born_frac = {q: float(((quad == q) & (status == "born")).sum() /
                               max(quad_n[q], 1)) for q in QUADS}
    overall_born_frac = float((status == "born").sum() / len(status))
    x = np.arange(len(QUADS))
    bars = ax.bar(x, [quad_born_frac[q] for q in QUADS],
                  color=PALETTE["lilac"], edgecolor=PALETTE["twilight"])
    ax.axhline(overall_born_frac, color=PALETTE["twilight"],
               linestyle="--", linewidth=1,
               label=f"overall born frac = {overall_born_frac:.3f}")
    for i, q in enumerate(QUADS):
        ax.text(i, quad_born_frac[q] + 0.005,
                f"n={quad_n[q]}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(QUADS, rotation=15, ha="right")
    ax.set_ylabel("fraction of features 'born' in this quadrant")
    ax.set_title(f"{chain} L18: born-feature concentration by quadrant")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, axis="y")

fig.suptitle(f"L18 encoder/decoder decoupling probe (cold-start: "
             f"{STAGE_A} -> {STAGE_B})")
fig.tight_layout()

out_fig = ROOT / "figures" / "fig_l18_decoupling_probe.png"
fig.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote figure -> {out_fig.relative_to(ROOT)}")

out_csv = ROOT / "results" / "l18_decoupling_probe.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
    w.writeheader()
    w.writerows(rows_csv)
print(f"wrote CSV    -> {out_csv.relative_to(ROOT)}")
