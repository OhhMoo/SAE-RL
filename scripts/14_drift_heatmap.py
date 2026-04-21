#!/usr/bin/env python3
"""
14_drift_heatmap.py
Produce a single drift heatmap figure instead of 24 separate histogram plots.

For every consecutive checkpoint pair × every layer, compute:
  - frac_stable:  fraction of features with best-match cosine sim > 0.9
  - frac_drifted: fraction of features with best-match cosine sim < 0.5
  - mean_max_sim: mean best-match cosine similarity

Then plot a 2-panel heatmap:
  Panel A — fraction of drifted features (cosine sim < 0.5)  [highlight = more change]
  Panel B — fraction of stable features (cosine sim > 0.9)   [highlight = less change]

Rows = layers (6, 12, 18, 23)
Cols = checkpoint transitions (SFT→10, 10→50, ..., 300→435)

The transition that stands out in panel A (most drift) is where RL-induced
representational change concentrates.

Usage:
    python scripts/14_drift_heatmap.py
    python scripts/14_drift_heatmap.py --sae_dir checkpoints/saes --output_dir results/feature_analysis
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ── SAE ───────────────────────────────────────────────────────────────────────

class TopKSAE(nn.Module):
    def __init__(self, d_model, d_sae, k):
        super().__init__()
        self.k = k
        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.encoder = nn.Linear(d_model, d_sae)
        self.decoder = nn.Linear(d_sae, d_model)

    def encode(self, x):
        z = self.encoder(x - self.b_pre)
        topk_vals, topk_idx = torch.topk(z, self.k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topk_idx, topk_vals)
        return z_sparse

    def forward(self, x):
        z = self.encode(x)
        return self.decoder(z), z


def load_sae(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sae = TopKSAE(cfg["d_model"], cfg["d_sae"], cfg["k"])
    sae.load_state_dict(ckpt["state_dict"], strict=False)
    return sae.to(device).eval(), cfg


# ── Drift computation ─────────────────────────────────────────────────────────

def compute_drift_stats(sae1: TopKSAE, sae2: TopKSAE,
                        chunk_size: int = 512) -> dict:
    """
    For each feature in sae1, find its best-matching feature in sae2 by cosine
    similarity of decoder weight columns.

    Returns dict with frac_stable, frac_drifted, mean_max_sim, and the full
    max_sim array.
    """
    # decoder.weight shape: (d_model, d_sae)
    W1 = sae1.decoder.weight.detach().float().T  # (d_sae, d_model)
    W2 = sae2.decoder.weight.detach().float().T

    W1n = nn.functional.normalize(W1, dim=1)
    W2n = nn.functional.normalize(W2, dim=1)

    d_sae = W1n.shape[0]
    max_sim = torch.full((d_sae,), -1.0)

    # Process in chunks to avoid OOM on large d_sae
    for start in range(0, d_sae, chunk_size):
        end = min(start + chunk_size, d_sae)
        sim_chunk = W1n[start:end] @ W2n.T     # (chunk, d_sae)
        max_sim[start:end] = sim_chunk.max(dim=1).values

    max_sim_np = max_sim.cpu().numpy()
    return {
        "max_sim":      max_sim_np,
        "frac_stable":  float((max_sim_np > 0.9).mean()),
        "frac_drifted": float((max_sim_np < 0.5).mean()),
        "mean_max_sim": float(max_sim_np.mean()),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

# Chronological stage order — must match the SAE filenames
STAGE_ORDER = [
    "sft",
    "ppo_step10",
    "ppo_step50",
    "ppo_step100",
    "ppo_step110",
    "ppo_step120",
    "ppo_step130",
    "ppo_step140",
    "ppo_step150",
    "ppo_step160",
    "ppo_step170",
    "ppo_step180",
    "ppo_step190",
    "ppo_step200",
    "ppo_step300",
    "ppo_step435",
]

# Short labels for the x-axis
STAGE_LABELS = {
    "sft": "SFT",
    "ppo_step10": "s10",
    "ppo_step50": "s50",
    "ppo_step100": "s100",
    "ppo_step110": "s110",
    "ppo_step120": "s120",
    "ppo_step130": "s130",
    "ppo_step140": "s140",
    "ppo_step150": "s150",
    "ppo_step160": "s160",
    "ppo_step170": "s170",
    "ppo_step180": "s180",
    "ppo_step190": "s190",
    "ppo_step200": "s200",
    "ppo_step300": "s300",
    "ppo_step435": "s435",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_dir",    default="checkpoints/saes",
                        help="Primary SAE directory (7 sparse checkpoints)")
    parser.add_argument("--dense_sae_dir", default=None,
                        help="Dense SAE directory (onset window steps 110-190). "
                             "If omitted, only --sae_dir is used.")
    parser.add_argument("--output_dir", default="results/feature_analysis")
    parser.add_argument("--layers",     type=int, nargs="+", default=[6, 12, 18, 23])
    parser.add_argument("--device",     default="cpu",
                        help="cpu is fine — only decoder weights needed, no activations")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sae_root = Path(args.sae_dir)

    # ── Discover available SAEs ───────────────────────────────────────────────
    search_dirs = [sae_root]
    if args.dense_sae_dir:
        search_dirs.append(Path(args.dense_sae_dir))

    available: dict[tuple, Path] = {}
    for search_dir in search_dirs:
        if not search_dir.exists():
            print(f"[warn] SAE dir not found: {search_dir}")
            continue
        for f in sorted(search_dir.glob("sae_*.pt")):
            stem = f.stem[len("sae_"):]
            parts = stem.rsplit("_layer", 1)
            if len(parts) == 2:
                stage, layer_str = parts
                try:
                    available[(stage, int(layer_str))] = f
                except ValueError:
                    pass

    # Filter to only stages we actually have
    stages_present = [s for s in STAGE_ORDER
                      if any((s, l) in available for l in args.layers)]

    # Consecutive transitions
    transitions = [(stages_present[i], stages_present[i + 1])
                   for i in range(len(stages_present) - 1)]
    trans_labels = [f"{STAGE_LABELS.get(s1, s1)}→{STAGE_LABELS.get(s2, s2)}"
                    for s1, s2 in transitions]

    print(f"Found {len(stages_present)} stages, {len(transitions)} transitions, "
          f"{len(args.layers)} layers")

    # ── Compute drift stats ───────────────────────────────────────────────────
    # drifted[layer_idx][trans_idx], stable[layer_idx][trans_idx]
    drifted_matrix = np.full((len(args.layers), len(transitions)), np.nan)
    stable_matrix  = np.full((len(args.layers), len(transitions)), np.nan)
    mean_sim_matrix = np.full((len(args.layers), len(transitions)), np.nan)

    csv_rows = []

    for li, layer in enumerate(args.layers):
        print(f"\nLayer {layer}")
        for ti, (s1, s2) in enumerate(transitions):
            k1 = (s1, layer)
            k2 = (s2, layer)
            if k1 not in available or k2 not in available:
                print(f"  {s1}→{s2}: missing SAE(s) — skipping")
                continue

            print(f"  {s1}→{s2}:", end="", flush=True)
            sae1, _ = load_sae(str(available[k1]), args.device)
            sae2, _ = load_sae(str(available[k2]), args.device)
            stats = compute_drift_stats(sae1, sae2)

            drifted_matrix[li, ti]  = stats["frac_drifted"]
            stable_matrix[li, ti]   = stats["frac_stable"]
            mean_sim_matrix[li, ti] = stats["mean_max_sim"]

            print(f"  drifted={stats['frac_drifted']:.1%}  "
                  f"stable={stats['frac_stable']:.1%}  "
                  f"mean_sim={stats['mean_max_sim']:.3f}")

            csv_rows.append({
                "layer": layer,
                "from_stage": s1,
                "to_stage": s2,
                "transition": trans_labels[ti],
                "frac_drifted": stats["frac_drifted"],
                "frac_stable":  stats["frac_stable"],
                "mean_max_sim": stats["mean_max_sim"],
            })

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = os.path.join(args.output_dir, "drift_heatmap_data.csv")
    pd.DataFrame(csv_rows).to_csv(out_csv, index=False)
    print(f"\nData saved → {out_csv}")

    # ── Plot heatmap ──────────────────────────────────────────────────────────
    layer_labels = [f"Layer {l}" for l in args.layers]
    n_trans = len(transitions)
    n_layers = len(args.layers)

    fig, axes = plt.subplots(1, 3, figsize=(max(14, n_trans * 0.8), 4))
    fig.suptitle("Feature Drift Across Training — All Layers & Transitions",
                 fontweight="bold", fontsize=12)

    panels = [
        (drifted_matrix,  "Fraction drifted (cos sim < 0.5)",  "Reds",   "High = major representational change"),
        (stable_matrix,   "Fraction stable (cos sim > 0.9)",   "Greens", "High = features preserved"),
        (mean_sim_matrix, "Mean best-match cosine similarity",  "RdYlGn", "Higher = more similar to previous ckpt"),
    ]

    for ax, (mat, title, cmap, subtitle) in zip(axes, panels):
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0,
                       vmax=1 if "sim" not in title.lower() else 1)
        ax.set_xticks(range(n_trans))
        ax.set_xticklabels(trans_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels(layer_labels, fontsize=9)
        ax.set_title(f"{title}\n{subtitle}", fontsize=9)

        # Annotate cells with values
        for i in range(n_layers):
            for j in range(n_trans):
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color="black" if val < 0.7 else "white")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Mark the clean/hacking boundary with a vertical line
    # Find the transition index corresponding to step 100→200 (or nearest onset window)
    boundary_trans = None
    onset_trans = None
    for ti, (s1, s2) in enumerate(transitions):
        if s1 == "ppo_step100" and s2 in ("ppo_step110", "ppo_step200"):
            boundary_trans = ti
        if s1 in ("ppo_step190", "ppo_step200") and s2 in ("ppo_step200", "ppo_step300"):
            onset_trans = ti

    for ax in axes:
        if boundary_trans is not None:
            ax.axvline(boundary_trans - 0.5, color="navy", linestyle="--",
                       linewidth=1.5, alpha=0.8)
            ax.text(boundary_trans - 0.4, -0.7, "onset\nwindow",
                    color="navy", fontsize=7, va="top",
                    transform=ax.transData)

    plt.tight_layout()
    out_fig = os.path.join(args.output_dir, "drift_heatmap.png")
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved → {out_fig}")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n── Drift Summary (sorted by frac_drifted desc) ─────────────────")
    summary = pd.DataFrame(csv_rows).sort_values("frac_drifted", ascending=False)
    print(summary[["layer", "transition", "frac_drifted", "frac_stable", "mean_max_sim"]]
          .head(10).to_string(index=False))


if __name__ == "__main__":
    main()
