"""
Step 6: Compare SAE features across training stages.

Loads TopK SAEs saved by 05_train_sae.py and compares:
- Feature activation frequency and L0 distributions
- Dead / born / stable feature counts across sft → ppo_step* stages
- Decoder weight cosine similarity (feature drift)

Usage:
    python scripts/06_analyze_features.py \
        --sae_dir checkpoints/saes \
        --activations_dir data/activations \
        --output_dir results/feature_analysis
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    """Matches the architecture used in 05_train_sae.py."""

    def __init__(self, d_model, d_sae, k):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.d_sae = d_sae
        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)

    def encode(self, x):
        z = self.encoder(x - self.b_pre)
        topk_values, topk_indices = torch.topk(z, self.k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topk_indices, topk_values)
        return z_sparse

    def forward(self, x):
        z_sparse = self.encode(x)
        return self.decoder(z_sparse), z_sparse


def load_sae_from_disk(sae_path: str, device: str = "cpu"):
    """Load a TopK SAE saved as a .pt dict with 'state_dict' and 'config'."""
    ckpt = torch.load(sae_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sae = TopKSAE(cfg["d_model"], cfg["d_sae"], cfg["k"])
    sae.load_state_dict(ckpt["state_dict"], strict=False)
    sae = sae.to(device)
    return sae, cfg


def collect_activations_for_sae(sae, acts: torch.Tensor, device: str = "cpu", batch_size: int = 512):
    """Run activations through the SAE encoder and collect per-feature stats."""
    sae = sae.to(device)
    acts = acts.to(device).float()

    all_z = []
    for i in range(0, len(acts), batch_size):
        batch = acts[i : i + batch_size]
        with torch.no_grad():
            feature_acts = sae.encode(batch)
        all_z.append(feature_acts.cpu())

    z = torch.cat(all_z, dim=0)
    active = (z != 0).float()

    freq = active.mean(dim=0).numpy()
    z_sum = z.sum(dim=0).numpy()
    count = active.sum(dim=0).numpy()
    mean_mag = np.divide(z_sum, count, out=np.zeros_like(z_sum), where=count > 0)
    mean_l0 = active.sum(dim=-1).mean().item()

    return {
        "freq": freq,
        "mean_magnitude": mean_mag,
        "mean_l0": mean_l0,
        "z": z,
    }


def plot_frequency_comparison(stats_by_stage: dict, layer: int, output_dir: str) -> None:
    stages = list(stats_by_stage.keys())
    fig, axes = plt.subplots(1, len(stages), figsize=(5 * len(stages), 4))
    if len(stages) == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages):
        freq = stats_by_stage[stage]["freq"]
        active_freq = freq[freq > 0]
        ax.hist(active_freq, bins=50, alpha=0.7)
        l0 = stats_by_stage[stage]["mean_l0"]
        ax.set_title(f"{stage} layer {layer}\nmean L0={l0:.1f}")
        ax.set_xlabel("Activation frequency")
        ax.set_ylabel("Number of features")
        ax.set_yscale("log")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"freq_layer{layer}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved plot -> {out_path}")


def compute_feature_drift(saes_by_stage: dict, layer: int, output_dir: str) -> None:
    """Cosine similarity of decoder weight columns between consecutive stages."""
    stages = list(saes_by_stage.keys())
    if len(stages) < 2:
        return

    for i in range(len(stages) - 1):
        s1, s2 = stages[i], stages[i + 1]
        # decoder.weight shape is (d_model, d_sae); transpose to (d_sae, d_model)
        W1 = saes_by_stage[s1].decoder.weight.detach().float().T
        W2 = saes_by_stage[s2].decoder.weight.detach().float().T

        # Normalise columns
        W1n = W1 / (W1.norm(dim=1, keepdim=True) + 1e-8)
        W2n = W2 / (W2.norm(dim=1, keepdim=True) + 1e-8)

        # Best-match cosine similarity: for each feature in s1, find max sim in s2
        sim_matrix = W1n @ W2n.T  # (d_sae, d_sae)
        max_sim = sim_matrix.max(dim=1).values.cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(max_sim, bins=50, alpha=0.7)
        ax.set_title(f"Feature drift {s1}→{s2}, layer {layer}")
        ax.set_xlabel("Max cosine similarity (best match in target)")
        ax.set_ylabel("Feature count")
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"drift_{s1}_{s2}_layer{layer}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()

        frac_stable = (max_sim > 0.9).mean()
        frac_drifted = (max_sim < 0.5).mean()
        print(f"  Drift {s1}→{s2}: stable(>0.9)={frac_stable:.1%}  drifted(<0.5)={frac_drifted:.1%}")
        print(f"  Saved plot -> {out_path}")


def compute_dead_alive_features(stats_by_stage: dict, threshold: float = 0.01) -> dict:
    stages = list(stats_by_stage.keys())
    results = {}
    for i in range(len(stages) - 1):
        s1, s2 = stages[i], stages[i + 1]
        alive1 = stats_by_stage[s1]["freq"] > threshold
        alive2 = stats_by_stage[s2]["freq"] > threshold
        results[f"{s1}->{s2}"] = {
            "born":       int((~alive1 & alive2).sum()),
            "died":       int((alive1 & ~alive2).sum()),
            "stable":     int((alive1 & alive2).sum()),
            "always_dead": int((~alive1 & ~alive2).sum()),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_dir", default="checkpoints/saes")
    parser.add_argument("--activations_dir", default="data/activations")
    parser.add_argument("--output_dir", default="results/feature_analysis")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover available (stage, layer) pairs
    # Expected layout: checkpoints/saes_lens/{stage}_layer{N}/
    sae_root = Path(args.sae_dir)
    act_root = Path(args.activations_dir)

    by_layer: dict[int, list[str]] = defaultdict(list)
    sae_paths: dict[tuple, Path] = {}
    act_paths: dict[tuple, Path] = {}

    # SAEs are saved as sae_{stage}_layer{N}.pt
    for f in sorted(sae_root.glob("sae_*.pt")):
        # strip leading "sae_" then split on "_layer"
        name = f.stem[len("sae_"):]
        parts = name.rsplit("_layer", 1)
        if len(parts) == 2:
            stage, layer_str = parts
            layer = int(layer_str)
            by_layer[layer].append(stage)
            sae_paths[(stage, layer)] = f

    for f in sorted(act_root.glob("*.pt")):
        parts = f.stem.rsplit("_layer", 1)
        if len(parts) == 2:
            stage, layer_str = parts
            act_paths[(stage, int(layer_str))] = f

    # Chronological order: SFT baseline → PPO steps
    stage_order = ["sft", "ppo_step10", "ppo_step50", "ppo_step100",
                   "ppo_step200", "ppo_step300", "ppo_step435"]

    for layer in sorted(by_layer.keys()):
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        stats_by_stage: dict[str, dict] = {}
        saes_by_stage: dict[str, object] = {}

        for stage in stage_order:
            key = (stage, layer)
            if key not in sae_paths:
                continue

            print(f"\n  Stage: {stage}")
            try:
                sae, _ = load_sae_from_disk(str(sae_paths[key]), device=args.device)
            except Exception as e:
                print(f"  Could not load SAE: {e}")
                continue

            saes_by_stage[stage] = sae

            if key in act_paths:
                acts = torch.load(act_paths[key], weights_only=True)
                stats = collect_activations_for_sae(sae, acts, args.device)
                stats_by_stage[stage] = stats

                n_active = (stats["freq"] > 0.01).sum()
                n_dead = (stats["freq"] <= 0.01).sum()
                print(f"    mean L0:           {stats['mean_l0']:.1f}")
                print(f"    active features:   {n_active}")
                print(f"    dead features:     {n_dead}")
            else:
                print(f"    (no cached activations found for {stage} layer {layer})")

        if stats_by_stage:
            plot_frequency_comparison(stats_by_stage, layer, args.output_dir)
            lifecycle = compute_dead_alive_features(stats_by_stage)
            for transition, counts in lifecycle.items():
                print(f"\n  Feature lifecycle {transition}:")
                for k, v in counts.items():
                    print(f"    {k}: {v}")

        if len(saes_by_stage) >= 2:
            compute_feature_drift(saes_by_stage, layer, args.output_dir)

    print(f"\nDone. Results -> {args.output_dir}")


if __name__ == "__main__":
    main()
