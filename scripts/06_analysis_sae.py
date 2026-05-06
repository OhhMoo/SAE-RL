#!/usr/bin/env python3
"""
SAE analyses for the SAE x RL project.

Subcommands:
  drift              decoder + encoder cosine between consecutive PPO stages,
                     aggregated to per-layer x per-transition heatmaps.
                     Outputs:
                       results/decoder_drift.csv
                       figures/fig1_drift_heatmap.png
                       figures/fig1b_encoder_drift_heatmap.png

  feature_match      best-match (renaming-aware) cosine vs index-aligned.
                     Tells you whether features stay put or rename across
                     stages. Optional Hungarian 1:1 matching via --hungarian.
                     Outputs:
                       results/feature_matching.csv
                       figures/feature_matching_overlay.png

  feature_lifecycle  per-feature firing density per (chain, layer, stage),
                     plus death/birth events per consecutive transition,
                     plus Shannon entropy of the firing distribution.
                     Needs val activations + SAE encoder.
                     Outputs:
                       results/feature_lifecycle.csv
                       results/feature_lifecycle_events.csv
                       results/cache/density_<chain>_L<n>_<stage>.pt
                       figures/feature_lifecycle_events.png

  feature_drift_top  rank features per layer by total decoder drift
                     (instruct_base -> final stage). Joins density from the
                     lifecycle cache when available. Use --min_density_end
                     to restrict the ranking pool to features that are still
                     active at the final stage.
                     Outputs:
                       results/top_drifted_features.csv         (unfiltered)
                       results/top_drifted_features_active.csv  (with filter)

  enc_vs_dec         per-feature scatter of decoder vs encoder cosine for the
                     first PPO transition of each chain. Highlights layers
                     where encoder/decoder drift decouple (e.g. L18 early).
                     Outputs:
                       figures/fig_enc_vs_dec_first_transition.png

  pipeline           run drift -> feature_match -> feature_lifecycle ->
                     feature_drift_top (unfiltered, then optionally filtered)
                     -> enc_vs_dec end-to-end.

Run from sae_rl/:
    python scripts/06_analysis_sae.py pipeline
    python scripts/06_analysis_sae.py drift
    python scripts/06_analysis_sae.py feature_lifecycle --device cuda
    python scripts/06_analysis_sae.py feature_match --hungarian
    python scripts/06_analysis_sae.py feature_drift_top --top_n 30
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch
import torch.nn as nn


PALETTE = {
    "blush":    "#FAE8EB",
    "rose":     "#F6CACA",
    "dusty":    "#E4C2C6",
    "lilac":    "#CD9FCC",
    "twilight": "#0A014F",
}
CMAP = LinearSegmentedColormap.from_list(
    "blush_twilight",
    [PALETTE["twilight"], PALETTE["lilac"], PALETTE["dusty"],
     PALETTE["rose"], PALETTE["blush"]],
)
COLOR_FLEX = PALETTE["lilac"]
COLOR_STRICT = PALETTE["twilight"]


# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
CACHE_DIR = RESULTS_DIR / "cache"

LAYERS = [6, 12, 18, 23]
DEAD_THRESHOLD = 1e-4   # matches train-time dead threshold

CHAINS = {
    "flexible": {
        "stages": [
            "instruct_base",
            "ppo_step10", "ppo_step30", "ppo_step60",
            "ppo_step100", "ppo_step140", "ppo_step180", "ppo_step200",
        ],
        "default_dir": ROOT / "checkpoints" / "saes",
        "layer_overrides": {},
        "activations_dir": ROOT / "data" / "activations",
    },
    "strict": {
        "stages": [
            "instruct_base",
            "ppo_step10", "ppo_step30", "ppo_step50",
            "ppo_step80", "ppo_step116",
        ],
        "default_dir": ROOT / "checkpoints" / "saes_strict",
        "layer_overrides": {
            23: ROOT / "checkpoints" / "saes_strict_l23_k256",
        },
        "activations_dir": ROOT / "data" / "activations_strict",
    },
    "shuffled": {
        "stages": [
            "instruct_base", "ppo_step10", "ppo_step30",
        ],
        "default_dir": ROOT / "checkpoints" / "saes_shuffled",
        "layer_overrides": {},
        "activations_dir": ROOT / "data" / "activations_shuffled",
    },
}


# ---------------------------------------------------------------------------
# TopK SAE -- must match scripts/05_train_sae.py
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


# ---------------------------------------------------------------------------
# Path / load helpers
# ---------------------------------------------------------------------------

def stage_step(stage: str) -> int:
    if stage == "instruct_base":
        return 0
    m = re.match(r"ppo_step(\d+)$", stage)
    if not m:
        raise ValueError(f"unrecognized stage name: {stage}")
    return int(m.group(1))


def sae_path(chain: str, layer: int, stage: str) -> Path:
    cfg = CHAINS[chain]
    base = cfg["layer_overrides"].get(layer, cfg["default_dir"])
    return base / f"sae_{stage}_layer{layer}.pt"


def activation_path(chain: str, layer: int, stage: str, split: str = "val") -> Path:
    return CHAINS[chain]["activations_dir"] / f"{stage}_layer{layer}_{split}.pt"


def load_decoder_weight(path: Path) -> torch.Tensor:
    """decoder.weight is (d_model, d_sae) -- features are columns."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["state_dict"]["decoder.weight"].float()


def load_encoder_weight(path: Path) -> torch.Tensor:
    """encoder.weight is (d_sae, d_model) -- feature read directions are rows."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["state_dict"]["encoder.weight"].float()


def load_sae(path: Path, device: str = "cpu") -> tuple[TopKSAE, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sae = TopKSAE(cfg["d_model"], cfg["d_sae"], cfg["k"])
    sae.load_state_dict(ckpt["state_dict"], strict=False)
    return sae.to(device).eval(), cfg


# ---------------------------------------------------------------------------
# Cosine helpers
# ---------------------------------------------------------------------------

def per_feature_decoder_cosine(W_a: torch.Tensor, W_b: torch.Tensor) -> torch.Tensor:
    """W_a, W_b are (d_model, d_sae). Returns (d_sae,)."""
    if W_a.shape != W_b.shape:
        raise ValueError(f"decoder shape mismatch {W_a.shape} vs {W_b.shape}")
    a = nn.functional.normalize(W_a, dim=0)
    b = nn.functional.normalize(W_b, dim=0)
    return (a * b).sum(dim=0)


def per_feature_encoder_cosine(W_a: torch.Tensor, W_b: torch.Tensor) -> torch.Tensor:
    """W_a, W_b are (d_sae, d_model). Returns (d_sae,)."""
    if W_a.shape != W_b.shape:
        raise ValueError(f"encoder shape mismatch {W_a.shape} vs {W_b.shape}")
    a = nn.functional.normalize(W_a, dim=1)
    b = nn.functional.normalize(W_b, dim=1)
    return (a * b).sum(dim=1)


def best_match_decoder_cosine(W_a: torch.Tensor, W_b: torch.Tensor,
                              hungarian: bool = False
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """For each column of W_a, find the best-cosine column in W_b.

    Returns:
      best_cos: (d_sae,) cosine of base feature i to its best match.
      best_idx: (d_sae,) destination feature index.
    """
    if W_a.shape != W_b.shape:
        raise ValueError(f"shape mismatch {W_a.shape} vs {W_b.shape}")
    a = nn.functional.normalize(W_a, dim=0)
    b = nn.functional.normalize(W_b, dim=0)
    sim = a.t() @ b   # (d_sae, d_sae)
    if hungarian:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-sim.numpy())
        best_idx = torch.from_numpy(col_ind).long()
        best_cos = sim[torch.from_numpy(row_ind).long(), best_idx]
    else:
        best_cos, best_idx = sim.max(dim=1)
    return best_cos, best_idx


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"  no rows to write for {path}")
        return
    keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {len(rows)} rows -> {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 1. Drift heatmap (decoder + encoder)
# ---------------------------------------------------------------------------

def compute_drift(chain: str) -> list[dict]:
    stages = CHAINS[chain]["stages"]
    rows: list[dict] = []
    for layer in LAYERS:
        prev_dec, prev_enc, prev_stage = None, None, None
        for stage in stages:
            path = sae_path(chain, layer, stage)
            if not path.exists():
                print(f"  [skip] {chain} L{layer} {stage}: missing {path.name}")
                prev_dec, prev_enc, prev_stage = None, None, None
                continue
            dec = load_decoder_weight(path)
            enc = load_encoder_weight(path)
            if prev_dec is not None:
                try:
                    dcos = per_feature_decoder_cosine(prev_dec, dec)
                    ecos = per_feature_encoder_cosine(prev_enc, enc)
                except ValueError as e:
                    print(f"  [skip] {chain} L{layer} {prev_stage}->{stage}: {e}")
                    prev_dec, prev_enc, prev_stage = dec, enc, stage
                    continue
                ds = stage_step(stage) - stage_step(prev_stage)
                d_mean = dcos.mean().item()
                e_mean = ecos.mean().item()
                rows.append({
                    "chain": chain,
                    "layer": layer,
                    "stage_from": prev_stage,
                    "stage_to": stage,
                    "step_from": stage_step(prev_stage),
                    "step_to": stage_step(stage),
                    "d_step": ds,
                    "n_features": dcos.numel(),
                    "dec_cos_mean": d_mean,
                    "dec_cos_median": dcos.median().item(),
                    "dec_cos_q25": torch.quantile(dcos, 0.25).item(),
                    "dec_cos_q75": torch.quantile(dcos, 0.75).item(),
                    "dec_cos_min": dcos.min().item(),
                    "dec_drift_per_step": (1.0 - d_mean) / max(ds, 1),
                    "enc_cos_mean": e_mean,
                    "enc_cos_median": ecos.median().item(),
                    "enc_drift_per_step": (1.0 - e_mean) / max(ds, 1),
                    "enc_dec_corr":
                        float(np.corrcoef(dcos.numpy(), ecos.numpy())[0, 1])
                        if dcos.numel() > 1 else float("nan"),
                })
            prev_dec, prev_enc, prev_stage = dec, enc, stage
    return rows


def _grid(rows, chain, value):
    sub = [r for r in rows if r["chain"] == chain]
    transitions: list[tuple[str, str]] = []
    for r in sub:
        t = (r["stage_from"], r["stage_to"])
        if t not in transitions:
            transitions.append(t)
    grid = np.full((len(LAYERS), len(transitions)), np.nan)
    for r in sub:
        i = LAYERS.index(r["layer"])
        j = transitions.index((r["stage_from"], r["stage_to"]))
        grid[i, j] = r[value]
    return grid, transitions


def _chains_with_data(rows):
    """Return chains in CHAINS order that have any rows in `rows`."""
    seen = {r["chain"] for r in rows}
    return tuple(c for c in CHAINS if c in seen)


def plot_heatmap(rows, path, value, title):
    chains = _chains_with_data(rows) or tuple(CHAINS.keys())
    grids = {c: _grid(rows, c, value) for c in chains}
    widths = [max(grids[c][0].shape[1], 1) for c in chains]
    finite = [r[value] for r in rows if np.isfinite(r[value])]
    vmin, vmax = (min(finite), max(finite)) if finite else (0.0, 1.0)
    fig, axes = plt.subplots(
        1, len(chains), figsize=(2 + 0.7 * sum(widths), 0.6 * len(LAYERS) + 2),
        gridspec_kw={"width_ratios": widths},
        squeeze=False,
    )
    axes = axes[0]
    im = None
    for ax, chain in zip(axes, chains):
        grid, transitions = grids[chain]
        im = ax.imshow(grid, aspect="auto", cmap=CMAP, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(transitions)))
        ax.set_xticklabels(
            [f"{stage_step(a)}->{stage_step(b)}" for a, b in transitions],
            rotation=45, ha="right", fontsize=8,
        )
        ax.set_yticks(range(len(LAYERS)))
        ax.set_yticklabels([f"L{l}" for l in LAYERS])
        ax.set_title(f"{chain}  (n={grid.shape[1]})")
        ax.set_xlabel("PPO transition")
        midpoint = 0.5 * (vmin + vmax)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if np.isfinite(v):
                    ax.text(
                        j, i, f"{v:.3f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if v < midpoint else "black",
                    )
    axes[0].set_ylabel("layer")
    fig.suptitle(title)
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figure -> {path.relative_to(ROOT)}")


def cmd_drift(args) -> None:
    print("[drift] decoder + encoder cosine across consecutive stages")
    all_rows: list[dict] = []
    for chain in tuple(CHAINS.keys()):
        print(f"  chain: {chain}")
        all_rows.extend(compute_drift(chain))
    write_csv(all_rows, RESULTS_DIR / "decoder_drift.csv")
    plot_heatmap(
        all_rows, FIGURES_DIR / "fig1_drift_heatmap.png",
        args.value, f"Decoder cosine ({args.value})",
    )
    plot_heatmap(
        all_rows, FIGURES_DIR / "fig1b_encoder_drift_heatmap.png",
        "enc_cos_mean", "Encoder cosine (enc_cos_mean)",
    )
    print("\n  summary  (mean per-step drift over layers x transitions):")
    for chain in tuple(CHAINS.keys()):
        sub = [r for r in all_rows if r["chain"] == chain]
        if sub:
            d = float(np.mean([r["dec_drift_per_step"] for r in sub]))
            e = float(np.mean([r["enc_drift_per_step"] for r in sub]))
            corr = float(np.mean([r["enc_dec_corr"] for r in sub
                                  if np.isfinite(r["enc_dec_corr"])]))
            print(f"    {chain:<9} dec={d:.6f}  enc={e:.6f}  "
                  f"corr(enc,dec)={corr:.3f}  n={len(sub)}")


# ---------------------------------------------------------------------------
# 2. Cross-stage feature matching (renaming-aware)
# ---------------------------------------------------------------------------

def compute_feature_match(chain: str, hungarian: bool = False) -> list[dict]:
    stages = CHAINS[chain]["stages"]
    rows: list[dict] = []
    for layer in LAYERS:
        prev_dec, prev_stage = None, None
        for stage in stages:
            path = sae_path(chain, layer, stage)
            if not path.exists():
                prev_dec, prev_stage = None, None
                continue
            dec = load_decoder_weight(path)
            if prev_dec is not None:
                try:
                    idx_cos = per_feature_decoder_cosine(prev_dec, dec)
                    best_cos, best_idx = best_match_decoder_cosine(
                        prev_dec, dec, hungarian=hungarian)
                except ValueError as e:
                    print(f"  [skip] {chain} L{layer} {prev_stage}->{stage}: {e}")
                    prev_dec, prev_stage = dec, stage
                    continue
                for i in range(idx_cos.numel()):
                    rows.append({
                        "chain": chain,
                        "layer": layer,
                        "stage_from": prev_stage,
                        "stage_to": stage,
                        "step_from": stage_step(prev_stage),
                        "step_to": stage_step(stage),
                        "feature_idx": int(i),
                        "cos_index_aligned": float(idx_cos[i].item()),
                        "cos_best_match": float(best_cos[i].item()),
                        "best_match_idx": int(best_idx[i].item()),
                        "renaming_gap": float((best_cos[i] - idx_cos[i]).item()),
                    })
                print(f"    {chain} L{layer} {prev_stage}->{stage}: "
                      f"idx_mean={idx_cos.mean():.3f}  "
                      f"best_mean={best_cos.mean():.3f}  "
                      f"gap={(best_cos.mean() - idx_cos.mean()):.4f}")
            prev_dec, prev_stage = dec, stage
    return rows


def plot_match_overlay(rows: list[dict], path: Path) -> None:
    """One panel per (chain, layer): histogram of index-aligned vs best-match
    cosine for the *final* transition of that chain.
    """
    if not rows:
        return
    chains = tuple(CHAINS.keys())
    fig, axes = plt.subplots(
        len(LAYERS), len(chains),
        figsize=(5 * len(chains), 2.4 * len(LAYERS)), squeeze=False,
    )
    for col, chain in enumerate(chains):
        sub = [r for r in rows if r["chain"] == chain]
        if not sub:
            continue
        last_to = max(r["step_to"] for r in sub)
        for row_i, layer in enumerate(LAYERS):
            ax = axes[row_i][col]
            cells = [r for r in sub
                     if r["layer"] == layer and r["step_to"] == last_to]
            if not cells:
                ax.set_visible(False)
                continue
            idx_vals = np.array([r["cos_index_aligned"] for r in cells])
            best_vals = np.array([r["cos_best_match"] for r in cells])
            lo = float(min(idx_vals.min(), best_vals.min())) - 0.01
            hi = float(max(idx_vals.max(), best_vals.max())) + 0.01
            bins = np.linspace(lo, hi, 60)
            ax.hist(idx_vals, bins=bins, alpha=0.55,
                    label="index-aligned", color=PALETTE["lilac"])
            ax.hist(best_vals, bins=bins, alpha=0.55,
                    label="best-match", color=PALETTE["twilight"])
            ax.set_yscale("log")
            ax.set_title(f"{chain} L{layer} (final transition)")
            if row_i == 0:
                ax.legend(fontsize=8)
            if row_i == len(LAYERS) - 1:
                ax.set_xlabel("decoder cosine")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figure -> {path.relative_to(ROOT)}")


def cmd_feature_match(args) -> None:
    print("[feature_match] index-aligned vs best-match decoder cosine")
    if args.hungarian:
        print("  (using Hungarian assignment -- this is slower, ~30-60s per pair)")
    all_rows: list[dict] = []
    for chain in tuple(CHAINS.keys()):
        print(f"  chain: {chain}")
        all_rows.extend(compute_feature_match(chain, hungarian=args.hungarian))
    write_csv(all_rows, RESULTS_DIR / "feature_matching.csv")
    plot_match_overlay(all_rows, FIGURES_DIR / "feature_matching_overlay.png")


# ---------------------------------------------------------------------------
# 3. Feature lifecycle (density, deaths, births, entropy)
# ---------------------------------------------------------------------------

def density_cache_path(chain: str, layer: int, stage: str) -> Path:
    return CACHE_DIR / f"density_{chain}_L{layer}_{stage}.pt"


def _density_lookup(chain: str, layer: int, stage: str) -> torch.Tensor | None:
    p = density_cache_path(chain, layer, stage)
    if p.exists():
        return torch.load(p, map_location="cpu", weights_only=False)["density"]
    return None


@torch.no_grad()
def compute_density(chain: str, layer: int, stage: str,
                    device: str = "cpu", batch_size: int = 1024,
                    refresh: bool = False
                    ) -> tuple[torch.Tensor | None, int]:
    cache = density_cache_path(chain, layer, stage)
    if cache.exists() and not refresh:
        d = torch.load(cache, map_location="cpu", weights_only=False)
        return d["density"], int(d["n_tokens"])

    sae_p = sae_path(chain, layer, stage)
    act_p = activation_path(chain, layer, stage, split="val")
    if not sae_p.exists() or not act_p.exists():
        return None, 0

    sae, cfg = load_sae(sae_p, device=device)
    acts = torch.load(act_p, map_location="cpu", weights_only=False).float()
    if acts.dim() > 2:
        acts = acts.reshape(-1, acts.shape[-1])

    nonzero = torch.zeros(cfg["d_sae"], dtype=torch.float64)
    n_tokens = 0
    for i in range(0, acts.shape[0], batch_size):
        x = acts[i:i + batch_size].to(device)
        z = sae.encode(x)
        nonzero += (z > 0).sum(dim=0).double().cpu()
        n_tokens += x.shape[0]
    density = (nonzero / max(n_tokens, 1)).float()
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"density": density, "n_tokens": n_tokens,
         "chain": chain, "layer": layer, "stage": stage},
        cache,
    )
    return density, n_tokens


def shannon_entropy(density: torch.Tensor) -> float:
    """Entropy (bits) of the firing-rate distribution (normalized density)."""
    p = density.clone().double()
    s = p.sum()
    if s.item() <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    return float(-(p * p.log2()).sum().item())


def cmd_feature_lifecycle(args) -> None:
    print("[feature_lifecycle] per-feature firing density per stage")
    rows_per_stage: list[dict] = []
    densities: dict[tuple[str, int, str], torch.Tensor] = {}
    for chain in tuple(CHAINS.keys()):
        for layer in LAYERS:
            for stage in CHAINS[chain]["stages"]:
                density, n_tok = compute_density(
                    chain, layer, stage,
                    device=args.device, batch_size=args.batch_size,
                    refresh=args.refresh,
                )
                if density is None:
                    print(f"  [skip] {chain} L{layer} {stage}: missing files")
                    continue
                alive_mask = density > DEAD_THRESHOLD
                ent = shannon_entropy(density)
                row = {
                    "chain": chain,
                    "layer": layer,
                    "stage": stage,
                    "step": stage_step(stage),
                    "n_tokens": n_tok,
                    "n_features": int(density.numel()),
                    "n_alive": int(alive_mask.sum().item()),
                    "frac_alive": float(alive_mask.float().mean().item()),
                    "entropy_bits": ent,
                    "effective_features": float(2 ** ent),
                    "mean_density": float(density.mean().item()),
                    "max_density": float(density.max().item()),
                }
                rows_per_stage.append(row)
                densities[(chain, layer, stage)] = density
                print(f"    {chain} L{layer} {stage}: "
                      f"alive={row['n_alive']}/{row['n_features']}  "
                      f"H={ent:.2f} bits  "
                      f"eff={row['effective_features']:.0f}")
    write_csv(rows_per_stage, RESULTS_DIR / "feature_lifecycle.csv")

    events: list[dict] = []
    for chain in tuple(CHAINS.keys()):
        for layer in LAYERS:
            stages = CHAINS[chain]["stages"]
            for a, b in zip(stages[:-1], stages[1:]):
                ka, kb = (chain, layer, a), (chain, layer, b)
                if ka not in densities or kb not in densities:
                    continue
                da, db = densities[ka], densities[kb]
                if da.shape != db.shape:
                    continue
                alive_a = da > DEAD_THRESHOLD
                alive_b = db > DEAD_THRESHOLD
                died = alive_a & ~alive_b
                born = ~alive_a & alive_b
                stayed_alive = alive_a & alive_b
                stayed_dead = ~alive_a & ~alive_b
                events.append({
                    "chain": chain,
                    "layer": layer,
                    "stage_from": a,
                    "stage_to": b,
                    "step_from": stage_step(a),
                    "step_to": stage_step(b),
                    "d_step": stage_step(b) - stage_step(a),
                    "n_died": int(died.sum().item()),
                    "n_born": int(born.sum().item()),
                    "n_stayed_alive": int(stayed_alive.sum().item()),
                    "n_stayed_dead": int(stayed_dead.sum().item()),
                })
    write_csv(events, RESULTS_DIR / "feature_lifecycle_events.csv")
    plot_lifecycle_events(events, FIGURES_DIR / "feature_lifecycle_events.png")
    plot_entropy_trajectory(rows_per_stage, FIGURES_DIR / "fig_effective_features.png")


def plot_lifecycle_events(events: list[dict], path: Path) -> None:
    if not events:
        return
    chains = tuple(CHAINS.keys())
    fig, axes = plt.subplots(
        len(LAYERS), len(chains),
        figsize=(5 * len(chains), 2.4 * len(LAYERS)),
        squeeze=False, sharey="row",
    )
    for col, chain in enumerate(chains):
        for row_i, layer in enumerate(LAYERS):
            ax = axes[row_i][col]
            cells = [e for e in events
                     if e["chain"] == chain and e["layer"] == layer]
            if not cells:
                ax.set_visible(False)
                continue
            x = np.arange(len(cells))
            born = np.array([c["n_born"] for c in cells])
            died = np.array([c["n_died"] for c in cells])
            ax.bar(x - 0.2, born, width=0.4, color=PALETTE["lilac"], label="born")
            ax.bar(x + 0.2, died, width=0.4, color=PALETTE["twilight"], label="died")
            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"{c['step_from']}->{c['step_to']}" for c in cells],
                rotation=45, ha="right", fontsize=7,
            )
            ax.set_title(f"{chain} L{layer}")
            if row_i == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Feature deaths / births per consecutive PPO transition")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figure -> {path.relative_to(ROOT)}")


def plot_entropy_trajectory(rows: list[dict], path: Path) -> None:
    """One panel per layer; lines for flexible vs strict; y = effective
    feature count (= 2^entropy of the firing distribution).
    """
    if not rows:
        return
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(3.5 * len(LAYERS), 3.5),
                             squeeze=False)
    axes = axes[0]
    for ax, layer in zip(axes, LAYERS):
        for chain, color, marker in (
            ("flexible", COLOR_FLEX, "o"),
            ("strict", COLOR_STRICT, "s"),
        ):
            cells = sorted(
                [r for r in rows
                 if r["chain"] == chain and r["layer"] == layer],
                key=lambda r: r["step"],
            )
            if not cells:
                continue
            xs = [c["step"] for c in cells]
            ys = [c["effective_features"] for c in cells]
            ax.plot(xs, ys, marker=marker, color=color, linewidth=2,
                    label=chain)
        ax.set_title(f"L{layer}")
        ax.set_xlabel("PPO step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("effective features  (= 2^entropy)")
    axes[0].legend(fontsize=9, loc="best")
    fig.suptitle("Effective feature count across PPO stages")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figure -> {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# 4. Top-drifted features (instruct_base -> final stage)
# ---------------------------------------------------------------------------

def cmd_feature_drift_top(args) -> None:
    print("[feature_drift_top] per-feature decoder cosine, base -> final")
    if args.min_density_end > 0:
        print(f"  filtering pool to features with density_end >= "
              f"{args.min_density_end:.1e}")
    rows: list[dict] = []
    for chain in tuple(CHAINS.keys()):
        stages = CHAINS[chain]["stages"]
        first, last = stages[0], stages[-1]
        for layer in LAYERS:
            p0, p1 = sae_path(chain, layer, first), sae_path(chain, layer, last)
            if not (p0.exists() and p1.exists()):
                print(f"  [skip] {chain} L{layer}: missing endpoint SAE")
                continue
            try:
                cos = per_feature_decoder_cosine(
                    load_decoder_weight(p0), load_decoder_weight(p1))
                ecos = per_feature_encoder_cosine(
                    load_encoder_weight(p0), load_encoder_weight(p1))
            except ValueError as e:
                print(f"  [skip] {chain} L{layer}: {e}")
                continue
            d_start = _density_lookup(chain, layer, first)
            d_end = _density_lookup(chain, layer, last)
            drift = (1.0 - cos).numpy()

            if args.min_density_end > 0:
                if d_end is None:
                    print(f"    {chain} L{layer}: no density cache; "
                          f"run feature_lifecycle first or remove the filter")
                    continue
                pool = np.where(d_end.numpy() >= args.min_density_end)[0]
                if pool.size == 0:
                    print(f"    {chain} L{layer}: 0 features pass density filter")
                    continue
            else:
                pool = np.arange(drift.shape[0])

            pool_drift = drift[pool]
            pool_order = np.argsort(-pool_drift)
            for rank, pool_i in enumerate(pool_order[:args.top_n], start=1):
                feat = int(pool[pool_i])
                rows.append({
                    "chain": chain,
                    "layer": layer,
                    "feature_idx": feat,
                    "rank": rank,
                    "stage_from": first,
                    "stage_to": last,
                    "dec_cos": float(cos[feat].item()),
                    "dec_drift": float(drift[feat]),
                    "enc_cos": float(ecos[feat].item()),
                    "density_start": (
                        float(d_start[feat].item()) if d_start is not None else None
                    ),
                    "density_end": (
                        float(d_end[feat].item()) if d_end is not None else None
                    ),
                })
            top_idx = int(pool[pool_order[0]])
            print(f"    {chain} L{layer}: pool={pool.size}  "
                  f"top-1 drift={drift[top_idx]:.3f}  "
                  f"pool mean drift={pool_drift.mean():.4f}")

    out_name = (
        "top_drifted_features.csv"
        if args.min_density_end == 0
        else "top_drifted_features_active.csv"
    )
    write_csv(rows, RESULTS_DIR / out_name)


# ---------------------------------------------------------------------------
# 5. Encoder vs decoder cosine scatter (first transition)
# ---------------------------------------------------------------------------

def cmd_enc_vs_dec(args) -> None:
    """4 layers x 2 chains scatter of per-feature (decoder cos, encoder cos)
    for the first PPO transition (instruct_base -> ppo_step10) of each chain.
    Highlights cases where encoder and decoder drift decouple (e.g., L18 early).
    """
    print("[enc_vs_dec] per-feature decoder vs encoder cosine, first transition")
    chains = tuple(CHAINS.keys())
    fig, axes = plt.subplots(
        len(LAYERS), len(chains),
        figsize=(4 * len(chains), 2.6 * len(LAYERS)),
        squeeze=False, sharex=True, sharey=True,
    )
    for col, chain in enumerate(chains):
        stages = CHAINS[chain]["stages"]
        if len(stages) < 2:
            continue
        a, b = stages[0], stages[1]
        for row_i, layer in enumerate(LAYERS):
            ax = axes[row_i][col]
            p0, p1 = sae_path(chain, layer, a), sae_path(chain, layer, b)
            if not (p0.exists() and p1.exists()):
                ax.set_visible(False)
                continue
            try:
                dcos = per_feature_decoder_cosine(
                    load_decoder_weight(p0), load_decoder_weight(p1))
                ecos = per_feature_encoder_cosine(
                    load_encoder_weight(p0), load_encoder_weight(p1))
            except ValueError:
                ax.set_visible(False)
                continue
            corr = float(np.corrcoef(dcos.numpy(), ecos.numpy())[0, 1])
            ax.scatter(dcos.numpy(), ecos.numpy(),
                       s=2, alpha=0.25, color=PALETTE["twilight"])
            ax.plot([-1, 1], [-1, 1], color=PALETTE["lilac"],
                    linewidth=0.8, linestyle="--")
            ax.set_title(f"{chain} L{layer}  corr={corr:+.2f}", fontsize=10)
            if row_i == len(LAYERS) - 1:
                ax.set_xlabel("decoder cosine")
            if col == 0:
                ax.set_ylabel("encoder cosine")
    fig.suptitle("Per-feature decoder vs encoder cosine "
                 "(first PPO transition: instruct_base -> ppo_step10)")
    fig.tight_layout()
    path = FIGURES_DIR / "fig_enc_vs_dec_first_transition.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figure -> {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Pipeline (run all)
# ---------------------------------------------------------------------------

def cmd_pipeline(args) -> None:
    print("=== running full SAE analysis pipeline ===\n")
    print("--- 1/5 drift ---")
    cmd_drift(argparse.Namespace(value="dec_cos_mean"))
    print("\n--- 2/5 feature_match ---")
    cmd_feature_match(argparse.Namespace(hungarian=False))
    print("\n--- 3/5 feature_lifecycle ---")
    cmd_feature_lifecycle(argparse.Namespace(
        device=args.device, batch_size=args.batch_size, refresh=False,
    ))
    print("\n--- 4/5 feature_drift_top (unfiltered) ---")
    cmd_feature_drift_top(argparse.Namespace(
        top_n=args.top_n, min_density_end=0.0,
    ))
    if args.min_density_end > 0:
        print(f"\n--- 4b/5 feature_drift_top "
              f"(active, density_end >= {args.min_density_end:.1e}) ---")
        cmd_feature_drift_top(argparse.Namespace(
            top_n=args.top_n, min_density_end=args.min_density_end,
        ))
    print("\n--- 5/5 enc_vs_dec ---")
    cmd_enc_vs_dec(argparse.Namespace())
    print("\n=== pipeline done ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    drift = sub.add_parser("drift", help="decoder + encoder cosine heatmap")
    drift.add_argument(
        "--value", default="dec_cos_mean",
        choices=[
            "dec_cos_mean", "dec_cos_median",
            "dec_drift_per_step",
        ],
        help="cell value for the decoder heatmap (encoder figure always uses enc_cos_mean)",
    )
    drift.set_defaults(func=cmd_drift)

    fm = sub.add_parser("feature_match", help="best-match (renaming-aware) cosine")
    fm.add_argument("--hungarian", action="store_true",
                    help="use scipy 1:1 assignment (slow; default uses argmax)")
    fm.set_defaults(func=cmd_feature_match)

    fl = sub.add_parser("feature_lifecycle",
                        help="per-feature density, entropy, deaths, births")
    fl.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    fl.add_argument("--batch_size", type=int, default=1024)
    fl.add_argument("--refresh", action="store_true",
                    help="ignore cached densities and recompute")
    fl.set_defaults(func=cmd_feature_lifecycle)

    ft = sub.add_parser("feature_drift_top",
                        help="rank features per layer by total drift")
    ft.add_argument("--top_n", type=int, default=20)
    ft.add_argument("--min_density_end", type=float, default=0.0,
                    help="only rank features whose end-stage firing density "
                         ">= this value (writes top_drifted_features_active.csv); "
                         "0 = no filter (writes top_drifted_features.csv). "
                         "Requires feature_lifecycle to have run.")
    ft.set_defaults(func=cmd_feature_drift_top)

    ev = sub.add_parser("enc_vs_dec",
                        help="per-feature dec vs enc cosine scatter, first transition")
    ev.set_defaults(func=cmd_enc_vs_dec)

    pp = sub.add_parser("pipeline", help="run all analyses end-to-end")
    pp.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    pp.add_argument("--batch_size", type=int, default=1024)
    pp.add_argument("--top_n", type=int, default=20)
    pp.add_argument("--min_density_end", type=float, default=1e-3,
                    help="also produce a density-filtered top_drifted_features_active.csv "
                         "(set 0 to skip)")
    pp.set_defaults(func=cmd_pipeline)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
