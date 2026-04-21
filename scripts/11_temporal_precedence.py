#!/usr/bin/env python3
"""
11_temporal_precedence.py
Generate the temporal feature-transition figure.

The PPO run did not exhibit reward hacking: solve_rate peaked at ~10% with no
performance cliff. The model underwent a format-learning transition between steps
50–100 (response_length 445→17 tokens, format_rate 10%→99%). This script
visualises how SAE feature frequencies track that transition.

  concise_rise features: become more active in the short-output (concise) phase
  verbose_rise features: were more active in the long-output (verbose) phase
  stable features:       frequency unchanged across training

Behavioral signals tracked:
  format_rate rise, response_length collapse (from training_curves.csv)

Inputs:
  results/training_curves.csv          (from 08_extract_wandb.py)
  results/precedence_analysis/         (from 10_classify_features.py)

Outputs (results/precedence_analysis/):
  precedence_figure.png                (main figure)
  onset_summary.csv                    (onset steps per signal)
  mean_freq_by_class.csv               (mean freq per class per checkpoint)

Usage:
    python scripts/11_temporal_precedence.py
    python scripts/11_temporal_precedence.py --focus_layer 23
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


# ── Onset detection ───────────────────────────────────────────────────────────

def find_onset_step(steps: np.ndarray, values: np.ndarray,
                    baseline_steps: np.ndarray, direction: str = "rise",
                    n_sigma: float = 2.0,
                    rel_threshold: float = 0.20) -> int | None:
    """
    Find first step where `values` deviates from baseline.

    With ≥2 baseline points: threshold is mu ± n_sigma·sigma.
    With 1 baseline point (or zero-variance baseline): threshold falls back to a
    relative-change rule — rise > baseline·(1+rel_threshold), fall < baseline·(1-rel_threshold).
    This keeps the detector usable on sparse training curves where only one
    pre-transition checkpoint exists.

    direction: "rise" = value goes up, "fall" = value goes down.
    Returns the PPO step number, or None if not found.
    """
    baseline_mask = np.isin(steps, baseline_steps)
    n_baseline = int(baseline_mask.sum())
    if n_baseline == 0:
        return None

    mu = values[baseline_mask].mean()

    if n_baseline >= 2:
        sigma = values[baseline_mask].std()
    else:
        sigma = 0.0

    if sigma > 0:
        threshold = mu + n_sigma * sigma if direction == "rise" else mu - n_sigma * sigma
    else:
        # Fallback: relative-change threshold against the lone baseline point.
        # Uses absolute value of mu so the direction is respected even when mu is negative.
        delta = abs(mu) * rel_threshold
        if delta == 0:
            # Baseline is exactly zero; use rel_threshold as an absolute floor.
            delta = rel_threshold
        threshold = mu + delta if direction == "rise" else mu - delta

    for step, val in zip(steps, values):
        if step in baseline_steps:
            continue
        if direction == "rise" and val > threshold:
            return int(step)
        if direction == "fall" and val < threshold:
            return int(step)

    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_curves", default="results/training_curves.csv")
    parser.add_argument("--analysis_dir", default="results/precedence_analysis")
    parser.add_argument("--output_dir", default="results/precedence_analysis")
    parser.add_argument("--focus_layer", type=int, default=23,
                        help="Layer to use for SAE feature trajectories (default: 23)")
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18, 23],
                        help="All layers to include in supplementary panels")
    parser.add_argument("--n_sigma", type=float, default=2.0,
                        help="Sigma threshold for onset detection")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load feature frequencies and classifications ───────────────────────
    freq_by_layer = {}
    clf_by_layer  = {}

    for layer in args.layers:
        freq_csv = os.path.join(args.analysis_dir, f"feature_frequencies_layer{layer}.csv")
        clf_csv  = os.path.join(args.analysis_dir, f"feature_classification_layer{layer}.csv")

        if not os.path.exists(freq_csv) or not os.path.exists(clf_csv):
            print(f"[warn] Layer {layer}: missing files, skipping")
            continue

        freq_df = pd.read_csv(freq_csv, index_col=[0, 1])
        clf_df  = pd.read_csv(clf_csv)
        freq_by_layer[layer] = freq_df
        clf_by_layer[layer]  = clf_df
        print(f"Layer {layer}: {len(freq_df)} checkpoints, {len(clf_df)} features")

    if not freq_by_layer:
        print("[error] No frequency data found. Run 10_classify_features.py first.")
        return

    # ── Compute mean frequency per class per checkpoint (focus layer) ─────
    layer = args.focus_layer
    if layer not in freq_by_layer:
        layer = list(freq_by_layer.keys())[-1]  # fallback to last layer
        print(f"[warn] Focus layer not available, using layer {layer}")

    freq_df = freq_by_layer[layer]
    clf_df  = clf_by_layer[layer]

    feat_cols = [c for c in freq_df.columns if c.startswith("feat_")]
    feat_idx  = np.array([int(c.split("_")[1]) for c in feat_cols])

    hacking_feats   = clf_df[clf_df["class"] == "concise_rise"]["feature_idx"].values
    reasoning_feats = clf_df[clf_df["class"] == "verbose_rise"]["feature_idx"].values
    stable_feats    = clf_df[clf_df["class"] == "stable"]["feature_idx"].values

    # Map feature_idx → column position in freq_df
    idx_to_col = {idx: f"feat_{idx}" for idx in feat_idx}

    def mean_freq_for_class(feat_indices):
        cols = [idx_to_col[i] for i in feat_indices if i in idx_to_col]
        if not cols:
            return np.full(len(freq_df), np.nan)
        return freq_df[cols].mean(axis=1).values

    sae_steps          = freq_df.index.get_level_values("ppo_step").values.astype(float)
    mean_hack_freq     = mean_freq_for_class(hacking_feats)
    mean_reason_freq   = mean_freq_for_class(reasoning_feats)
    mean_stable_freq   = mean_freq_for_class(stable_feats)

    # Save mean freq per class
    mean_freq_df = pd.DataFrame({
        "ppo_step":              sae_steps,
        "stage":                 freq_df.index.get_level_values("stage").values,
        "mean_concise_rise_freq": mean_hack_freq,
        "mean_verbose_rise_freq": mean_reason_freq,
        "mean_stable_freq":       mean_stable_freq,
        "n_concise_rise_feats":   len(hacking_feats),
        "n_verbose_rise_feats":   len(reasoning_feats),
    })
    mean_freq_df.to_csv(os.path.join(args.output_dir, "mean_freq_by_class.csv"), index=False)

    # ── Load WandB training curves ─────────────────────────────────────────
    curves_available = os.path.exists(args.training_curves)
    if curves_available:
        curves = pd.read_csv(args.training_curves)
        curves = curves.sort_values("step")
        print(f"Training curves: {len(curves)} steps, columns: {list(curves.columns)}")
    else:
        print(f"[warn] Training curves not found at {args.training_curves}")
        print("       Run 08_extract_wandb.py first for behavioral metrics.")
        curves = pd.DataFrame()

    # ── Onset detection ────────────────────────────────────────────────────
    # Baseline = verbose phase (steps 0, 10, 50) before the format collapse
    baseline_steps = np.array([0, 10, 50])
    onset_results  = {}

    # SAE concise_rise feature onset
    valid = ~np.isnan(mean_hack_freq)
    if valid.sum() >= 3:
        onset = find_onset_step(
            sae_steps[valid], mean_hack_freq[valid],
            baseline_steps=baseline_steps,
            direction="rise", n_sigma=args.n_sigma,
        )
        onset_results["sae_concise_rise"] = onset
        print(f"SAE concise_rise feature onset (layer {layer}): step {onset}")

    # SAE verbose_rise feature onset (decline)
    valid = ~np.isnan(mean_reason_freq)
    if valid.sum() >= 3:
        onset = find_onset_step(
            sae_steps[valid], mean_reason_freq[valid],
            baseline_steps=baseline_steps,
            direction="fall", n_sigma=args.n_sigma,
        )
        onset_results["sae_verbose_rise_decline"] = onset
        print(f"SAE verbose_rise decline onset: step {onset}")

    # Behavioral onsets from WandB
    if not curves.empty:
        for col, direction, label in [
            ("format_rate",    "rise", "format_rate_rise"),
            ("response_length","fall", "response_length_collapse"),
            ("solve_rate",     "rise", "solve_rate_rise"),
            ("kl_div",         "rise", "kl_divergence_rise"),
        ]:
            if col not in curves.columns:
                continue
            col_vals = curves[col].dropna()
            col_steps = curves.loc[col_vals.index, "step"].values
            # Verbose-phase boundary: step 50 is last pre-collapse checkpoint
            bl = col_steps[col_steps <= 50]
            onset = find_onset_step(
                col_steps, col_vals.values,
                baseline_steps=bl,
                direction=direction, n_sigma=args.n_sigma,
            )
            onset_results[label] = onset
            print(f"{label}: step {onset}")

    # Save onset summary
    onset_df = pd.DataFrame([
        {"signal": k, "onset_step": v}
        for k, v in onset_results.items()
    ])
    onset_df.to_csv(os.path.join(args.output_dir, "onset_summary.csv"), index=False)
    print(f"\nOnset summary:")
    print(onset_df.to_string(index=False))

    # ── Figure ────────────────────────────────────────────────────────────
    n_behavior_panels = sum(
        1 for col in ["kl_div", "solve_rate", "response_length"]
        if not curves.empty and col in curves.columns
    )
    n_rows = 2 + (1 if n_behavior_panels > 0 else 0)

    fig = plt.figure(figsize=(12, 4 * n_rows))
    gs  = gridspec.GridSpec(n_rows, 1, hspace=0.45)

    COLORS = {
        "concise_rise": "#e63946",
        "verbose_rise": "#2a9d8f",
        "stable":       "#adb5bd",
    }
    ONSET_COLORS = {
        "sae_concise_rise":         "#e63946",
        "sae_verbose_rise_decline": "#2a9d8f",
        "format_rate_rise":         "#f4a261",
        "response_length_collapse": "#8338ec",
        "solve_rate_rise":          "#264653",
        "kl_divergence_rise":       "#fb8500",
    }

    # ── Panel 1: SAE feature frequency by class ────────────────────────────
    ax1 = fig.add_subplot(gs[0])

    ax1.plot(sae_steps, mean_hack_freq,   color=COLORS["concise_rise"],
             marker="o", markersize=5, linewidth=2, label=f"Concise-rise features (n={len(hacking_feats)})")
    ax1.plot(sae_steps, mean_reason_freq, color=COLORS["verbose_rise"],
             marker="s", markersize=5, linewidth=2, label=f"Verbose-rise features (n={len(reasoning_feats)})")
    ax1.plot(sae_steps, mean_stable_freq, color=COLORS["stable"],
             marker=".", markersize=4, linewidth=1, alpha=0.6, label=f"Stable features (n={len(stable_feats)})")

    # Onset vertical lines for SAE signals
    for signal in ["sae_concise_rise", "sae_verbose_rise_decline"]:
        step = onset_results.get(signal)
        if step is not None:
            ax1.axvline(step, color=ONSET_COLORS[signal], linestyle="--", linewidth=1.5, alpha=0.8)
            ax1.text(step + 2, ax1.get_ylim()[1] * 0.95, f"SAE onset\n(step {step})",
                     color=ONSET_COLORS[signal], fontsize=8, va="top")

    ax1.axvspan(0, 50,  alpha=0.06, color="green", label="Verbose phase (0-50)")
    ax1.axvspan(100, 435, alpha=0.06, color="blue", label="Concise phase (100-435)")
    ax1.set_xlabel("PPO Step")
    ax1.set_ylabel("Mean Activation Frequency")
    ax1.set_title(f"SAE Feature Frequency by Class — Layer {layer}", fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: All-layer concise_rise feature frequency comparison ──────
    ax2 = fig.add_subplot(gs[1])
    for lay in args.layers:
        if lay not in freq_by_layer:
            continue
        f_df = freq_by_layer[lay]
        c_df = clf_by_layer[lay]
        h_feats = c_df[c_df["class"] == "concise_rise"]["feature_idx"].values
        cols    = [f"feat_{i}" for i in h_feats if f"feat_{i}" in f_df.columns]
        if not cols:
            continue
        steps_l = f_df.index.get_level_values("ppo_step").values.astype(float)
        mean_f  = f_df[cols].mean(axis=1).values
        ax2.plot(steps_l, mean_f, marker="o", markersize=4, linewidth=2,
                 label=f"Layer {lay} ({len(h_feats)} concise-rise feats)")

    ax2.axvspan(0, 50,   alpha=0.06, color="green")
    ax2.axvspan(100, 435, alpha=0.06, color="blue")
    ax2.set_xlabel("PPO Step")
    ax2.set_ylabel("Mean Concise-Rise Feature Frequency")
    ax2.set_title("Concise-Rise Feature Frequency — All Layers", fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Behavioral metrics from WandB ────────────────────────────
    if n_behavior_panels > 0:
        ax3 = fig.add_subplot(gs[2])
        ax3b = ax3.twinx()

        plotted_left  = False
        plotted_right = False

        for col, ax_target, color, label, side in [
            ("solve_rate",      ax3,  "#264653", "Solve Rate",      "left"),
            ("format_rate",     ax3,  "#2a9d8f", "Format Rate",     "left"),
            ("response_length", ax3b, "#8338ec", "Response Length", "right"),
            ("kl_div",          ax3b, "#f4a261", "KL Divergence",   "right"),
        ]:
            if curves.empty or col not in curves.columns:
                continue
            data = curves[["step", col]].dropna()
            vals = data[col].values.astype(float)
            # Light smoothing for dense WandB data
            if len(vals) > 20:
                vals = uniform_filter1d(vals, size=5)
            ax_target.plot(data["step"].values, vals, color=color, linewidth=1.5,
                           alpha=0.85, label=label)
            if side == "left":
                plotted_left = True
            else:
                plotted_right = True

        # Onset lines for behavioral signals
        for signal in ["format_rate_rise", "response_length_collapse", "solve_rate_rise"]:
            step = onset_results.get(signal)
            if step is not None:
                ax3.axvline(step, color=ONSET_COLORS[signal], linestyle=":", linewidth=1.5, alpha=0.85)
                ax3.text(step + 2, 0.02, f"{signal.replace('_', ' ')}\n(step {step})",
                         color=ONSET_COLORS[signal], fontsize=7, transform=ax3.get_xaxis_transform(),
                         va="bottom")

        ax3.axvspan(0, 50,   alpha=0.06, color="green")
        ax3.axvspan(100, 435, alpha=0.06, color="blue")
        ax3.set_xlabel("PPO Step")
        if plotted_left:
            ax3.set_ylabel("Solve Rate", color="#264653")
        if plotted_right:
            ax3b.set_ylabel("KL / Response Length", color="#8338ec")
        ax3.set_title("Behavioral Metrics (WandB)", fontweight="bold")

        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3b.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")
        ax3.grid(True, alpha=0.3)

    # ── Transition annotation ──────────────────────────────────────────────
    sae_onset  = onset_results.get("sae_concise_rise")
    behav_onsets = [
        onset_results.get("format_rate_rise"),
        onset_results.get("response_length_collapse"),
    ]
    behav_onsets = [s for s in behav_onsets if s is not None]

    if sae_onset is not None and behav_onsets:
        earliest_behav = min(behav_onsets)
        lead_steps = earliest_behav - sae_onset
        fig.suptitle(
            f"Format-Learning Transition — SAE Feature Frequencies vs Behavioral Metrics\n"
            f"Layer {layer}  |  SAE concise-rise onset: step {sae_onset}  |  "
            f"Earliest behavioral transition: step {earliest_behav}  |  "
            f"Lead: {lead_steps:+d} steps",
            fontsize=11, fontweight="bold",
        )
    else:
        fig.suptitle(
            f"Format-Learning Transition — SAE Feature Frequencies vs Behavioral Metrics\n"
            f"Layer {layer}",
            fontsize=11, fontweight="bold",
        )

    out_fig = os.path.join(args.output_dir, "precedence_figure.png")
    plt.savefig(out_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved → {out_fig}")

    if sae_onset is not None and behav_onsets:
        earliest_behav = min(behav_onsets)
        lead_steps = earliest_behav - sae_onset
        print(f"\nKey result: SAE concise-rise features emerge at step {sae_onset}, "
              f"behavioral format transition at step {earliest_behav} → "
              f"{abs(lead_steps)}-step {'lead' if lead_steps > 0 else 'lag'}")


if __name__ == "__main__":
    main()
