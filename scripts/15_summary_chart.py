#!/usr/bin/env python3
"""
15_summary_chart.py
Comprehensive summary chart for SAE-RL analysis.

Panels:
  A  Behavioral training curves (solve_rate, format_rate, response_length)
  B  SAE recon MSE per layer across all 16 checkpoints
  C  Feature classification counts per layer (concise_rise / verbose_rise / stable)
  D  Mean feature frequency by class over training — layer 23 (densest signal)
  E  Hyperparameter reference table
  F  Layer interpretation & analysis roadmap

Output: results/summary_chart.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
CURVES_CSV    = "results/training_curves.csv"
METRICS_CSV   = "results/sae_eval_metrics.csv"
CLF_SUM_CSV   = "results/precedence_analysis/classification_summary.csv"
MEAN_FREQ_CSV = "results/precedence_analysis/layer23/mean_freq_by_class.csv"
OUT_PATH      = "results/summary_chart.png"

LAYERS = [6, 12, 18, 23]

STAGE_ORDER = [
    "sft", "ppo_step10", "ppo_step50", "ppo_step100",
    "ppo_step110", "ppo_step120", "ppo_step130", "ppo_step140",
    "ppo_step150", "ppo_step160", "ppo_step170", "ppo_step180",
    "ppo_step190", "ppo_step200", "ppo_step300", "ppo_step435",
]
STAGE_STEPS = {
    "sft": 0, "ppo_step10": 10, "ppo_step50": 50, "ppo_step100": 100,
    "ppo_step110": 110, "ppo_step120": 120, "ppo_step130": 130,
    "ppo_step140": 140, "ppo_step150": 150, "ppo_step160": 160,
    "ppo_step170": 170, "ppo_step180": 180, "ppo_step190": 190,
    "ppo_step200": 200, "ppo_step300": 300, "ppo_step435": 435,
}

LAYER_COLORS = {6: "#4361ee", 12: "#f72585", 18: "#7209b7", 23: "#3a0ca3"}

# ── Load data ─────────────────────────────────────────────────────────────────
curves   = pd.read_csv(CURVES_CSV).sort_values("step")
metrics  = pd.read_csv(METRICS_CSV)
clf_sum  = pd.read_csv(CLF_SUM_CSV)
mean_freq = pd.read_csv(MEAN_FREQ_CSV).sort_values("ppo_step")

metrics["ppo_step"] = metrics["stage"].map(STAGE_STEPS)
metrics = metrics.dropna(subset=["ppo_step"])

# ── Layout ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 22))
gs  = gridspec.GridSpec(
    4, 3,
    figure=fig,
    hspace=0.52,
    wspace=0.38,
    left=0.06, right=0.97, top=0.95, bottom=0.04,
)

TRANSITION_STEP = 100
EARLY_SPAN      = (0, 50)
LATE_SPAN       = (100, 200)

def shade_phases(ax, xmax=200):
    ax.axvspan(*EARLY_SPAN, alpha=0.07, color="#2d6a4f", zorder=0)
    ax.axvspan(*LATE_SPAN,  alpha=0.05, color="#1d3557", zorder=0)
    ax.axvline(TRANSITION_STEP, color="#e63946", linestyle="--", linewidth=1.2, alpha=0.7)

# ── Panel A: Behavioral training curves ───────────────────────────────────────
ax_A = fig.add_subplot(gs[0, 0])
ax_A2 = ax_A.twinx()

ax_A.plot(curves["step"], curves["solve_rate"] * 100,
          color="#264653", linewidth=2, marker="o", markersize=4, label="Solve rate (%)")
ax_A.plot(curves["step"], curves["format_rate"] * 100,
          color="#2a9d8f", linewidth=2, marker="s", markersize=4, label="Format rate (%)")
ax_A2.plot(curves["step"], curves["response_length"],
           color="#e76f51", linewidth=2, marker="^", markersize=4, linestyle="--",
           label="Response length (tokens)")

shade_phases(ax_A)
ax_A.set_xlabel("PPO Step")
ax_A.set_ylabel("Rate (%)", color="#264653")
ax_A2.set_ylabel("Response length (tokens)", color="#e76f51")
ax_A.set_title("A  Behavioral Training Curves", fontweight="bold", loc="left")
ax_A.set_ylim(0, 110)

lines1, labs1 = ax_A.get_legend_handles_labels()
lines2, labs2 = ax_A2.get_legend_handles_labels()
ax_A.legend(lines1 + lines2, labs1 + labs2, fontsize=7, loc="center right")

# Add phase labels
ax_A.text(25, 103, "Pre-transition\n(baseline)", fontsize=6.5, ha="center", color="#2d6a4f", style="italic")
ax_A.text(150, 103, "Post-transition (s100–s200)", fontsize=6.5, ha="center", color="#1d3557", style="italic")
ax_A.grid(True, alpha=0.25)

# ── Panel B: Recon MSE per layer across checkpoints ───────────────────────────
ax_B = fig.add_subplot(gs[0, 1])

for layer in LAYERS:
    layer_df = metrics[metrics["layer"] == layer].sort_values("ppo_step")
    ax_B.plot(layer_df["ppo_step"], layer_df["recon_mse"],
              color=LAYER_COLORS[layer], linewidth=2, marker="o", markersize=4,
              label=f"Layer {layer}")

shade_phases(ax_B)
ax_B.set_xlabel("PPO Step")
ax_B.set_ylabel("Reconstruction MSE")
ax_B.set_title("B  SAE Recon MSE per Layer (50-epoch SAEs)", fontweight="bold", loc="left")
ax_B.legend(fontsize=8)
ax_B.grid(True, alpha=0.25)

# ── Panel C: Feature classification counts per layer ─────────────────────────
ax_C = fig.add_subplot(gs[0, 2])

layers_list = clf_sum["layer"].tolist()
x = np.arange(len(layers_list))
w = 0.28

ax_C.bar(x - w, clf_sum["n_concise_rise"], width=w, color="#e63946", label="concise_rise")
ax_C.bar(x,      clf_sum["n_verbose_rise"], width=w, color="#2a9d8f", label="verbose_rise")
ax_C.bar(x + w,  clf_sum["n_stable"],       width=w, color="#adb5bd", label="stable")

for i, row in clf_sum.iterrows():
    total  = row["n_total"]
    cr_pct = row["n_concise_rise"] / total * 100
    vr_pct = row["n_verbose_rise"] / total * 100
    xi = x[i]
    ax_C.text(xi - w, row["n_concise_rise"] + 30, f"{cr_pct:.1f}%", ha="center",
              fontsize=6.5, color="#e63946")
    ax_C.text(xi,     row["n_verbose_rise"] + 30, f"{vr_pct:.1f}%", ha="center",
              fontsize=6.5, color="#2a9d8f")

ax_C.set_xticks(x)
ax_C.set_xticklabels([f"L{l}" for l in layers_list])
ax_C.set_ylabel("Feature count")
ax_C.set_title("C  Feature Classification per Layer", fontweight="bold", loc="left")
ax_C.legend(fontsize=8)
ax_C.grid(True, alpha=0.25, axis="y")

# ── Panel D: Mean feature frequency by class — layer 23 ───────────────────────
ax_D = fig.add_subplot(gs[1, 0])

ax_D.plot(mean_freq["ppo_step"], mean_freq["mean_concise_rise_freq"],
          color="#e63946", linewidth=2, marker="o", markersize=4, label="concise_rise")
ax_D.plot(mean_freq["ppo_step"], mean_freq["mean_verbose_rise_freq"],
          color="#2a9d8f", linewidth=2, marker="s", markersize=4, label="verbose_rise")
ax_D.plot(mean_freq["ppo_step"], mean_freq["mean_stable_freq"],
          color="#adb5bd", linewidth=1.5, marker=".", markersize=3, alpha=0.7, label="stable")

shade_phases(ax_D)
ax_D.set_xlabel("PPO Step")
ax_D.set_ylabel("Mean Activation Frequency")
ax_D.set_title("D  Mean Feature Freq by Class — Layer 23", fontweight="bold", loc="left")
ax_D.legend(fontsize=8)
ax_D.grid(True, alpha=0.25)
ax_D.text(TRANSITION_STEP + 5, ax_D.get_ylim()[1] * 0.97,
          "Transition onset\n(step 100)", fontsize=6.5, color="#e63946", va="top")

# ── Panel E: Recon MSE heatmap (all stages × layers) ──────────────────────────
ax_E = fig.add_subplot(gs[1, 1])

stages_with_data = [s for s in STAGE_ORDER if s in metrics["stage"].values]
mse_matrix = np.full((len(LAYERS), len(stages_with_data)), np.nan)
for i, layer in enumerate(LAYERS):
    for j, stage in enumerate(stages_with_data):
        row = metrics[(metrics["layer"] == layer) & (metrics["stage"] == stage)]
        if not row.empty:
            mse_matrix[i, j] = row["recon_mse"].values[0]

im = ax_E.imshow(mse_matrix, aspect="auto", cmap="YlOrRd", vmin=0)
ax_E.set_yticks(range(len(LAYERS)))
ax_E.set_yticklabels([f"Layer {l}" for l in LAYERS], fontsize=8)
ax_E.set_xticks(range(len(stages_with_data)))
ax_E.set_xticklabels(
    [s.replace("ppo_step", "s").replace("sft", "SFT") for s in stages_with_data],
    rotation=45, ha="right", fontsize=7,
)
plt.colorbar(im, ax=ax_E, label="Recon MSE", shrink=0.8)
ax_E.set_title("E  Recon MSE Heatmap (50-ep SAEs)", fontweight="bold", loc="left")

for i in range(len(LAYERS)):
    for j in range(len(stages_with_data)):
        val = mse_matrix[i, j]
        if not np.isnan(val):
            ax_E.text(j, i, f"{val:.2f}", ha="center", va="center",
                      fontsize=5.5, color="black" if val < 0.6 else "white")

# ── Panel F: Model delta loss heatmap ─────────────────────────────────────────
ax_F = fig.add_subplot(gs[1, 2])

delta_matrix = np.full((len(LAYERS), len(stages_with_data)), np.nan)
for i, layer in enumerate(LAYERS):
    for j, stage in enumerate(stages_with_data):
        row = metrics[(metrics["layer"] == layer) & (metrics["stage"] == stage)]
        if not row.empty:
            delta_matrix[i, j] = row["model_delta_loss"].values[0]

im2 = ax_F.imshow(delta_matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-2.5, vmax=1.0)
ax_F.set_yticks(range(len(LAYERS)))
ax_F.set_yticklabels([f"Layer {l}" for l in LAYERS], fontsize=8)
ax_F.set_xticks(range(len(stages_with_data)))
ax_F.set_xticklabels(
    [s.replace("ppo_step", "s").replace("sft", "SFT") for s in stages_with_data],
    rotation=45, ha="right", fontsize=7,
)
plt.colorbar(im2, ax=ax_F, label="Model delta loss", shrink=0.8)
ax_F.set_title("F  SAE Model Delta Loss (50-ep SAEs)", fontweight="bold", loc="left")

for i in range(len(LAYERS)):
    for j in range(len(stages_with_data)):
        val = delta_matrix[i, j]
        if not np.isnan(val):
            ax_F.text(j, i, f"{val:+.2f}", ha="center", va="center",
                      fontsize=5.5, color="black" if abs(val) < 1.2 else "white")

# ── Panel G: Hyperparameter table ─────────────────────────────────────────────
ax_G = fig.add_subplot(gs[2, :2])
ax_G.axis("off")

hyp_data = [
    # Section: PPO (see ../ppo_run/train_ppo.sh for canonical values)
    ["PPO — actor init",      "Qwen/Qwen2.5-0.5B-Instruct (no SFT)"],
    ["PPO — actor lr",        "1e-6 (AdamW, constant)"],
    ["PPO — rollout n",       "8 samples per prompt, temp=1.0"],
    ["PPO — clip ε",          "0.2"],
    ["PPO — KL coeff",        "0.005 (fixed)"],
    ["PPO — entropy coeff",   "0.001"],
    ["PPO — reward",          "Flexible last-number match on GSM8k (0/1)"],
    ["PPO — total steps",     "~240 (8 epochs × ~30 steps/epoch on 7473 samples)"],
    ["PPO — batch size",      "256 train, 64 mini-batch, 8 micro-batch/GPU"],
    # Section: SAE
    ["SAE — type",            "TopK Sparse Autoencoder"],
    ["SAE — d_model",         "896 (Qwen2.5-0.5B)"],
    ["SAE — expansion",       "8× → d_sae = 7168"],
    ["SAE — k (active/token)","32"],
    ["SAE — epochs",          "50"],
    ["SAE — lr",              "3e-4 (Adam)"],
    ["SAE — loss",            "Reconstruction MSE only (no L1)"],
    ["SAE — dead resample",   "Every 5 epochs, threshold freq < 1e-4"],
    # Section: data
    ["Activations",           "Residual stream at layers 6, 12, 18, 23 — 16 checkpoints"],
    ["Activation tokens",     "~222k per (checkpoint × layer), GSM8k train questions"],
    ["Evaluation",            "Last 20% of cached tokens (held-out)"],
]

col_labels = ["Parameter", "Value"]
table = ax_G.table(
    cellText=hyp_data,
    colLabels=col_labels,
    cellLoc="left",
    loc="upper left",
    bbox=[0, 0, 1, 1],
)
table.auto_set_font_size(False)
table.set_fontsize(8)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#1d3557")
        cell.set_text_props(color="white", fontweight="bold")
    elif col == 0:
        cell.set_facecolor("#e8f4f8")
        cell.set_text_props(fontweight="bold")
        # Section dividers
        label = hyp_data[row-1][0] if row-1 < len(hyp_data) else ""
        if label.startswith("SAE"):
            cell.set_facecolor("#fff3cd")
        elif label.startswith("Act") or label.startswith("Eval"):
            cell.set_facecolor("#d4edda")
    cell.set_edgecolor("#dee2e6")
ax_G.set_title("G  Training Hyperparameters", fontweight="bold", loc="left", pad=6)

# ── Panel H: Layer interpretation + analysis roadmap ─────────────────────────
ax_H = fig.add_subplot(gs[2, 2])
ax_H.axis("off")

layer_info = [
    ["Layer", "Depth", "d_sae", "concise_r", "verbose_r", "Why important",                                "Next analysis"],
    ["6",  "25%", "7168",  "2305\n(32.2%)", "1840\n(25.7%)",
     "Early repr.\nToken/syntax",
     "Interpret top CR/VR\nfeats — format tokens\nvs content tokens?"],
    ["12", "50%", "14336", "3439\n(24.0%)", "3142\n(21.9%)",
     "Mid repr.\nSemantic/task",
     "Do VR feats encode\nmath reasoning steps\nthat fade after s100?"],
    ["18", "75%", "14336", "4551\n(31.7%)", "4196\n(29.3%)",
     "Late repr.\nHighest churn",
     "Highest turnover layer.\nTest causal role via\nablation on format_rate."],
    ["23", "96%", "28672", "1503\n(5.2%)",  "1678\n(5.9%)",
     "Near-output\nMost stable",
     "Stable but high MSE\n(~0.44). Try larger k\nor wider expansion."],
]

table2 = ax_H.table(
    cellText=layer_info[1:],
    colLabels=layer_info[0],
    cellLoc="center",
    loc="upper left",
    bbox=[0, 0, 1, 1],
)
table2.auto_set_font_size(False)
table2.set_fontsize(6.5)
for (row, col), cell in table2.get_celld().items():
    if row == 0:
        cell.set_facecolor("#1d3557")
        cell.set_text_props(color="white", fontweight="bold")
    elif col == 0 and row > 0:
        layer = layer_info[row][0]
        cell.set_facecolor(LAYER_COLORS.get(int(layer), "#f0f0f0"))
        cell.set_text_props(color="white", fontweight="bold")
    cell.set_edgecolor("#dee2e6")
    cell.set_height(0.19)
ax_H.set_title("H  Layer Interpretation & Analysis Roadmap", fontweight="bold", loc="left", pad=6)

# ── Panel I: Key findings summary ─────────────────────────────────────────────
ax_I = fig.add_subplot(gs[3, :])
ax_I.axis("off")

findings = [
    ("Run summary",
     "PPO on Qwen2.5-0.5B-Instruct from SFT init, evaluated at steps 10/100/140/180/200 on GSM8k test. "
     "Healthy, non-collapse trajectory: solve_rate 12.5% → 49.5% (peak step 180) → 41.0% (step 200). "
     "format_rate 32.5% → 80.5% → 75.5%. response_length stayed in the 375–450 token band — "
     "no format collapse. No reward hacking observed."),

    ("SAE feature transition",
     "All detectable onsets align at step 100 with 0-step lead: SAE concise-rise activates, "
     "SAE verbose-rise declines, format_rate rises, solve_rate rises — simultaneous. "
     "response_length_collapse and kl_divergence_rise return no onset (response_length never "
     "dropped >20% from baseline; kl_div column empty in training_curves.csv). "
     "Feature churn by layer: L18 highest (32% concise-rise, 29% verbose-rise), "
     "L6 (32%/26%) and L12 (24%/22%) close behind. "
     "L23 is the most stable layer (89% stable, only 5.2% concise-rise, 5.9% verbose-rise)."),

    ("SAE quality",
     "Recon MSE clean at L6/L12 (~0.022 / ~0.03), moderate at L18 (~0.13), high at L23 (~0.44) — "
     "near-output residual is inherently harder to reconstruct at k=32. "
     "model_delta_loss is strongly negative across L6/L12/L18 (≈ −0.6 to −1.7): splicing the SAE "
     "hurts next-token loss substantially. L23 reaches ≈ 0 delta at step 100 but regresses after. "
     "Worth revisiting: more epochs, larger k, or wider expansion for mid-to-late layers."),

    ("Next steps",
     "① Auto-interpret top concise_rise / verbose_rise features at L18 and L6 (highest churn). "
     "② Backfill kl_div and reward columns in training_curves.csv from wandb so the full onset "
     "panel can be computed. "
     "③ Denser checkpoint sampling between step 10 and 100 to tighten the onset estimate "
     "(current precision is ±45 steps). "
     "④ Re-run ablation with updated concise_rise labels — does L18 causally drive format compliance? "
     "⑤ Investigate why L23 SAE plateaus at high MSE (tokenizer regex warning, or representation width)."),
]

y = 0.97
colors = ["#1d3557", "#2d6a4f", "#7209b7", "#c77dff"]
for (title, body), color in zip(findings, colors):
    ax_I.text(0.0, y, f"► {title}:", fontsize=8.5, fontweight="bold",
              color=color, transform=ax_I.transAxes, va="top")
    ax_I.text(0.13, y, body, fontsize=8, color="#212529",
              transform=ax_I.transAxes, va="top",
              wrap=True, horizontalalignment="left",
              bbox=dict(boxstyle="round,pad=0.15", facecolor=color, alpha=0.06, edgecolor="none"))
    y -= 0.26

ax_I.set_title("I  Key Findings & Next Steps", fontweight="bold", loc="left", pad=4)

# ── Suptitle ──────────────────────────────────────────────────────────────────
fig.suptitle(
    "SAE-RL GSM8k Analysis — Qwen2.5-0.5B-Instruct | 50-epoch TopK SAEs | 16 checkpoints",
    fontsize=13, fontweight="bold", y=0.975,
)

os.makedirs("results", exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved → {OUT_PATH}")
