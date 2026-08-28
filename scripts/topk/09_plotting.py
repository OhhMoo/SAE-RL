"""
Step 9: Unified paper-figure pipeline.

Reads CSVs already produced by upstream pipelines (06 drift / 07d ablation /
08 contexts / 08c cross-chain identity) and emits the three paper figures
with shared style:

    Axis 1 - statistical reorganization (REWARD-INVARIANT)
        figures/fig1_drift_heatmap.png            decoder cosine, layer x PPO transition
        figures/fig1b_encoder_drift_heatmap.png   encoder cosine, layer x PPO transition
        Source: results/decoder_drift.csv (06).

    Axis 2 - causal coverage (REWARD-CONTENT-GRADED)
        figures/fig2_coverage_gradient.png        load-bearing K=50 split by firing breadth
        Source: results/l18_topk_summary.csv (08, derived from 07d).

    Axis 3 - feature identity (PARTIAL-SHARED, MAGNITUDE-GRADED)
        figures/fig3_cross_chain_identity.png     fid 1622 cross-chain + chain-specific named features
        Sources: results/named_feature_summary.csv (08b),
                 results/flex_named_feature_summary.csv (08c).

Usage:
    python scripts/09_plotting.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO_ROOT / "results"
FIGURES = REPO_ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Shared style — palette matches scripts/06_analysis_sae.py
# ---------------------------------------------------------------------------

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
CHAIN_COLOR = {
    "strict":   PALETTE["twilight"],
    "flexible": PALETTE["lilac"],
    "shuffled": PALETTE["rose"],
}
BAND_COLOR = {
    "broad":  PALETTE["twilight"],
    "sparse": PALETTE["lilac"],
    "ghost":  PALETTE["rose"],
}
CHAIN_ORDER = ["strict", "flexible", "shuffled"]
LAYERS = [6, 12, 18, 23]

mpl.rcParams.update({
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "figure.dpi":       110,
    "font.size":        10,
    "axes.titlesize":   12,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
})


def _style_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _box_legend(leg):
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(0.6)


def _stage_step(stage: str) -> int:
    if stage == "instruct_base":
        return 0
    return int(stage.removeprefix("ppo_step"))


# ---------------------------------------------------------------------------
# Axis 1 - drift heatmaps (decoder + encoder cosine)
# ---------------------------------------------------------------------------

def _drift_grid(df: pd.DataFrame, chain: str, value: str):
    sub = df[df["chain"] == chain].sort_values(["step_from", "layer"])
    transitions: list[tuple[str, str]] = []
    for _, r in sub.iterrows():
        t = (r["stage_from"], r["stage_to"])
        if t not in transitions:
            transitions.append(t)
    grid = np.full((len(LAYERS), len(transitions)), np.nan)
    for _, r in sub.iterrows():
        i = LAYERS.index(int(r["layer"]))
        j = transitions.index((r["stage_from"], r["stage_to"]))
        grid[i, j] = float(r[value])
    return grid, transitions


def plot_drift_heatmap(df: pd.DataFrame, value: str, title: str, out: Path):
    chains = [c for c in CHAIN_ORDER if c in set(df["chain"])]
    grids = {c: _drift_grid(df, c, value) for c in chains}
    widths = [max(grids[c][0].shape[1], 1) for c in chains]
    finite = df[value][np.isfinite(df[value])]
    vmin, vmax = float(finite.min()), float(finite.max())

    fig, axes = plt.subplots(
        1, len(chains),
        figsize=(2 + 0.7 * sum(widths), 0.6 * len(LAYERS) + 2),
        gridspec_kw={"width_ratios": widths},
        squeeze=False,
    )
    axes = axes[0]
    im = None
    # White text only for cells in the darkest ~20% of the value range; the
    # cmap (twilight -> lilac -> dusty -> rose -> blush) goes from dark navy
    # to very light pink, and any cell at >=20% of range sits on a medium
    # pink that takes black text better.
    text_threshold = vmin + 0.20 * (vmax - vmin)
    for ax, chain in zip(axes, chains):
        grid, transitions = grids[chain]
        im = ax.imshow(grid, aspect="auto", cmap=CMAP, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(transitions)))
        ax.set_xticklabels(
            [f"{_stage_step(a)}->{_stage_step(b)}" for a, b in transitions],
            rotation=45, ha="right", fontsize=8,
        )
        ax.set_yticks(range(len(LAYERS)))
        ax.set_yticklabels([f"L{l}" for l in LAYERS])
        ax.set_title(f"{chain}  (n={grid.shape[1]})")
        ax.set_xlabel("PPO transition")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}",
                            ha="center", va="center", fontsize=7,
                            color="white" if v < text_threshold else "black")
    axes[0].set_ylabel("layer")
    fig.suptitle(title)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Axis 1 - L18 encoder/decoder decoupling probe (cold-start transition)
# ---------------------------------------------------------------------------

DEAD_THRESHOLD = 1e-4
COLD_STAGE_A = "instruct_base"
COLD_STAGE_B = "ppo_step10"
QUADS = ["stable", "reweighted", "redirected", "rotated"]


def _load_density_cache(chain: str, stage: str):
    """Read the per-feature density cache produced upstream (.pt -> ndarray)."""
    import torch  # local import: only needed for fig1c
    p = RESULTS / "cache" / f"density_{chain}_L18_{stage}.pt"
    return torch.load(p, map_location="cpu",
                      weights_only=False)["density"].numpy()


def _decoupling_quadrants(dcos: np.ndarray, ecos: np.ndarray):
    dec_med = float(np.median(dcos))
    enc_med = float(np.median(ecos))
    quad = np.empty(len(dcos), dtype=object)
    quad[(dcos >= dec_med) & (ecos >= enc_med)] = "stable"
    quad[(dcos < dec_med) & (ecos < enc_med)] = "rotated"
    quad[(dcos >= dec_med) & (ecos < enc_med)] = "reweighted"
    quad[(dcos < dec_med) & (ecos >= enc_med)] = "redirected"
    return quad, dec_med, enc_med


def _lifecycle_status(d_a: np.ndarray, d_b: np.ndarray):
    alive_a = d_a > DEAD_THRESHOLD
    alive_b = d_b > DEAD_THRESHOLD
    status = np.empty(len(d_a), dtype=object)
    status[alive_a & alive_b]   = "stayed_alive"
    status[~alive_a & ~alive_b] = "stayed_dead"
    status[~alive_a & alive_b]  = "born"
    status[alive_a & ~alive_b]  = "died"
    return status


def plot_l18_decoupling(out: Path):
    """Top row: per-feature (dec_cos, enc_cos) scatter, colored by lifecycle.
    Bottom row: born-feature concentration per quadrant. One column per chain.

    Reads dec_cos / enc_cos from results/l18_attribution.csv (already cold-
    start: instruct_base -> ppo_step10) and density caches from
    results/cache/density_{chain}_L18_{stage}.pt for lifecycle status.
    """
    attr = pd.read_csv(RESULTS / "l18_attribution.csv")

    fig, axes = plt.subplots(2, len(CHAIN_ORDER),
                             figsize=(6.0 * len(CHAIN_ORDER), 10.0),
                             squeeze=False)

    for col, chain in enumerate(CHAIN_ORDER):
        sub = attr[attr["chain"] == chain].sort_values("feature_idx")
        dcos = sub["dec_cos"].to_numpy()
        ecos = sub["enc_cos"].to_numpy()
        d_a = _load_density_cache(chain, COLD_STAGE_A)
        d_b = _load_density_cache(chain, COLD_STAGE_B)
        status = _lifecycle_status(d_a, d_b)
        quad, dec_med, enc_med = _decoupling_quadrants(dcos, ecos)
        corr = float(np.corrcoef(dcos, ecos)[0, 1])

        # Top row: lifecycle-colored scatter
        ax = axes[0, col]
        for st, color, label in (
            ("stayed_alive", PALETTE["dusty"],    "stayed alive"),
            ("stayed_dead",  PALETTE["rose"],     "stayed dead"),
            ("born",         PALETTE["lilac"],    "born"),
            ("died",         PALETTE["twilight"], "died"),
        ):
            m = status == st
            if m.any():
                ax.scatter(dcos[m], ecos[m], s=6, alpha=0.55,
                           color=color, label=f"{label} (n={int(m.sum())})")
        ax.axvline(dec_med, color="grey", linestyle="--", linewidth=0.6)
        ax.axhline(enc_med, color="grey", linestyle="--", linewidth=0.6)
        ax.set_title(f"{chain} L18 cold-start  corr={corr:+.2f}")
        ax.set_xlabel("decoder cosine")
        if col == 0:
            ax.set_ylabel("encoder cosine")
        # Common range across panels so cross-chain comparison is fair.
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0.5, 1.0)
        _box_legend(ax.legend(fontsize=7, loc="lower left"))
        ax.grid(alpha=0.2)

        # Bottom row: born fraction per quadrant
        ax = axes[1, col]
        quad_n = {q: int((quad == q).sum()) for q in QUADS}
        quad_born_frac = {
            q: float(((quad == q) & (status == "born")).sum()
                     / max(quad_n[q], 1))
            for q in QUADS
        }
        overall_born_frac = float((status == "born").sum() / len(status))
        x = np.arange(len(QUADS))
        ax.bar(x, [quad_born_frac[q] for q in QUADS],
               color=PALETTE["lilac"], edgecolor=PALETTE["twilight"])
        ax.axhline(overall_born_frac, color=PALETTE["twilight"],
                   linestyle="--", linewidth=1.2,
                   label=f"overall born frac = {overall_born_frac:.3f}")
        # Per-bar count: sit JUST INSIDE the top of the bar so it doesn't
        # collide with the chart frame above. For very short bars, fall
        # back to placing inside near the bottom in dimgrey.
        max_h = max(quad_born_frac.values())
        for i, q in enumerate(QUADS):
            h = quad_born_frac[q]
            if h >= 0.04 * max(max_h, 0.01):
                ax.text(i, h - 0.01 * max_h, f"n={quad_n[q]}",
                        ha="center", va="top",
                        color=PALETTE["twilight"], fontsize=8.5)
            else:
                ax.text(i, 0.01 * max_h, f"n={quad_n[q]}",
                        ha="center", va="bottom",
                        color="dimgrey", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(QUADS, rotation=15, ha="right")
        if col == 0:
            ax.set_ylabel("fraction of features 'born' in quadrant")
        ax.set_title(f"{chain} L18: born concentration by quadrant")
        ax.set_ylim(0, max_h * 1.18)
        _box_legend(ax.legend(fontsize=8, loc="upper right"))
        ax.grid(alpha=0.2, axis="y")
        _style_spines(ax)

    fig.suptitle(f"L18 encoder/decoder decoupling probe "
                 f"(cold-start: {COLD_STAGE_A} -> {COLD_STAGE_B})")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Axis 2 — coverage gradient
# ---------------------------------------------------------------------------

def coverage_counts():
    df = pd.read_csv(RESULTS / "l18_topk_summary.csv")
    rows = []
    for chain in CHAIN_ORDER:
        sub = df[df["chain"] == chain]
        n = sub["n_prompts_fired"].to_numpy()
        rows.append({
            "chain":  chain,
            "broad":  int((n >= 10).sum()),
            "sparse": int(((n >= 1) & (n < 10)).sum()),
            "ghost":  int((n == 0).sum()),
            "total":  int(len(n)),
        })
    return pd.DataFrame(rows)


def plot_coverage(counts: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    x = np.arange(len(counts))
    bottoms = np.zeros(len(counts))
    bands = [
        ("broad",  r"fires broadly ($n_{\mathrm{fired}} \geq 10$)"),
        ("sparse", r"fires sparsely ($1 \leq n_{\mathrm{fired}} < 10$)"),
        ("ghost",  r"BOS-only ghost ($n_{\mathrm{fired}} = 0$)"),
    ]
    for key, label in bands:
        vals = counts[key].to_numpy()
        ax.bar(x, vals, bottom=bottoms, width=0.62,
               color=BAND_COLOR[key], edgecolor="white", linewidth=1.0,
               label=label)
        for xi, v, b in zip(x, vals, bottoms):
            if v >= 3:
                ax.text(xi, b + v / 2, str(int(v)), ha="center", va="center",
                        fontsize=10,
                        color="white" if key == "broad" else "black")
        bottoms = bottoms + vals

    ax.set_xticks(x)
    ax.set_xticklabels(counts["chain"].tolist())
    ax.set_ylabel("# features in load-bearing $K{=}50$")
    ax.set_ylim(0, 55)
    ax.set_title("L18 causal coverage by chain")
    _box_legend(ax.legend(loc="upper right"))
    _style_spines(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_coverage_ecdf(out: Path):
    """ECDF of n_prompts_fired per chain over the load-bearing K=50.

    Shows the full firing-breadth distribution rather than the 3-band
    split. The chain gradient is visible at every threshold: the strict
    curve dominates flex which dominates shuffled.
    """
    df = pd.read_csv(RESULTS / "l18_topk_summary.csv")

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for chain in CHAIN_ORDER:
        sub = df[df["chain"] == chain]
        n = np.sort(sub["n_prompts_fired"].to_numpy())
        # ECDF: y = fraction of K with n_prompts_fired <= x
        y = np.arange(1, len(n) + 1) / len(n)
        # Step plot
        ax.step(np.concatenate([[0], n]),
                np.concatenate([[0], y]),
                where="post", color=CHAIN_COLOR[chain],
                linewidth=2.0, label=chain)

    ax.axvline(10, color="grey", linestyle=":", linewidth=0.8)
    # Threshold annotation lives in the upper-middle gap between the
    # flex curve (~0.5 at n=10) and shuffled curve (~0.7 at n=10), away
    # from any line. Boxed so it stays readable when curves cross.
    ax.text(10.5, 0.62, r"$n_{\mathrm{fired}}{=}10$ threshold",
            fontsize=8.5, color="dimgrey", va="center", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", pad=2))
    ax.set_xlabel(r"$n_{\mathrm{prompts\_fired}}$ on 100-prompt eval set")
    ax.set_ylabel("ECDF over load-bearing $K{=}50$")
    ax.set_xlim(-2, 102)
    ax.set_ylim(0, 1.02)
    ax.set_title("Coverage gradient: full firing-breadth distribution")
    _box_legend(ax.legend(loc="lower right"))
    _style_spines(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_dce_vs_nfired(out: Path):
    """Scatter of delta_ce vs n_prompts_fired per feature, colored by chain.

    Each chain's load-bearing K=50 is one cloud. The plot exposes a
    structural difference: shuffled's high-delta_ce features cluster on
    the n_fired=0 axis (BOS-only artifacts), strict's cluster on the
    high-n_fired side, flex sits between.

    Log y-axis spans the ~50x dynamic range from the smallest delta_ce in K
    (~0.01) to fid 1622's outlier at ~0.54 without crushing the bulk.
    """
    df = pd.read_csv(RESULTS / "l18_topk_summary.csv").copy()
    df["delta_ce"] = (df["delta_ce"].astype(str)
                      .str.lstrip("+").astype(float))

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for chain in CHAIN_ORDER:
        sub = df[df["chain"] == chain]
        ax.scatter(sub["n_prompts_fired"], sub["delta_ce"],
                   s=44, alpha=0.78,
                   color=CHAIN_COLOR[chain],
                   edgecolor="white", linewidth=0.8,
                   label=f"{chain}  (K=50)")

    ax.set_yscale("log")
    ax.set_ylim(0.005, 1.0)
    ax.axvline(10, color="grey", linestyle=":", linewidth=0.8)

    # Annotate the strict outlier (fid 1622) so the reader can locate it.
    fid_row = df[(df["chain"] == "strict") & (df["feature_idx"] == 1622)]
    if not fid_row.empty:
        x0 = float(fid_row["n_prompts_fired"].iloc[0])
        y0 = float(fid_row["delta_ce"].iloc[0])
        ax.annotate("fid 1622\n(leading-digit)",
                    xy=(x0, y0), xytext=(60, 0.78),
                    fontsize=9, color=PALETTE["twilight"],
                    arrowprops=dict(arrowstyle="-",
                                    color=PALETTE["twilight"], lw=0.7))

    ax.set_xlabel(r"$n_{\mathrm{prompts\_fired}}$ on 100-prompt eval set")
    ax.set_ylabel(r"causal $\Delta_{\mathrm{CE}}$  (log scale)")
    ax.set_xlim(-2, 102)
    ax.text(11, 0.0058, r"$n_{\mathrm{fired}}{=}10$",
            fontsize=8.5, color="dimgrey", va="bottom")
    ax.set_title("Per-feature causal load vs firing breadth")
    _box_legend(ax.legend(loc="lower right"))
    _style_spines(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Axis 3 — cross-chain identity
# ---------------------------------------------------------------------------

def identity_rows():
    """Build a small frame: (chain, fid, role_label, delta_ce, pass_rate, n_fired)."""

    # Strict named features. named_feature_summary.csv has predicate-level
    # rows; pick one row per fid and pull the matching predicate's pass_rate
    # plus the per-feature delta_ce (07d's load-bearing CSV).
    nf = pd.read_csv(RESULTS / "named_feature_summary.csv")
    strict_named = nf[nf["role"] == "loadbearing"].copy()
    # Predicate per feature is recorded in `feature_name`.
    strict_pass = {
        int(r["feature_idx"]): float(r["pass_rate"])
        for _, r in strict_named.iterrows()
        if r["feature_name"] == r["predicate"]
    }

    # Strict delta_ce per fid: pull from l18_topk_summary.csv.
    topk = pd.read_csv(RESULTS / "l18_topk_summary.csv")
    topk["delta_ce"] = topk["delta_ce"].astype(str).str.lstrip("+").astype(float)
    strict_topk = topk[topk["chain"] == "strict"]
    strict_dce = {int(r["feature_idx"]): float(r["delta_ce"])
                  for _, r in strict_topk.iterrows()}
    strict_nfired = {int(r["feature_idx"]): int(r["n_prompts_fired"])
                     for _, r in strict_topk.iterrows()}

    # Flex named features. flex_named_feature_summary.csv carries the
    # predicate pass_rate alongside delta_ce and n_fired directly.
    flex = pd.read_csv(RESULTS / "flex_named_feature_summary.csv")
    flex["delta_ce"] = flex["delta_ce"].astype(str).str.lstrip("+").astype(float)

    rows = []
    # 1622 strict (load-bearing rk0)
    rows.append({
        "chain": "strict", "fid": 1622, "label": "fid 1622 (strict)",
        "delta_ce": strict_dce[1622],
        "pass_rate": strict_pass[1622],
        "n_fired": strict_nfired[1622],
        "predicate": "leading-digit",
        "share_kind": "shared",
    })
    # 1622 flex (cross_chain_in_K)
    f1622 = flex[(flex["feature_idx"] == 1622)
                 & (flex["predicate"] == "leading_digit")].iloc[0]
    rows.append({
        "chain": "flexible", "fid": 1622, "label": "fid 1622 (flex)",
        "delta_ce": float(f1622["delta_ce"]),
        "pass_rate": float(f1622["pass_rate"]),
        "n_fired": int(f1622["n_fired"]),
        "predicate": "leading-digit",
        "share_kind": "shared",
    })
    # 6557 strict (load-bearing rk1, calc-block-opener — qualitative)
    rows.append({
        "chain": "strict", "fid": 6557, "label": "fid 6557 (strict)",
        "delta_ce": strict_dce[6557],
        "pass_rate": strict_pass[6557],
        "n_fired": strict_nfired[6557],
        "predicate": "calc-block",
        "share_kind": "chain-only",
    })
    # 3799 flex (load-bearing rk3, second leading-digit feature)
    f3799 = flex[(flex["feature_idx"] == 3799)
                 & (flex["predicate"] == "leading_digit")].iloc[0]
    rows.append({
        "chain": "flexible", "fid": 3799, "label": "fid 3799 (flex)",
        "delta_ce": float(f3799["delta_ce"]),
        "pass_rate": float(f3799["pass_rate"]),
        "n_fired": int(f3799["n_fired"]),
        "predicate": "leading-digit",
        "share_kind": "chain-only",
    })
    return pd.DataFrame(rows)


def plot_identity(df: pd.DataFrame, out: Path):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.5, 4.8),
        gridspec_kw={"width_ratios": [2, 2]})

    # --- Left panel: shared identity (fid 1622 strict vs flex). Both bars
    #     are the same feature, so the fid + predicate live in the title;
    #     x-ticks just say which chain. Above each bar: delta_ce, then a
    #     smaller line with predicate pass-rate and n_fired.
    shared = df[df["share_kind"] == "shared"].reset_index(drop=True)
    x = np.arange(len(shared))
    colors = [CHAIN_COLOR[c] for c in shared["chain"]]
    bars = ax1.bar(x, shared["delta_ce"], color=colors, width=0.55,
                   edgecolor="white", linewidth=1.0)
    ax1.set_yscale("log")
    ax1.set_ylim(0.005, 2.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(shared["chain"].tolist(), fontsize=11)
    ax1.set_ylabel(r"causal $\Delta_{\mathrm{CE}}$  (log scale)")
    ax1.set_title("Shared identity: fid 1622 (leading-digit primitive)")
    for bar, row in zip(bars, shared.itertuples()):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h * 1.12,
                 f"{h:.3f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
        ax1.text(bar.get_x() + bar.get_width() / 2, h * 1.65,
                 f"pass {row.pass_rate*100:.0f}%   "
                 f"$n_{{\\mathrm{{fired}}}}{{=}}{row.n_fired}$",
                 ha="center", va="bottom", fontsize=8.5, color="dimgrey")

    ratio = shared.iloc[0]["delta_ce"] / shared.iloc[1]["delta_ce"]
    # Arrow runs bar-edge to bar-edge (bar centers at x=0/1, width=0.55).
    ax1.annotate("",
                 xy=(0.725, shared.iloc[1]["delta_ce"]),
                 xytext=(0.275, shared.iloc[0]["delta_ce"]),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["twilight"],
                                 lw=1.4))
    # Ratio sits in the bar gap, shifted right of the gap midpoint so it
    # doesn't sit directly on top of the arrow's steepest segment. White
    # box keeps the glyphs readable where they overlap the arrow line.
    ax1.text(0.62, 0.17, rf"${ratio:.0f}\times$ lower",
             ha="center", va="center", fontsize=13,
             fontweight="bold", color=PALETTE["twilight"],
             bbox=dict(facecolor="white", edgecolor=PALETTE["twilight"],
                       boxstyle="round,pad=0.32", linewidth=0.8))
    _style_spines(ax1)

    # --- Right panel: chain-specific named features. Different fid per
    #     bar, so fid + predicate ride in the x-tick. Above bar: delta_ce,
    #     then a smaller line with pass-rate / n_fired.
    only = df[df["share_kind"] == "chain-only"].reset_index(drop=True)
    x = np.arange(len(only))
    colors = [CHAIN_COLOR[c] for c in only["chain"]]
    bars = ax2.bar(x, only["delta_ce"], color=colors, width=0.55,
                   edgecolor="white", linewidth=1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [f"{r.chain}\nfid {r.fid} ({r.predicate})" for r in only.itertuples()],
        fontsize=10)
    ax2.set_ylabel(r"causal $\Delta_{\mathrm{CE}}$")
    ax2.set_title("Chain-specific named features")
    for bar, row in zip(bars, only.itertuples()):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                 f"{h:.3f}", ha="center", va="bottom",
                 fontsize=11, fontweight="bold")
        if row.predicate == "calc-block":
            sub = f"qualitative   $n_{{\\mathrm{{fired}}}}{{=}}{row.n_fired}$"
        else:
            sub = (f"pass {row.pass_rate*100:.0f}%   "
                   f"$n_{{\\mathrm{{fired}}}}{{=}}{row.n_fired}$")
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.022,
                 sub, ha="center", va="bottom", fontsize=8.5, color="dimgrey")
    ax2.set_ylim(0, max(only["delta_ce"]) * 1.35)
    _style_spines(ax2)

    handles = [plt.Rectangle((0, 0), 1, 1, color=CHAIN_COLOR[c])
               for c in ("strict", "flexible")]
    _box_legend(fig.legend(handles, ["strict", "flexible"],
                           loc="lower center", ncol=2,
                           bbox_to_anchor=(0.5, -0.01)))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def plot_cross_chain_dce_scatter(out: Path):
    """For every fid in the union of {strict, flex} load-bearing K=50,
    plot (strict delta_ce, flex delta_ce). Diagonal line = equal magnitude.

    Points off-diagonal in the strict>>flex direction visualize the
    magnitude-graded shared identity story: the same feature-id can carry
    very different causal weight under the two reward shapes. Strict-only
    points (not in flex's load-bearing K but still in flex's full SAE)
    sit near the y=0 axis; flex-only points sit near the x=0 axis.
    """
    attr = pd.read_csv(RESULTS / "l18_attribution.csv")
    topk = pd.read_csv(RESULTS / "l18_topk_summary.csv")
    topk["delta_ce"] = (topk["delta_ce"].astype(str)
                        .str.lstrip("+").astype(float))

    strict_K = set(topk.loc[topk["chain"] == "strict", "feature_idx"].astype(int))
    flex_K   = set(topk.loc[topk["chain"] == "flexible", "feature_idx"].astype(int))
    union_K  = sorted(strict_K | flex_K)

    pivot = (attr.pivot_table(index="feature_idx", columns="chain",
                              values="delta_ce", aggfunc="first")
                 .reindex(union_K))
    pivot["in_strict_K"] = pivot.index.isin(strict_K)
    pivot["in_flex_K"]   = pivot.index.isin(flex_K)

    fig, ax = plt.subplots(figsize=(6.6, 6.2))

    # Diagonal reference
    lo = min(pivot["strict"].min(), pivot["flexible"].min(), 0.0)
    hi = max(pivot["strict"].max(), pivot["flexible"].max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color="grey", linestyle="--", linewidth=0.8, label="strict = flex")
    ax.axhline(0, color="lightgrey", linewidth=0.5)
    ax.axvline(0, color="lightgrey", linewidth=0.5)

    # Three groupings. Order so densest set draws first.
    groups = [
        (pivot["in_strict_K"] & pivot["in_flex_K"],
         "in both K", PALETTE["twilight"], 70),
        (pivot["in_strict_K"] & ~pivot["in_flex_K"],
         "strict K only", PALETTE["lilac"], 50),
        (~pivot["in_strict_K"] & pivot["in_flex_K"],
         "flex K only", PALETTE["rose"], 50),
    ]
    for mask, label, color, size in groups:
        sub = pivot[mask]
        ax.scatter(sub["strict"], sub["flexible"],
                   s=size, alpha=0.78, color=color,
                   edgecolor="white", linewidth=0.7, label=label)

    # Annotate the named features. Position labels above their points
    # with a vertical arrow pointing down -- the upper region of the plot
    # is data-empty, so this avoids overlap with the cluster near origin.
    annotation_offsets = {
        1622: (-0.05, 0.18),   # to the upper-left of fid 1622's point
        6557: (-0.02, 0.10),   # to the upper-left of fid 6557's point
    }
    annotation_names = {
        1622: "fid 1622\nleading-digit",
        6557: "fid 6557\ncalc-block",
    }
    for fid in (1622, 6557):
        if fid in pivot.index:
            xy = (pivot.at[fid, "strict"], pivot.at[fid, "flexible"])
            dx, dy = annotation_offsets[fid]
            ax.annotate(annotation_names[fid],
                        xy=xy, xytext=(xy[0] + dx, xy[1] + dy),
                        fontsize=9, color=PALETTE["twilight"],
                        ha="center", va="bottom",
                        arrowprops=dict(arrowstyle="-",
                                        color=PALETTE["twilight"],
                                        lw=0.7),
                        bbox=dict(facecolor="white", edgecolor="none",
                                  pad=1.5))

    ax.set_xlabel(r"strict  $\Delta_{\mathrm{CE}}$")
    ax.set_ylabel(r"flexible  $\Delta_{\mathrm{CE}}$")
    # Tighten plot to data range so the gradient cluster fills the canvas
    # rather than getting squeezed by an oversized empty quadrant.
    axis_max = max(pivot["strict"].max(), pivot["flexible"].max()) * 1.10
    axis_min = min(pivot["strict"].min(), pivot["flexible"].min()) - 0.02
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect("equal")
    ax.set_title("Cross-chain causal load per feature-id (L18, ppo_step30)")
    _box_legend(ax.legend(loc="upper right", fontsize=9))
    _style_spines(ax)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    # --- Axis 1: drift heatmaps + L18 decoupling probe ---
    drift = pd.read_csv(RESULTS / "decoder_drift.csv")
    print(f"Drift rows: {len(drift)} ({drift['chain'].nunique()} chains)")
    plot_drift_heatmap(
        drift, "dec_cos_mean", "Decoder cosine (dec_cos_mean)",
        FIGURES / "fig1_drift_heatmap.png")
    print(f"  -> {FIGURES / 'fig1_drift_heatmap.png'}")
    plot_drift_heatmap(
        drift, "enc_cos_mean", "Encoder cosine (enc_cos_mean)",
        FIGURES / "fig1b_encoder_drift_heatmap.png")
    print(f"  -> {FIGURES / 'fig1b_encoder_drift_heatmap.png'}")
    plot_l18_decoupling(FIGURES / "fig1c_l18_decoupling_probe.png")
    print(f"  -> {FIGURES / 'fig1c_l18_decoupling_probe.png'}")

    # --- Axis 2: coverage gradient + ECDF + per-feature scatter ---
    counts = coverage_counts()
    print("\nCoverage band counts:")
    print(counts.to_string(index=False))
    plot_coverage(counts, FIGURES / "fig2_coverage_gradient.png")
    print(f"  -> {FIGURES / 'fig2_coverage_gradient.png'}")
    plot_coverage_ecdf(FIGURES / "fig2b_coverage_ecdf.png")
    print(f"  -> {FIGURES / 'fig2b_coverage_ecdf.png'}")
    plot_dce_vs_nfired(FIGURES / "fig2c_dce_vs_nfired.png")
    print(f"  -> {FIGURES / 'fig2c_dce_vs_nfired.png'}")

    # --- Axis 3: cross-chain identity bars + cross-chain delta_ce scatter ---
    ident = identity_rows()
    print("\nIdentity rows:")
    print(ident.to_string(index=False))
    plot_identity(ident, FIGURES / "fig3_cross_chain_identity.png")
    print(f"  -> {FIGURES / 'fig3_cross_chain_identity.png'}")
    plot_cross_chain_dce_scatter(FIGURES / "fig3b_cross_chain_dce_scatter.png")
    print(f"  -> {FIGURES / 'fig3b_cross_chain_dce_scatter.png'}")


if __name__ == "__main__":
    main()
