#!/usr/bin/env python3
"""Per-step decoder drift comparison across three PPO chains:

    flexible  kl=0.005  flexible reward
    strict    kl=0.005  strict reward
    kl_high   kl=0.025  flexible reward     <-- the new run

The 5x KL coefficient predicts ~5x per-step drift if drift rate is set by
the KL budget. This script prints the per-step rate per (transition, layer,
chain) and the kl_high / flexible ratio.

Pass / fail bands (load-bearing for paper framing):
    ratio in [3.5, 6.5]  -> mechanism confirmed, headline locked
    ratio in [2.0, 3.5]  -> sublinear scaling, softer headline
    ratio outside        -> reframe paper around depth-specific reorganization

L23 is omitted: existing flexible chain has L23 at k=256, kl_high trains at
k=64. A separate L23 k=256 retrain would be needed to compare; defer until
results at L6/L12/L18 justify it.

Run from sae_rl/:
    python scripts/analyze_kl_sweep.py
"""
from pathlib import Path
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]

CHAINS = {
    "flexible": ROOT / "checkpoints" / "saes",
    "strict":   ROOT / "checkpoints" / "saes_strict",
    "kl_high":  ROOT / "checkpoints" / "saes_kl0p025",
}
LAYERS = [6, 12, 18]
TRANSITIONS = [
    ("instruct_base", "ppo_step10", 10),
    ("ppo_step10",    "ppo_step30", 20),
]


def load_dec(path: Path) -> torch.Tensor:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["state_dict"]["decoder.weight"].float()


def per_feature_cos(W_a: torch.Tensor, W_b: torch.Tensor) -> torch.Tensor:
    a = nn.functional.normalize(W_a, dim=0)
    b = nn.functional.normalize(W_b, dim=0)
    return (a * b).sum(dim=0)


print(f"\n{'transition':<28s}{'layer':<6s}", end="")
for c in CHAINS:
    print(f"{c:>13s}", end="")
print(f"{'kl_high/flex':>15s}")
print("-" * 90)

for a, b, dstep in TRANSITIONS:
    label = f"{a} -> {b}"
    for layer in LAYERS:
        print(f"{label:<28s}L{layer:<5d}", end="")
        rates: dict[str, float | None] = {}
        for chain, sae_dir in CHAINS.items():
            p_a = sae_dir / f"sae_{a}_layer{layer}.pt"
            p_b = sae_dir / f"sae_{b}_layer{layer}.pt"
            if not (p_a.exists() and p_b.exists()):
                rates[chain] = None
                print(f"{'(missing)':>13s}", end="")
                continue
            W_a, W_b = load_dec(p_a), load_dec(p_b)
            if W_a.shape != W_b.shape:
                rates[chain] = None
                print(f"{'(shape)':>13s}", end="")
                continue
            cos = per_feature_cos(W_a, W_b)
            rate = (1.0 - cos.mean().item()) / dstep
            rates[chain] = rate
            print(f"{rate:>13.6f}", end="")
        flex, klhi = rates.get("flexible"), rates.get("kl_high")
        if flex is not None and klhi is not None and flex > 0:
            ratio = klhi / flex
            if 3.5 <= ratio <= 6.5:
                tag = "PASS"
            elif 2.0 <= ratio < 3.5:
                tag = "SOFT"
            else:
                tag = "FAIL"
            print(f"{ratio:>10.2f} {tag}")
        else:
            print()
print()
