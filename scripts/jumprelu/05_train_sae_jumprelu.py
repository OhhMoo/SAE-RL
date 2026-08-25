"""05_train_sae_jumprelu.py

JumpReLU counterpart to 05_train_sae.py (which trains TopK SAEs). Trains a
JumpReLU SAE at a given layer, either from cached activations or extracted
directly via forward hook for layers not covered by the cached activation
dataset (only layers 6, 12, 18, 23 are cached).

Usage:
    python scripts/jumprelu/05_train_sae_jumprelu.py \
        --layer 21 --l0 1e-3 --lr 1e-3 --steps 5000 --seed 0 \
        --out results/jumprelu_layer21_l0_1e-03_seed0.pt
"""
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jumprelu_extract import (  # noqa: E402
    build_val_loader,
    extract_layer_activations,
    load_gsm8k_split,
    normalize_activations,
)
from jumprelu_train import train_sae  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--l0", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bandwidth", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expansion", type=int, default=16)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out", type=str, required=True, help="Path to save the .pt checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, val_ds = load_gsm8k_split(seed=0)
    val_loader, _ = build_val_loader(val_ds, args.model_name)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float16).to(device)
    model.eval()
    raw_acts = extract_layer_activations(model, args.layer, val_loader, device=device)
    del model
    torch.cuda.empty_cache()

    norm = normalize_activations(raw_acts, seed=args.seed)

    sae, logs = train_sae(
        train_norm=norm["train_norm"],
        val_norm=norm["val_norm"],
        act_mean=norm["act_mean"],
        act_scale=norm["act_scale"],
        l0_coefficient=args.l0,
        lr=args.lr,
        bandwidth=args.bandwidth,
        steps=args.steps,
        seed=args.seed,
        expansion=args.expansion,
        device=device,
    )

    best = min(logs, key=lambda l: l["val_mse"])
    print(
        f"layer={args.layer} l0={args.l0:.0e} seed={args.seed} | "
        f"val_mse={best['val_mse']:.4f} nmse={best['nmse']:.4f} "
        f"expl_var={best['expl_var']:.4f} hard_l0={best['hard_l0']:.1f} "
        f"dead_frac={best['dead_frac']:.4f}"
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sae_state_dict": sae.state_dict(),
            "d_in": norm["train_norm"].shape[1],
            "d_sae": norm["train_norm"].shape[1] * args.expansion,
            "expansion": args.expansion,
            "l0_coefficient": args.l0,
            "lr": args.lr,
            "bandwidth": args.bandwidth,
            "steps": args.steps,
            "layer": args.layer,
            "seed": args.seed,
            "act_mean": norm["act_mean"],
            "act_scale": norm["act_scale"],
            "logs": logs,
            "best": best,
        },
        args.out,
    )
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
