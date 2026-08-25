"""05b_sweep_jumprelu.py

JumpReLU counterpart to 05b_sweep_hyperparams.sh. Two modes:

  --mode l0-sweep    sweep l0 coefficients at one layer, select the best
                      checkpoint under a hard_l0 sparsity cutoff (mirrors the
                      TopK arm's approach to standardising sparsity across
                      layers -- see 05b_sweep_hyperparams.sh).
  --mode seed-check   train the same (layer, l0) config multiple times varying
                      only the random seed, to check whether an observed
                      quality difference between layers is a reproducible
                      effect or a single-run artifact.

Usage:
    python scripts/jumprelu/05b_sweep_jumprelu.py --mode l0-sweep \
        --layer 21 --l0s 3e-5 1e-4 3e-4 5e-4 1e-3 \
        --out results/jumprelu_layer21_sweep.csv

    python scripts/jumprelu/05b_sweep_jumprelu.py --mode seed-check \
        --layers 20 21 --seeds 0 1 2 --l0 1e-3 \
        --out results/jumprelu_layer20_21_seed_check.csv
"""
import argparse
import csv
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
from jumprelu_train import best_qualifying_checkpoint, train_sae  # noqa: E402


CSV_FIELDS = [
    "mode", "layer", "seed", "l0_coefficient", "lr", "bandwidth", "steps",
    "qualifies", "step", "val_mse", "nmse", "expl_var", "hard_l0",
    "soft_l0", "l0_ratio", "dead_frac",
]


def write_rows(rows, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {out_path}")


def get_activations(model, layers, model_name, device):
    _, val_ds = load_gsm8k_split(seed=0)
    val_loader, _ = build_val_loader(val_ds, model_name)
    norm_data = {}
    for layer in layers:
        raw_acts = extract_layer_activations(model, layer, val_loader, device=device)
        norm_data[layer] = normalize_activations(raw_acts, seed=0)
    return norm_data


def run_l0_sweep(args, device):
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float16).to(device)
    model.eval()
    norm_data = get_activations(model, [args.layer], args.model_name, device)
    del model
    torch.cuda.empty_cache()

    norm = norm_data[args.layer]
    rows = []
    for l0_coef in args.l0s:
        sae, logs = train_sae(
            train_norm=norm["train_norm"], val_norm=norm["val_norm"],
            act_mean=norm["act_mean"], act_scale=norm["act_scale"],
            l0_coefficient=l0_coef, lr=args.lr, bandwidth=args.bandwidth,
            steps=args.steps, seed=args.seed, device=device,
        )
        best = best_qualifying_checkpoint(logs, hard_l0_cutoff=args.hard_l0_cutoff)
        qualifies = best is not None
        source = best if qualifies else min(logs, key=lambda l: l["val_mse"])
        print(
            f"l0={l0_coef:.0e} qualifies={qualifies} "
            f"expl_var={source['expl_var']:.4f} nmse={source['nmse']:.4f} "
            f"hard_l0={source['hard_l0']:.1f}"
        )
        rows.append(
            {
                "mode": "l0-sweep", "layer": args.layer, "seed": args.seed,
                "l0_coefficient": l0_coef, "lr": args.lr, "bandwidth": args.bandwidth,
                "steps": args.steps, "qualifies": qualifies, "step": source["step"],
                "val_mse": source["val_mse"], "nmse": source["nmse"],
                "expl_var": source["expl_var"], "hard_l0": source["hard_l0"],
                "soft_l0": source["soft_l0"], "l0_ratio": source["l0_ratio"],
                "dead_frac": source["dead_frac"],
            }
        )
        del sae
        torch.cuda.empty_cache()

    write_rows(rows, args.out)


def run_seed_check(args, device):
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float16).to(device)
    model.eval()
    norm_data = get_activations(model, args.layers, args.model_name, device)
    del model
    torch.cuda.empty_cache()

    rows = []
    for seed in args.seeds:
        for layer in args.layers:
            sae, logs = train_sae(
                train_norm=norm_data[layer]["train_norm"], val_norm=norm_data[layer]["val_norm"],
                act_mean=norm_data[layer]["act_mean"], act_scale=norm_data[layer]["act_scale"],
                l0_coefficient=args.l0, lr=args.lr, bandwidth=args.bandwidth,
                steps=args.steps, seed=seed, device=device,
            )
            best = min(logs, key=lambda l: l["val_mse"])
            print(
                f"layer={layer} seed={seed} expl_var={best['expl_var']:.4f} "
                f"nmse={best['nmse']:.4f} dead_frac={best['dead_frac']:.4f}"
            )
            rows.append(
                {
                    "mode": "seed-check", "layer": layer, "seed": seed,
                    "l0_coefficient": args.l0, "lr": args.lr, "bandwidth": args.bandwidth,
                    "steps": args.steps, "qualifies": "", "step": best["step"],
                    "val_mse": best["val_mse"], "nmse": best["nmse"],
                    "expl_var": best["expl_var"], "hard_l0": best["hard_l0"],
                    "soft_l0": best["soft_l0"], "l0_ratio": best["l0_ratio"],
                    "dead_frac": best["dead_frac"],
                }
            )
            del sae
            torch.cuda.empty_cache()

    write_rows(rows, args.out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["l0-sweep", "seed-check"], required=True)
    parser.add_argument("--layer", type=int, help="single layer, for --mode l0-sweep")
    parser.add_argument("--layers", type=int, nargs="+", help="layers to compare, for --mode seed-check")
    parser.add_argument("--l0s", type=float, nargs="+", default=[3e-5, 1e-4, 3e-4, 5e-4, 1e-3])
    parser.add_argument("--l0", type=float, default=1e-3, help="fixed l0, for --mode seed-check")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=0, help="fixed seed, for --mode l0-sweep")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bandwidth", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--hard-l0-cutoff", type=float, default=300.0)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == "l0-sweep":
        if args.layer is None:
            parser.error("--layer is required for --mode l0-sweep")
        run_l0_sweep(args, device)
    else:
        if not args.layers:
            parser.error("--layers is required for --mode seed-check")
        run_seed_check(args, device)


if __name__ == "__main__":
    main()
