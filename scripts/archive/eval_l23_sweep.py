"""
Eval all sweep SAEs in --sae_dir on a single (stage, layer) and write one CSV.

Each SAE's filename can be anything matching `sae_*.pt`; (k, d_sae) come from
the saved config, not the filename. NMSE on `{stage}_layer{L}_val.pt`, frac_rec
via the same delta-loss recipe used in scripts/eval_sae.py.

Output columns: sae_file, k, d_sae, expansion, best_epoch, best_val_loss,
                nmse, mean_l0, L_base, L_sae, L_mean, frac_rec.

Run from sae_rl/:
    python scripts/eval_l23_sweep.py \
        --sae_dir checkpoints/saes_sweep_l23 \
        --stage ppo_step100 --layer 23 \
        --model_path checkpoints/ppo_merged/step_100 \
        --output_csv results/l23_sweep.csv
"""

import argparse
import csv
import importlib.util
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sae_dir",         required=True)
    ap.add_argument("--activations_dir", default="data/activations")
    ap.add_argument("--stage",           required=True)
    ap.add_argument("--layer",           type=int, required=True)
    ap.add_argument("--model_path",      required=True)
    ap.add_argument("--output_csv",      required=True)
    ap.add_argument("--device",          default="cuda")
    ap.add_argument("--n_delta_prompts", type=int, default=100)
    args = ap.parse_args()

    scripts = Path(__file__).parent
    eval_mod = _load_module(scripts / "eval_sae.py", "eval_sae_mod")
    TopKSAE       = eval_mod.TopKSAE
    eval_nmse_l0  = eval_mod.eval_nmse_l0
    eval_delta    = eval_mod.eval_delta_loss

    val_path = Path(args.activations_dir) / f"{args.stage}_layer{args.layer}_val.pt"
    if not val_path.exists():
        raise FileNotFoundError(f"Missing val activations: {val_path}")
    val_acts = torch.load(val_path, weights_only=True)
    print(f"Val activations: {val_acts.shape} from {val_path}")

    print("Loading GSM8k test prompts...")
    test_prompts = [ex["question"] for ex in
                    load_dataset("openai/gsm8k", "main", split="test")]
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.float16
    ).to(args.device)

    sae_dir = Path(args.sae_dir)
    sae_files = sorted(sae_dir.glob("sae_*.pt"))
    if not sae_files:
        raise FileNotFoundError(f"No SAEs in {sae_dir}")
    print(f"Evaluating {len(sae_files)} SAE files")

    rows = []
    for f in sae_files:
        ckpt = torch.load(f, map_location=args.device, weights_only=False)
        cfg = ckpt["config"]
        sae = TopKSAE(cfg["d_model"], cfg["d_sae"], cfg["k"])
        sae.load_state_dict(ckpt["state_dict"], strict=False)
        sae.to(args.device).eval()

        nmse, mean_l0 = eval_nmse_l0(sae, val_acts, args.device)
        L_base, L_sae, L_mean, frac = eval_delta(
            sae, model, tokenizer, args.layer, test_prompts,
            args.device, n_prompts=args.n_delta_prompts,
        )
        expansion = cfg["d_sae"] // cfg["d_model"]
        print(f"\n[{f.name}]  k={cfg['k']}  exp={expansion}×  "
              f"nmse={nmse:.4f}  L0={mean_l0:.2f}  frac_rec={frac:.4f}")

        rows.append({
            "sae_file":    f.name,
            "k":           cfg["k"],
            "d_sae":       cfg["d_sae"],
            "expansion":   expansion,
            "best_epoch":  cfg.get("best_epoch", ""),
            "best_val_loss": (f"{cfg['best_val_loss']:.6f}"
                              if cfg.get("best_val_loss") is not None else ""),
            "nmse":        f"{nmse:.4f}",
            "mean_l0":     f"{mean_l0:.2f}",
            "L_base":      f"{L_base:.4f}",
            "L_sae":       f"{L_sae:.4f}",
            "L_mean":      f"{L_mean:.4f}",
            "frac_rec":    f"{frac:.4f}",
        })

        del sae
        if args.device == "cuda":
            torch.cuda.empty_cache()

    fieldnames = ["sae_file", "k", "d_sae", "expansion", "best_epoch",
                  "best_val_loss", "nmse", "mean_l0",
                  "L_base", "L_sae", "L_mean", "frac_rec"]
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nDone. Results -> {out}")


if __name__ == "__main__":
    main()
