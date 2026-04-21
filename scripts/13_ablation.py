#!/usr/bin/env python3
"""
13_ablation.py
Causal ablation: zero out concise-rise or verbose-rise feature decoder directions
from the residual stream at inference time, measuring the effect on GSM8k solve rate.

Note: this PPO run did not exhibit reward hacking (solve_rate peaked at ~10% with
no performance cliff). The ablation therefore tests whether concise-phase features
are causally load-bearing for format compliance and short-output generation, not
whether they are responsible for a degenerate strategy.

  step100 = early concise phase (response_length collapsed, format_rate ~99%)
  step300 = late concise phase (same regime, later in training)

Saves results/ablation/ablation_results.csv and ablation_figure.png.

Inputs:
  checkpoints/saes/               (or --sae_dir)
  results/precedence_analysis/    (feature_classification_layer{N}.csv)
  checkpoints/ppo_merged/step_*/  (merged PPO checkpoints)

Usage:
    python scripts/13_ablation.py
    python scripts/13_ablation.py --layers 23 --n_hacking_feats 10 20 50
"""

import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── SAE (matches 05_train_sae.py) ────────────────────────────────────────────

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


# ── Helpers ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a math problem solver. Think step by step. "
    "You MUST end your response with '#### <number>' where <number> is the "
    "final numerical answer (digits only, no units or markdown)."
)
INSTRUCTION_SUFFIX = "Let's think step by step and output the final answer after '####'."


def build_prompts(dataset, tokenizer, n: int = 200):
    """Return (prompt_strings, ground_truth_answers)."""
    prompts, answers = [], []
    for ex in list(dataset)[:n]:
        chat = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": ex["question"] + "\n" + INSTRUCTION_SUFFIX},
        ]
        p = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        # Ground truth: last number after #### in the solution
        gt = ex["answer"].split("####")[-1].strip().replace(",", "")
        prompts.append(p)
        answers.append(gt)
    return prompts, answers


def extract_answer(text: str) -> str | None:
    """Extract the number after #### from a model response."""
    if "####" not in text:
        return None
    after = text.split("####")[-1].strip().split()[0] if text.split("####")[-1].strip() else None
    if after:
        return after.replace(",", "").rstrip(".")
    return None


def evaluate_solve_rate(
    model, tokenizer, prompts: list[str], answers: list[str],
    device: str, batch_size: int = 8, max_new_tokens: int = 256,
    hook_fn=None, layer_idx: int | None = None,
) -> float:
    """Generate responses and compute exact-match solve rate.

    If hook_fn is provided, register it on model.model.layers[layer_idx].
    """
    model.eval()
    handles = []
    if hook_fn is not None and layer_idx is not None:
        handles.append(model.model.layers[layer_idx].register_forward_hook(hook_fn))

    correct = 0
    total = 0

    try:
        for i in range(0, len(prompts), batch_size):
            batch_p = prompts[i : i + batch_size]
            batch_a = answers[i : i + batch_size]

            enc = tokenizer(
                batch_p, return_tensors="pt", padding=True, truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                out_ids = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode only the new tokens
            prompt_len = enc["input_ids"].shape[1]
            for j, ids in enumerate(out_ids):
                new_ids = ids[prompt_len:]
                response = tokenizer.decode(new_ids, skip_special_tokens=True)
                pred = extract_answer(response)
                if pred is not None and pred == batch_a[j]:
                    correct += 1
                total += 1

    finally:
        for h in handles:
            h.remove()

    return correct / total if total > 0 else 0.0


# ── Ablation hook factory ─────────────────────────────────────────────────────

def make_ablation_hook(decoder_directions: torch.Tensor):
    """
    Returns a forward hook that projects out the given decoder directions from
    the residual stream after each layer forward pass.

    decoder_directions: (n_features, d_model) — unit-norm decoder columns to ablate.
    """
    # Pre-compute projection matrix: P = I - D @ D.T  (removes D-subspace)
    # For many features, apply iteratively to avoid huge matrix.
    # We use the simpler: x_ablated = x - (x @ D.T) @ D  (sum of rank-1 projections)
    directions = decoder_directions  # (n, d_model)

    def hook_fn(module, inp, output):
        is_tuple = isinstance(output, tuple)
        hidden = (output[0] if is_tuple else output).float()  # (B, T, d_model)
        B, T, D = hidden.shape
        flat = hidden.reshape(B * T, D)                        # (B*T, d_model)

        dirs = directions.to(flat.device)                      # (n, d_model)
        # Project out: flat -= (flat @ dirs.T) @ dirs
        # Shape: (B*T, n) @ (n, d_model) → (B*T, d_model)
        projections = (flat @ dirs.T) @ dirs
        ablated = (flat - projections).reshape(B, T, D)

        orig_dtype = output[0].dtype if is_tuple else output.dtype
        ablated = ablated.to(orig_dtype)
        return (ablated,) + output[1:] if is_tuple else ablated

    return hook_fn


# ── Main ──────────────────────────────────────────────────────────────────────

# Map from stage label → model path and PPO step
EVAL_CONFIGS = [
    # (stage_label, model_path, ppo_step, phase)
    ("ppo_step100", "checkpoints/ppo_merged/step_100", 100, "early_concise"),
    ("ppo_step300", "checkpoints/ppo_merged/step_300", 300, "late_concise"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_dir", default="checkpoints/saes")
    parser.add_argument("--classification_dir", default="results/precedence_analysis")
    parser.add_argument("--output_dir", default="results/ablation")
    parser.add_argument("--layers", type=int, nargs="+", default=[23, 18])
    parser.add_argument("--n_hacking_feats", type=int, nargs="+", default=[10, 20, 50],
                        help="Number of top hacking features to ablate (sweep)")
    parser.add_argument("--n_prompts", type=int, default=200,
                        help="Number of GSM8k test prompts to evaluate on")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading GSM8k test set...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    rows = []

    for layer in args.layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        # Load feature classification
        clf_csv = os.path.join(args.classification_dir, f"feature_classification_layer{layer}.csv")
        if not os.path.exists(clf_csv):
            print(f"  [skip] Classification file not found: {clf_csv}")
            continue

        clf_df = pd.read_csv(clf_csv)
        hacking_feats = clf_df[clf_df["class"] == "concise_rise"].nlargest(
            max(args.n_hacking_feats), "concise_score"
        )
        print(f"  Found {len(hacking_feats)} concise_rise features")

        for stage_label, model_path, ppo_step, phase in EVAL_CONFIGS:
            if not Path(model_path).exists():
                print(f"  [skip] Model not found: {model_path}")
                continue

            # Load SAE for this stage + layer
            sae_path = os.path.join(args.sae_dir, f"sae_{stage_label}_layer{layer}.pt")
            if not os.path.exists(sae_path):
                print(f"  [skip] SAE not found: {sae_path}")
                continue

            print(f"\n  Stage: {stage_label} ({phase} phase)")
            print(f"  Loading model: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto"
            )

            prompts, answers = build_prompts(dataset, tokenizer, n=args.n_prompts)

            # Load SAE decoder weights — shape (d_model, d_sae), cols are feature directions
            sae, cfg = load_sae(sae_path, args.device)
            # decoder.weight: (d_model, d_sae) → transpose to (d_sae, d_model)
            decoder_W = sae.decoder.weight.detach().float().T  # (d_sae, d_model)
            # Unit-norm (should already be, but enforce)
            decoder_W = nn.functional.normalize(decoder_W, dim=1)

            # ── Baseline (no ablation) ──────────────────────────────────────
            print(f"    Baseline (no ablation)...", end="", flush=True)
            baseline_rate = evaluate_solve_rate(
                model, tokenizer, prompts, answers,
                args.device, args.batch_size, args.max_new_tokens,
            )
            print(f" solve_rate={baseline_rate:.3f}")

            rows.append({
                "stage": stage_label,
                "ppo_step": ppo_step,
                "phase": phase,
                "layer": layer,
                "n_ablated": 0,
                "ablation_type": "none",
                "solve_rate": baseline_rate,
            })

            # ── Concise-rise feature ablation sweep ────────────────────────
            for n_feats in args.n_hacking_feats:
                top_n_idx = hacking_feats["feature_idx"].values[:n_feats]
                directions = decoder_W[top_n_idx]  # (n_feats, d_model)

                hook = make_ablation_hook(directions)
                print(f"    Ablating top-{n_feats} concise_rise features...", end="", flush=True)
                rate = evaluate_solve_rate(
                    model, tokenizer, prompts, answers,
                    args.device, args.batch_size, args.max_new_tokens,
                    hook_fn=hook, layer_idx=layer,
                )
                delta = rate - baseline_rate
                print(f" solve_rate={rate:.3f}  Δ={delta:+.3f}")

                rows.append({
                    "stage": stage_label,
                    "ppo_step": ppo_step,
                    "phase": phase,
                    "layer": layer,
                    "n_ablated": n_feats,
                    "ablation_type": "concise_rise",
                    "solve_rate": rate,
                })

            # ── Random feature ablation (control) ──────────────────────────
            # Ablate the same number of randomly chosen features to check specificity
            rng = np.random.default_rng(42)
            for n_feats in args.n_hacking_feats:
                rand_idx = rng.choice(cfg["d_sae"], size=n_feats, replace=False)
                directions = decoder_W[rand_idx]

                hook = make_ablation_hook(directions)
                print(f"    Ablating {n_feats} RANDOM features (control)...", end="", flush=True)
                rate = evaluate_solve_rate(
                    model, tokenizer, prompts, answers,
                    args.device, args.batch_size, args.max_new_tokens,
                    hook_fn=hook, layer_idx=layer,
                )
                delta = rate - baseline_rate
                print(f" solve_rate={rate:.3f}  Δ={delta:+.3f}")

                rows.append({
                    "stage": stage_label,
                    "ppo_step": ppo_step,
                    "phase": phase,
                    "layer": layer,
                    "n_ablated": n_feats,
                    "ablation_type": "random",
                    "solve_rate": rate,
                })

            del model
            torch.cuda.empty_cache()

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_csv = os.path.join(args.output_dir, "ablation_results.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved → {out_csv}")
    print(df.to_string(index=False))

    # ── Plot ──────────────────────────────────────────────────────────────────
    for layer in args.layers:
        layer_df = df[df["layer"] == layer]
        if layer_df.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
        fig.suptitle(f"Causal Ablation — Layer {layer}", fontweight="bold")

        for ax, (phase, phase_label) in zip(axes, [("early_concise", "Early concise phase (step 100)"),
                                                     ("late_concise",  "Late concise phase (step 300)")]):
            phase_df = layer_df[layer_df["phase"] == phase]
            if phase_df.empty:
                ax.set_visible(False)
                continue

            baseline = phase_df[phase_df["ablation_type"] == "none"]["solve_rate"].values
            baseline_val = baseline[0] if len(baseline) else 0.0

            for atype, color, label in [
                ("concise_rise", "#e63946", "Concise-rise features ablated"),
                ("random",       "#adb5bd", "Random features ablated (control)"),
            ]:
                sub = phase_df[phase_df["ablation_type"] == atype].sort_values("n_ablated")
                if sub.empty:
                    continue
                ax.plot(sub["n_ablated"], sub["solve_rate"],
                        marker="o", color=color, linewidth=2, label=label)

            ax.axhline(baseline_val, color="black", linestyle="--", linewidth=1.5,
                       label=f"Baseline ({baseline_val:.3f})")

            ax.set_xlabel("Number of ablated features")
            ax.set_ylabel("Solve rate")
            ax.set_title(phase_label)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

        plt.tight_layout()
        out_fig = os.path.join(args.output_dir, f"ablation_layer{layer}.png")
        plt.savefig(out_fig, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Figure saved → {out_fig}")


if __name__ == "__main__":
    main()
