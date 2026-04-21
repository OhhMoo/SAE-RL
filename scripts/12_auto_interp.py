#!/usr/bin/env python3
"""
12_auto_interp.py
Auto-interpret top concise_rise and verbose_rise features using the Claude API.

The classifier (10_classify_features.py) labels SAE features by whether they
became more active after the verbose→concise format collapse during PPO. This
script picks the strongest examples of each class and asks Claude to label what
they represent.

For each feature to interpret:
  1. Load the SAE and activation tensor for a target checkpoint
  2. Find the top-K tokens that most strongly activate the feature
  3. Reconstruct surrounding text context from the GSM8k dataset
  4. Send to Claude API with a structured prompt
  5. Save interpretations to results/precedence_analysis/auto_interp.json

Inputs:
  results/precedence_analysis/feature_classification_layer{N}.csv  (from 10)
  data/activations/{stage}_layer{N}.pt
  checkpoints/saes/sae_{stage}_layer{N}.pt

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/12_auto_interp.py
    python scripts/12_auto_interp.py --layer 23 --n_features 20 --top_k_tokens 10
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset


# ── SAE model (matches 05_train_sae.py) ───────────────────────────────────────

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


# ── Token text reconstruction ─────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a math problem solver. Think step by step. "
    "You MUST end your response with '#### <number>' where <number> is the "
    "final numerical answer (digits only, no units or markdown)."
)
INSTRUCTION_SUFFIX = "Let's think step by step and output the final answer after '####'."


def build_token_index(tokenizer, n_samples: int = 5000, max_length: int = 512,
                      batch_size: int = 16) -> list[tuple[str, int]]:
    """
    Build a list mapping flat token index → (surrounding_text_window).
    Replicates the collection order used in 04_collect_activations.py.

    Returns: list of (text_window, token_pos_in_prompt) in flat order.
    """
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    questions = [ex["question"] for ex in dataset]
    if n_samples:
        questions = questions[:n_samples]

    token_map = []  # flat list: each entry is a 10-token window around that token

    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i : i + batch_size]

        for q in batch_questions:
            prompt = [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": q + "\n" + INSTRUCTION_SUFFIX},
            ]
            prompt_str = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            tokens = tokenizer(
                prompt_str, return_tensors="pt",
                truncation=True, max_length=max_length,
            )
            input_ids = tokens["input_ids"][0]  # (seq_len,)
            decoded   = [tokenizer.decode([tid]) for tid in input_ids]

            for pos, token_str in enumerate(decoded):
                # 5-token context window around this token
                start  = max(0, pos - 5)
                end    = min(len(decoded), pos + 6)
                window = "".join(decoded[start:end]).strip()
                token_map.append((window, token_str.strip()))

    return token_map


# ── Claude API call ───────────────────────────────────────────────────────────

CLASS_DESCRIPTIONS = {
    "concise_rise": (
        "more active AFTER the verbose→concise format collapse during PPO "
        "(i.e. characteristic of the short, format-compliant phase)"
    ),
    "verbose_rise": (
        "more active BEFORE the verbose→concise format collapse during PPO "
        "(i.e. characteristic of the early long, low-format-compliance phase)"
    ),
}


def interpret_feature(client, examples: list[str], feature_class: str,
                      model: str = "claude-opus-4-6") -> str:
    """Call Claude API to interpret a feature given top-activating examples."""
    examples_str = "\n".join(f"  {i+1}. {ex!r}" for i, ex in enumerate(examples))
    class_desc = CLASS_DESCRIPTIONS.get(feature_class, feature_class)

    prompt = f"""I am analyzing internal features of a language model trained on GSM8k math reasoning with PPO.

The following text snippets are from math problem contexts that most strongly activate a specific internal feature. This feature is classified as "{feature_class}" — {class_desc}:

{examples_str}

Based on these activating examples:
1. What concept or pattern does this feature likely represent?
2. Does the pattern look more like a format/style cue (length, punctuation, template tokens) or a content/reasoning cue (numbers, operations, problem entities)?
3. Give a short label (3-7 words) for this feature.

Be concise. Format your response as:
LABEL: <short label>
CONCEPT: <1-2 sentences>
KIND: format / reasoning / mixed
"""
    response = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--stage", type=str, default="ppo_step100",
                        help="Checkpoint to run SAE on for top-token extraction")
    parser.add_argument("--sae_dir", type=str, default="checkpoints/saes")
    parser.add_argument("--activations_dir", type=str, default="data/activations")
    parser.add_argument("--classification_dir", type=str, default="results/precedence_analysis")
    parser.add_argument("--output_dir", type=str, default="results/precedence_analysis")
    parser.add_argument("--n_features", type=int, default=20,
                        help="Number of top concise_rise + top verbose_rise features to interpret")
    parser.add_argument("--top_k_tokens", type=int, default=10,
                        help="Number of top-activating tokens per feature")
    parser.add_argument("--n_samples", type=int, default=3000,
                        help="Number of GSM8k prompts to use for token index (keep low for memory)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model", type=str, default="claude-opus-4-6")
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip API calls; just show which features would be interpreted")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("[error] Set ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    try:
        import anthropic
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"[error] Missing dependency: {e}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load classification ────────────────────────────────────────────────
    clf_csv = os.path.join(args.classification_dir,
                           f"feature_classification_layer{args.layer}.csv")
    if not os.path.exists(clf_csv):
        print(f"[error] Classification file not found: {clf_csv}")
        print("Run 10_classify_features.py first.")
        sys.exit(1)

    clf_df = pd.read_csv(clf_csv).sort_values("concise_score")

    top_concise = clf_df[clf_df["class"] == "concise_rise"].nlargest(args.n_features, "concise_score")
    top_verbose = clf_df[clf_df["class"] == "verbose_rise"].nsmallest(args.n_features, "concise_score")
    features_to_interp = pd.concat([top_concise, top_verbose])

    print(f"Features to interpret: {len(top_concise)} concise_rise, {len(top_verbose)} verbose_rise")

    if args.dry_run:
        print("\n[dry run] Would interpret these features:")
        print(features_to_interp[["feature_idx", "class", "concise_score"]].to_string(index=False))
        return

    # ── Load SAE and activations ───────────────────────────────────────────
    sae_path = os.path.join(args.sae_dir, f"sae_{args.stage}_layer{args.layer}.pt")
    act_path = os.path.join(args.activations_dir, f"{args.stage}_layer{args.layer}.pt")

    if not os.path.exists(sae_path):
        print(f"[error] SAE not found: {sae_path}")
        sys.exit(1)
    if not os.path.exists(act_path):
        print(f"[error] Activations not found: {act_path}")
        sys.exit(1)

    print(f"Loading SAE: {sae_path}")
    sae, cfg = load_sae(sae_path, args.device)

    print(f"Loading activations: {act_path}")
    acts = torch.load(act_path, weights_only=True).float()
    print(f"  Activations shape: {acts.shape}")

    # ── Compute z matrix in batches ───────────────────────────────────────
    print("Computing feature activations (this may take a moment)...")
    batch_size = 1024
    z_list = []
    for i in range(0, len(acts), batch_size):
        batch = acts[i : i + batch_size].to(args.device)
        with torch.no_grad():
            z_batch = sae.encode(batch).cpu()
        z_list.append(z_batch)
    z = torch.cat(z_list, dim=0)  # (n_tokens, d_sae)
    print(f"  Feature activations shape: {z.shape}")

    # ── Build token text index ─────────────────────────────────────────────
    # Any Qwen2.5-0.5B tokenizer works; prefer local merged checkpoints so we
    # don't hit HF if offline, otherwise fall back to the base instruct model.
    tokenizer_path = None
    for candidate in [
        "checkpoints/ppo_merged/step_100",
        "Qwen/Qwen2.5-0.5B-Instruct",
    ]:
        if os.path.isdir(candidate) or "/" in candidate:
            tokenizer_path = candidate
            break

    print(f"Building token text index using tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    token_map = build_token_index(
        tokenizer, n_samples=args.n_samples, max_length=512
    )

    n_map    = len(token_map)
    n_tokens = z.shape[0]
    print(f"  Token map: {n_map} entries | Activation tensor: {n_tokens} tokens")

    # Token indices are aligned: activation i ↔ token_map[i] (same collection order)
    usable = min(n_map, n_tokens)

    # ── Run interpretations ────────────────────────────────────────────────
    client = anthropic.Anthropic(api_key=api_key)
    results = []

    for _, row in features_to_interp.iterrows():
        feat_idx = int(row["feature_idx"])
        feat_class = row["class"]
        concise_score = float(row["concise_score"])

        # Get activation strengths for this feature across tokens
        feat_activations = z[:usable, feat_idx].numpy()

        # Top-K tokens by activation strength
        top_indices = np.argsort(feat_activations)[-args.top_k_tokens:][::-1]
        top_examples = []
        for tok_idx in top_indices:
            if feat_activations[tok_idx] > 0:
                window, token = token_map[tok_idx]
                top_examples.append(window)

        if not top_examples:
            print(f"  [skip] Feature {feat_idx} ({feat_class}): no activating tokens found")
            continue

        print(f"  Interpreting feature {feat_idx} ({feat_class}, score={concise_score:.4f})", end="", flush=True)

        try:
            interpretation = interpret_feature(client, top_examples, feat_class, args.model)
            print(" ✓")
        except Exception as e:
            print(f" ✗ ({e})")
            interpretation = f"[error: {e}]"
            time.sleep(2)

        results.append({
            "feature_idx":    feat_idx,
            "layer":          args.layer,
            "class":          feat_class,
            "concise_score":  concise_score,
            "top_examples":   top_examples,
            "interpretation": interpretation,
            "stage_used":     args.stage,
        })

        time.sleep(0.5)  # rate limit

    # ── Save ───────────────────────────────────────────────────────────────
    out_json = os.path.join(args.output_dir, f"auto_interp_layer{args.layer}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} interpretations → {out_json}")

    # Print summary
    print("\n── Interpretation Summary ──────────────────────────────")
    for r in results:
        label_line = next(
            (line for line in r["interpretation"].split("\n") if "LABEL:" in line), ""
        )
        print(f"  [{r['class']:>9}] feat {r['feature_idx']:>4}  {label_line}")


if __name__ == "__main__":
    main()
