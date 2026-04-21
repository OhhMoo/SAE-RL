#!/usr/bin/env python3
"""
08_extract_wandb.py
Extract PPO training metrics from WandB and save to CSV for temporal precedence analysis.

Saves results/training_curves.csv with columns:
  step, kl_div, reward, response_length, solve_rate (and any other logged metrics)

Usage:
    python scripts/08_extract_wandb.py
    python scripts/08_extract_wandb.py --run_id <id>
    python scripts/08_extract_wandb.py --project sae_rl_gsm8k --entity myteam
"""

import argparse
import os
import sys

import pandas as pd


KNOWN_METRIC_ALIASES = {
    # WandB key → our canonical column name
    "actor/kl_penalty": "kl_div",
    "actor/kl_loss": "kl_div",
    "actor/ppo_kl": "kl_div",
    "actor/reward_kl_penalty": "kl_div",
    "critic/kl": "kl_div",
    "train/kl": "kl_div",
    "kl": "kl_div",
    "actor/pg_loss": "pg_loss",
    "train/reward": "reward",
    "reward": "reward",
    "critic/reward": "reward",
    "critic/rewards/mean": "reward",
    "critic/score/mean": "reward",
    # verl response length columns
    "train/response_length": "response_length",
    "response_length": "response_length",
    "response_length/mean": "response_length",
    "rollout/response_length": "response_length",
    "response_length_non_aborted/mean": "response_length",
    # verl solve rate / accuracy columns
    "train/solve_rate": "solve_rate",
    "val/solve_rate": "solve_rate",
    "test/solve_rate": "solve_rate",
    "train/acc": "solve_rate",
    "val/acc": "solve_rate",
    "val-core/openai/gsm8k/acc/mean@1": "solve_rate",
    "val-aux/openai/gsm8k/reward/mean@1": "solve_rate",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="sae_rl_gsm8k")
    parser.add_argument("--entity", default=None,
                        help="WandB entity (username/org). Leave blank to use default.")
    parser.add_argument("--run_id", default=None,
                        help="Specific WandB run ID. If omitted, uses most recent PPO run.")
    parser.add_argument("--output", default="results/training_curves.csv")
    args = parser.parse_args()

    try:
        import wandb
    except ImportError:
        print("[error] wandb not installed. Run: pip install wandb")
        sys.exit(1)

    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}" if args.entity else args.project

    # ── Find the run ────────────────────────────────────────────────────────
    if args.run_id:
        run = api.run(f"{project_path}/{args.run_id}")
        print(f"Using run: {run.name} ({run.id})")
    else:
        runs = list(api.runs(project_path))
        if not runs:
            print(f"[error] No runs found in project '{project_path}'.")
            sys.exit(1)

        ppo_runs = [r for r in runs if "ppo" in r.name.lower()]
        if not ppo_runs:
            print(f"[warn] No runs with 'ppo' in name. Available runs:")
            for r in runs:
                print(f"  {r.name} ({r.id})")
            run = runs[0]
        else:
            run = sorted(ppo_runs, key=lambda r: r.created_at, reverse=True)[0]

        print(f"Using run: {run.name} ({run.id})  [created {run.created_at}]")

    # ── Download history ────────────────────────────────────────────────────
    print("Downloading run history (this may take a moment)...")
    history = run.history(samples=100_000, pandas=True)

    if history.empty:
        print("[error] Run history is empty.")
        sys.exit(1)

    print(f"\nRaw columns ({len(history.columns)}):")
    for col in sorted(history.columns):
        non_null = history[col].notna().sum()
        print(f"  {col:<60} ({non_null} non-null)")

    # ── Rename known metrics to canonical names ─────────────────────────────
    rename_map = {}
    for raw_col in history.columns:
        for alias, canonical in KNOWN_METRIC_ALIASES.items():
            if raw_col.lower() == alias.lower() and canonical not in rename_map.values():
                rename_map[raw_col] = canonical
                break

    history = history.rename(columns=rename_map)

    # ── Identify step column ────────────────────────────────────────────────
    step_col = None
    for candidate in ["_step", "global_step", "step", "trainer/global_step"]:
        if candidate in history.columns:
            step_col = candidate
            break

    if step_col is None:
        print("[warn] No step column found. Using row index as step.")
        history["step"] = history.index
        step_col = "step"
    elif step_col != "step":
        history = history.rename(columns={step_col: "step"})

    # Sort by step and drop duplicate steps (keep last)
    history = history.sort_values("step").drop_duplicates(subset=["step"], keep="last")

    # ── Save ────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    history.to_csv(args.output, index=False)

    canonical_found = [c for c in ["kl_div", "reward", "response_length", "solve_rate"]
                       if c in history.columns]
    print(f"\nCanonical metrics found: {canonical_found}")
    print(f"Step range: {int(history['step'].min())} → {int(history['step'].max())}")
    print(f"Saved {len(history)} rows → {args.output}")


if __name__ == "__main__":
    main()
