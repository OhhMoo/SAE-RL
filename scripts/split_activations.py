"""
One-shot migration: split existing monolithic activation files into
`{stem}_train.pt` + `{stem}_val.pt` without re-running the model.

Existing SAEs that were trained on the monolithic file have already seen the
"val" slice, so their FVE on `_val.pt` is still in-sample. The point of this
split is to make *future* training runs honest.

Usage:
    python scripts/split_activations.py --activations_dir data/activations
    python scripts/split_activations.py --source ppo_step100_layer23
"""
import argparse
from pathlib import Path

import torch


def split_one(src: Path, val_fraction: float, overwrite: bool, seed: int) -> None:
    stem = src.stem
    if stem.endswith("_train") or stem.endswith("_val"):
        return
    out_dir = src.parent
    train_path = out_dir / f"{stem}_train.pt"
    val_path = out_dir / f"{stem}_val.pt"
    if train_path.exists() and val_path.exists() and not overwrite:
        print(f"[skip] {train_path.name} and {val_path.name} already exist")
        return

    tensor = torch.load(src, weights_only=True)
    n_total = tensor.shape[0]
    n_val = int(n_total * val_fraction)
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    torch.save(tensor[train_idx], train_path)
    torch.save(tensor[val_idx], val_path)
    print(f"[ok] {src.name}: {n_total} -> train={n_train} + val={n_val} (seed={seed})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_dir", default="data/activations")
    parser.add_argument("--source", default=None,
                        help="Optional single stem to split (e.g. ppo_step100_layer23). "
                             "If unset, splits every monolithic *.pt in the directory.")
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for the train/val shuffle. Same seed across "
                             "files keeps the split reproducible.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    act_dir = Path(args.activations_dir)
    if args.source is not None:
        srcs = [act_dir / f"{args.source}.pt"]
    else:
        srcs = sorted(p for p in act_dir.glob("*.pt")
                      if not (p.stem.endswith("_train") or p.stem.endswith("_val")))
    if not srcs:
        print(f"No monolithic activation files found in {act_dir}")
        return
    for src in srcs:
        if not src.exists():
            print(f"[missing] {src}")
            continue
        split_one(src, args.val_fraction, args.overwrite, args.seed)


if __name__ == "__main__":
    main()
