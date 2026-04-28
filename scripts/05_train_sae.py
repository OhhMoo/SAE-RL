"""
Step 5: Train Sparse Autoencoders on cached activations.

Trains a TopK SAE on the residual stream activations collected in step 4.
Trains one SAE per (checkpoint, layer) pair for comparison.

Usage:
    python scripts/05_train_sae.py \
        --activations_dir data/activations \
        --save_dir checkpoints/saes \
        --expansion_factor 8 \
        --k 32
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def set_seed(seed: int) -> None:
    """Seed torch/numpy/python RNGs for SAE-training reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder with pre-encoder centering."""

    def __init__(self, d_model, d_sae, k):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.d_sae = d_sae

        self.b_pre = nn.Parameter(torch.zeros(d_model))  # pre-encoder centering
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)

        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )

    def encode(self, x):
        z = self.encoder(x - self.b_pre)
        topk_values, topk_indices = torch.topk(z, self.k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topk_indices, topk_values)
        return z_sparse, z  # z_sparse for reconstruction, z (pre-topk) for aux loss

    def forward(self, x):
        z_sparse, z_pre = self.encode(x)
        x_hat = self.decoder(z_sparse)
        return x_hat, z_sparse, z_pre


def resample_dead_features(sae, activations, dead_mask, device, batch_size=512):
    """Reinitialise dead encoder rows toward high-reconstruction-error examples.

    For each dead feature we pick a random token from the top-loss quartile,
    set its encoder row to that token's normalised residual direction, and reset
    its decoder column and encoder bias. The shared decoder bias is NOT reset —
    it encodes the learned mean of the residual stream and must be preserved.
    """
    n_dead = dead_mask.sum().item()
    if n_dead == 0:
        return

    n_sample = min(len(activations), 8192)
    idx = torch.randperm(len(activations))[:n_sample]
    sample = activations[idx].to(device).float()

    with torch.no_grad():
        losses = []
        for i in range(0, len(sample), batch_size):
            b = sample[i : i + batch_size]
            x_hat, _, _ = sae(b)
            losses.append((b - x_hat).pow(2).mean(dim=-1))
        loss_per_token = torch.cat(losses)

        threshold = loss_per_token.quantile(0.75)
        candidates = sample[loss_per_token >= threshold]
        if len(candidates) == 0:
            candidates = sample

        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        chosen = candidates[torch.randint(len(candidates), (n_dead,))]
        chosen_normed = nn.functional.normalize(chosen, dim=-1)

        sae.encoder.weight.data[dead_indices] = chosen_normed
        sae.encoder.bias.data[dead_indices] = 0.0
        sae.decoder.weight.data[:, dead_indices] = chosen_normed.T
        # NOTE: decoder.bias is a shared d_model vector (learned mean of residual
        # stream). Resetting it destroys centering and causes catastrophic loss
        # spikes — do NOT touch it here.

    print(f"    Resampled {n_dead} dead features")


@torch.no_grad()
def _eval_val_mse(sae, val_acts, device, batch_size=512):
    """Mean per-element reconstruction MSE on val_acts."""
    sae.eval()
    total = 0.0
    n = 0
    for i in range(0, len(val_acts), batch_size):
        b = val_acts[i:i + batch_size].to(device).float()
        x_hat, _, _ = sae(b)
        total += (b - x_hat).pow(2).sum().item()
        n += b.numel()
    sae.train()
    return total / n if n > 0 else float("inf")


def train_sae(activations, d_sae, k, epochs=50, lr=3e-4, batch_size=256, device="cuda",
              resample_interval=5, dead_threshold=1e-4, aux_coeff=1/32,
              init_state_dict=None, val_activations=None):
    """Train a TopK SAE with periodic dead-feature resampling and auxiliary loss.

    Args:
        resample_interval: resample dead features every this many epochs.
        dead_threshold:    features whose mean activation frequency falls below
                           this across the epoch are considered dead.
        aux_coeff:         weight of the auxiliary loss that pushes dead features
                           to activate on high-error tokens. Set 0 to disable.
        init_state_dict:   optional state_dict from a previous SAE. When provided,
                           feature indices stay aligned across checkpoints, which
                           makes decoder-cosine drift metrics meaningful.
        val_activations:   optional held-out tensor. When provided, best-epoch
                           selection uses val MSE; otherwise falls back to
                           training-loss selection.
    """
    d_model = activations.shape[-1]
    sae = TopKSAE(d_model, d_sae, k).to(device)

    if init_state_dict is not None:
        sae.load_state_dict(init_state_dict)
    else:
        # Pre-encoder centering: initialise b_pre to the data mean so the encoder
        # operates on zero-centred activations from the first step. Skipped on
        # warm-start so the previous checkpoint's b_pre carries over; Adam will
        # re-adapt it if the activation mean has shifted.
        with torch.no_grad():
            data_mean = activations.float().mean(dim=0).to(device)
            sae.b_pre.data.copy_(data_mean)

    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    dataset = TensorDataset(activations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_steps = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=lr / 10
    )

    # Dead-feature mask from the end of the previous epoch (used for aux loss)
    prev_dead_mask = None

    # Track best epoch by val MSE when val is available, else by training loss.
    # The decoder unit-norm projection after every step can fight Adam's moments
    # and produce sporadic late-epoch spikes; saving the best state shields the
    # checkpoint from those.
    best_metric = float("inf")
    best_state = None
    best_epoch = -1
    best_train_loss = float("inf")
    best_val_loss = None
    selection_metric = "val" if val_activations is not None else "train"

    for epoch in range(epochs):
        total_recon = 0.0
        feature_counts = torch.zeros(d_sae, device=device)
        n_tokens = 0
        n_batches = 0

        for (batch,) in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch = batch.to(device).float()
            x_hat, z_sparse, z_pre = sae(batch)

            recon_loss = (batch - x_hat).pow(2).mean()
            loss = recon_loss

            # Auxiliary loss: activate top-k dead features on the current batch
            # and penalise their reconstruction error. This ensures dead features
            # receive gradient even when they don't win the TopK competition.
            if aux_coeff > 0 and prev_dead_mask is not None and prev_dead_mask.any():
                dead_indices = prev_dead_mask.nonzero(as_tuple=True)[0]
                k_aux = min(k * 16, len(dead_indices))
                if k_aux > 0:
                    z_dead_only = z_pre[:, dead_indices]  # (B, n_dead)
                    topk_v, topk_local = torch.topk(z_dead_only, k_aux, dim=-1)
                    topk_global = dead_indices[topk_local.reshape(-1)].reshape_as(topk_local)
                    z_aux = torch.zeros_like(z_pre).scatter(-1, topk_global, topk_v)
                    x_hat_aux = sae.decoder(z_aux)
                    aux_loss = (batch - x_hat_aux).pow(2).mean()
                    loss = recon_loss + aux_coeff * aux_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                sae.decoder.weight.data = nn.functional.normalize(
                    sae.decoder.weight.data, dim=0
                )

            total_recon += recon_loss.item()
            feature_counts += (z_sparse != 0).float().sum(dim=0)
            n_tokens += batch.shape[0]
            n_batches += 1

        avg_loss = total_recon / n_batches
        feature_freq = feature_counts / n_tokens
        n_dead = (feature_freq < dead_threshold).sum().item()
        n_active = d_sae - n_dead

        val_loss = None
        if val_activations is not None:
            val_loss = _eval_val_mse(sae, val_activations, device)

        log = (f"  Epoch {epoch+1:3d}/{epochs}: recon_loss={avg_loss:.6f}  "
               f"active={n_active}/{d_sae}  dead={n_dead}")
        if val_loss is not None:
            log += f"  val_loss={val_loss:.6f}"
        print(log)

        current_metric = val_loss if val_loss is not None else avg_loss
        if current_metric < best_metric:
            best_metric = current_metric
            best_epoch = epoch + 1
            best_train_loss = avg_loss
            best_val_loss = val_loss
            best_state = {k_: v.detach().cpu().clone() for k_, v in sae.state_dict().items()}

        prev_dead_mask = (feature_freq < dead_threshold)

        if (epoch + 1) % resample_interval == 0 and epoch < epochs - 1:
            dead_mask = feature_freq < dead_threshold
            resample_dead_features(sae, activations, dead_mask, device)

    if best_state is not None:
        sae.load_state_dict(best_state)
        tag = f"val_loss={best_val_loss:.6f}" if best_val_loss is not None else f"recon_loss={best_train_loss:.6f}"
        print(f"  [best/{selection_metric}] Restored epoch {best_epoch} ({tag})")

    return sae, {
        "best_epoch": best_epoch,
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss,
        "selection_metric": selection_metric,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations_dir", type=str, default="data/activations")
    parser.add_argument("--save_dir", type=str, default="checkpoints/saes")
    parser.add_argument("--expansion_factor", type=int, default=8,
                        help="SAE hidden dim = expansion_factor * d_model")
    parser.add_argument("--k", type=int, default=32, help="TopK sparsity")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resample_interval", type=int, default=5,
                        help="Resample dead features every N epochs")
    parser.add_argument("--dead_threshold", type=float, default=1e-4,
                        help="Features firing less than this fraction are considered dead")
    parser.add_argument("--aux_coeff", type=float, default=1/32,
                        help="Auxiliary loss coefficient for dead feature revival (0 to disable)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Set RNG seed for SAE init + training order. Unset = default torch behavior.")
    parser.add_argument("--source", type=str, default=None,
                        help="Train only on this activation file stem (e.g., ppo_step100_layer12). "
                             "If unset, trains on all *.pt in activations_dir.")
    parser.add_argument("--output_suffix", type=str, default="",
                        help="Suffix appended before .pt in output filename (e.g., '_seed42').")
    parser.add_argument("--init_from_stage", type=str, default=None,
                        help="Stage label of a previously-trained SAE to warm-start from "
                             "(e.g. 'instruct_base' or 'ppo_step10'). For each source "
                             "'{stage}_layer{N}' this loads 'sae_{init_from_stage}_layer{N}.pt' "
                             "from --init_from_dir and copies its weights before training. "
                             "Preserves feature-index alignment across checkpoints.")
    parser.add_argument("--init_from_dir", type=str, default=None,
                        help="Directory to look in for the init SAE. Defaults to --save_dir.")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.seed is not None:
        set_seed(args.seed)

    act_dir = Path(args.activations_dir)

    def _resolve_source(source: str) -> tuple[Path, Path | None, str]:
        """Return (train_path, val_path_or_None, source_kind).

        Prefers {source}_train.pt + {source}_val.pt; falls back to {source}.pt.
        """
        train_path = act_dir / f"{source}_train.pt"
        val_path = act_dir / f"{source}_val.pt"
        if train_path.exists() and val_path.exists():
            return train_path, val_path, "split"
        legacy = act_dir / f"{source}.pt"
        if legacy.exists():
            return legacy, None, "legacy"
        raise FileNotFoundError(f"No activations for {source} in {act_dir}")

    if args.source is not None:
        sources = [args.source]
    else:
        seen: set[str] = set()
        for f in sorted(act_dir.glob("*_train.pt")):
            if (act_dir / f"{f.stem[:-len('_train')]}_val.pt").exists():
                seen.add(f.stem[:-len("_train")])
        for f in sorted(act_dir.glob("*.pt")):
            stem = f.stem
            if stem.endswith("_train") or stem.endswith("_val"):
                continue
            seen.add(stem)
        sources = sorted(seen)
        if not sources:
            print(f"No activation files found in {act_dir}")
            return

    init_dir = Path(args.init_from_dir) if args.init_from_dir else Path(args.save_dir)

    for source in sources:
        try:
            train_path, val_path, source_kind = _resolve_source(source)
        except FileNotFoundError as e:
            print(f"[error] {e}")
            continue

        save_path = os.path.join(args.save_dir, f"sae_{source}{args.output_suffix}.pt")
        if os.path.exists(save_path):
            print(f"\n[skip] {save_path} already exists")
            continue
        print(f"\n{'='*60}")
        print(f"Training SAE for: {source}")
        print(f"{'='*60}")

        activations = torch.load(train_path, weights_only=True)
        val_activations = torch.load(val_path, weights_only=True) if val_path is not None else None
        d_model = activations.shape[-1]
        d_sae = d_model * args.expansion_factor

        print(f"  Train activations shape: {activations.shape}")
        if val_activations is not None:
            print(f"  Val   activations shape: {val_activations.shape}")
        else:
            print(f"  [warn] No {source}_val.pt — best-epoch will use training loss.")
        print(f"  SAE: d_model={d_model}, d_sae={d_sae}, k={args.k}")

        init_state_dict = None
        if args.init_from_stage is not None:
            layer_suffix = source.split("_layer", 1)[-1]
            init_name = f"sae_{args.init_from_stage}_layer{layer_suffix}.pt"
            init_path = init_dir / init_name
            if not init_path.exists():
                print(f"[error] Init SAE not found: {init_path}")
                return
            print(f"  Warm-starting from: {init_path}")
            init_ckpt = torch.load(init_path, map_location=args.device, weights_only=True)
            init_state_dict = init_ckpt["state_dict"]

        sae, stats = train_sae(
            activations, d_sae, args.k,
            epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, device=args.device,
            resample_interval=args.resample_interval,
            dead_threshold=args.dead_threshold,
            aux_coeff=args.aux_coeff,
            init_state_dict=init_state_dict,
            val_activations=val_activations,
        )

        torch.save({
            "state_dict": sae.state_dict(),
            "config": {
                "d_model": d_model,
                "d_sae": d_sae,
                "k": args.k,
                "source": source,
                "source_kind": source_kind,
                "seed": args.seed,
                "init_from_stage": args.init_from_stage,
                "best_epoch": stats["best_epoch"],
                "best_train_loss": stats["best_train_loss"],
                "best_val_loss": stats["best_val_loss"],
                "selection_metric": stats["selection_metric"],
            },
        }, save_path)
        print(f"  Saved SAE -> {save_path}")


if __name__ == "__main__":
    main()
