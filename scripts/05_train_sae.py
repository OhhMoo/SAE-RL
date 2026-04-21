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
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


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


def train_sae(activations, d_sae, k, epochs=50, lr=3e-4, batch_size=256, device="cuda",
              resample_interval=5, dead_threshold=1e-4, aux_coeff=1/32):
    """Train a TopK SAE with periodic dead-feature resampling and auxiliary loss.

    Args:
        resample_interval: resample dead features every this many epochs.
        dead_threshold:    features whose mean activation frequency falls below
                           this across the epoch are considered dead.
        aux_coeff:         weight of the auxiliary loss that pushes dead features
                           to activate on high-error tokens. Set 0 to disable.
    """
    d_model = activations.shape[-1]
    sae = TopKSAE(d_model, d_sae, k).to(device)

    # Pre-encoder centering: initialise b_pre to the data mean so the encoder
    # operates on zero-centred activations from the first step.
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

    # Track best epoch by reconstruction loss. The decoder unit-norm projection
    # after every step can fight Adam's moments and produce sporadic late-epoch
    # spikes; saving the best-loss state shields the checkpoint from those.
    best_loss = float("inf")
    best_state = None
    best_epoch = -1

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

        print(f"  Epoch {epoch+1:3d}/{epochs}: recon_loss={avg_loss:.6f}  "
              f"active={n_active}/{d_sae}  dead={n_dead}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            best_state = {k_: v.detach().cpu().clone() for k_, v in sae.state_dict().items()}

        prev_dead_mask = (feature_freq < dead_threshold)

        if (epoch + 1) % resample_interval == 0 and epoch < epochs - 1:
            dead_mask = feature_freq < dead_threshold
            resample_dead_features(sae, activations, dead_mask, device)

    if best_state is not None:
        sae.load_state_dict(best_state)
        print(f"  [best] Restored epoch {best_epoch} (recon_loss={best_loss:.6f})")

    return sae


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
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    activation_files = sorted(Path(args.activations_dir).glob("*.pt"))
    if not activation_files:
        print(f"No activation files found in {args.activations_dir}")
        return

    for act_file in activation_files:
        name = act_file.stem  # e.g., "pretrained_layer12"
        save_path = os.path.join(args.save_dir, f"sae_{name}.pt")
        if os.path.exists(save_path):
            print(f"\n[skip] {save_path} already exists")
            continue
        print(f"\n{'='*60}")
        print(f"Training SAE for: {name}")
        print(f"{'='*60}")

        activations = torch.load(act_file, weights_only=True)
        d_model = activations.shape[-1]
        d_sae = d_model * args.expansion_factor

        print(f"  Activations shape: {activations.shape}")
        print(f"  SAE: d_model={d_model}, d_sae={d_sae}, k={args.k}")

        sae = train_sae(
            activations, d_sae, args.k,
            epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, device=args.device,
            resample_interval=args.resample_interval,
            dead_threshold=args.dead_threshold,
            aux_coeff=args.aux_coeff,
        )

        torch.save({
            "state_dict": sae.state_dict(),
            "config": {
                "d_model": d_model,
                "d_sae": d_sae,
                "k": args.k,
                "source": name,
            },
        }, save_path)
        print(f"  Saved SAE -> {save_path}")


if __name__ == "__main__":
    main()
