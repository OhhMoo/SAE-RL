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
    """Sparse Autoencoder with TopK activation function."""

    def __init__(self, d_model, d_sae, k):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.d_sae = d_sae

        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)

        # Initialize decoder columns to unit norm
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )

    def encode(self, x):
        z = self.encoder(x)
        # TopK: zero out all but top-k activations
        topk_values, topk_indices = torch.topk(z, self.k, dim=-1)
        z_sparse = torch.zeros_like(z)
        z_sparse.scatter_(-1, topk_indices, topk_values)
        return z_sparse

    def forward(self, x):
        z_sparse = self.encode(x)
        x_hat = self.decoder(z_sparse)
        return x_hat, z_sparse


def resample_dead_features(sae, activations, dead_mask, device, batch_size=512):
    """Reinitialise dead encoder rows toward high-reconstruction-error examples.

    For each dead feature we pick a random token from the top-loss quartile,
    set its encoder row to that token's normalised residual direction, and reset
    its decoder column and biases.  This mirrors the resampling strategy used in
    Anthropic's SAE work.
    """
    n_dead = dead_mask.sum().item()
    if n_dead == 0:
        return

    # Compute per-token reconstruction loss on a random subset (up to 8k samples)
    n_sample = min(len(activations), 8192)
    idx = torch.randperm(len(activations))[:n_sample]
    sample = activations[idx].to(device).float()

    with torch.no_grad():
        losses = []
        for i in range(0, len(sample), batch_size):
            b = sample[i : i + batch_size]
            x_hat, _ = sae(b)
            losses.append((b - x_hat).pow(2).mean(dim=-1))
        loss_per_token = torch.cat(losses)

        # Use the top-25% highest-loss tokens as candidates
        threshold = loss_per_token.quantile(0.75)
        candidates = sample[loss_per_token >= threshold]
        if len(candidates) == 0:
            candidates = sample

        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        chosen = candidates[torch.randint(len(candidates), (n_dead,))]
        # Normalise to unit norm → encoder direction
        chosen_normed = nn.functional.normalize(chosen, dim=-1)

        sae.encoder.weight.data[dead_indices] = chosen_normed
        sae.encoder.bias.data[dead_indices] = 0.0
        sae.decoder.weight.data[:, dead_indices] = chosen_normed.T
        sae.decoder.bias.data[:] = 0.0  # reset output bias too

    print(f"    Resampled {n_dead} dead features")


def train_sae(activations, d_sae, k, epochs=50, lr=3e-4, batch_size=256, device="cuda",
              resample_interval=5, dead_threshold=1e-4):
    """Train a TopK SAE with periodic dead-feature resampling.

    Args:
        resample_interval: resample dead features every this many epochs.
        dead_threshold:    features whose mean activation frequency falls below
                           this across the epoch are considered dead.
    """
    d_model = activations.shape[-1]
    sae = TopKSAE(d_model, d_sae, k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    dataset = TensorDataset(activations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        # Track per-feature activation counts across the epoch
        feature_counts = torch.zeros(d_sae, device=device)
        n_tokens = 0
        n_batches = 0

        for (batch,) in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            batch = batch.to(device).float()
            x_hat, z_sparse = sae(batch)

            recon_loss = (batch - x_hat).pow(2).mean()
            optimizer.zero_grad()
            recon_loss.backward()
            optimizer.step()

            # Normalise decoder columns after each step
            with torch.no_grad():
                sae.decoder.weight.data = nn.functional.normalize(
                    sae.decoder.weight.data, dim=0
                )

            total_loss += recon_loss.item()
            feature_counts += (z_sparse != 0).float().sum(dim=0)
            n_tokens += batch.shape[0]
            n_batches += 1

        avg_loss = total_loss / n_batches
        feature_freq = feature_counts / n_tokens
        n_dead = (feature_freq < dead_threshold).sum().item()
        n_active = d_sae - n_dead

        print(f"  Epoch {epoch+1:3d}/{epochs}: recon_loss={avg_loss:.6f}  "
              f"active={n_active}/{d_sae}  dead={n_dead}")

        # Resample dead features periodically (skip final epoch)
        if (epoch + 1) % resample_interval == 0 and epoch < epochs - 1:
            dead_mask = feature_freq < dead_threshold
            resample_dead_features(sae, activations, dead_mask, device)

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
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    activation_files = sorted(Path(args.activations_dir).glob("*.pt"))
    if not activation_files:
        print(f"No activation files found in {args.activations_dir}")
        return

    for act_file in activation_files:
        name = act_file.stem  # e.g., "pretrained_layer12"
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
        )

        save_path = os.path.join(args.save_dir, f"sae_{name}.pt")
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
