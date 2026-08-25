"""Training loop for JumpReLUSAE.

Loss = MSE(reconstruction) + l0_coefficient * soft_l0 (soft L0 used so the
sparsity penalty is differentiable; hard L0 is reported as a metric only).
Model selection: the checkpoint with the lowest validation MSE across all
logged steps is what gets returned, not necessarily the final step.
"""
import random
from typing import Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm

try:
    from .jumprelu_model import JumpReLUSAE
except ImportError:
    # allow running as a plain script (matches how 05_*.py scripts import
    # their siblings via sys.path insertion rather than package-relative imports)
    from jumprelu_model import JumpReLUSAE


def train_sae(
    train_norm: torch.Tensor,
    val_norm: torch.Tensor,
    act_mean: torch.Tensor,
    act_scale: torch.Tensor,
    l0_coefficient: float = 1e-3,
    lr: float = 1e-3,
    bandwidth: float = 0.1,
    batch_size: int = 4096,
    steps: int = 5000,
    log_every: int = 500,
    seed: int = 0,
    expansion: int = 16,
    device: Optional[str] = None,
):
    """Train a JumpReLUSAE on pre-normalized activations.

    train_norm / val_norm: (n_tokens, d_in) tensors, already mean/std-normalized.
    act_mean / act_scale: the normalization stats used to produce train_norm/val_norm,
        stored in the returned logs/checkpoint for reproducibility -- not used in training
        itself, since train_norm/val_norm are assumed already normalized.

    Returns (sae, logs). `sae` has the state_dict of the best (lowest val_mse)
    logged checkpoint loaded, not the final step's weights. `logs` is a list of
    dicts, one per logged step, with keys: step, val_mse, hard_l0, soft_l0,
    l0_ratio, expl_var, dead_frac, nmse.

    nmse follows the TopK arm's definition (NMSE = MSE / Var(x), pooled over
    held-out activations) so the two arms are directly comparable in the
    paper's Section 5.3. It is mathematically close to (1 - expl_var) but
    computed independently here to match the TopK arm's formula exactly
    rather than being derived from expl_var.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)

    d_in = train_norm.shape[1]
    d_sae = d_in * expansion
    n_tokens = train_norm.shape[0]

    sae = JumpReLUSAE(d_in, d_sae, init_threshold=0.03, bandwidth=bandwidth).to(device)
    opt = torch.optim.AdamW(sae.parameters(), lr=lr)

    best_val_mse = float("inf")
    best_state = None
    logs = []

    for step in tqdm(range(steps), desc=f"l0={l0_coefficient:.0e}, seed={seed}"):
        idx = torch.randint(0, n_tokens, (batch_size,))
        x = train_norm[idx].to(device)

        x_hat, sae_acts, hard_mask, soft_mask = sae(x)

        mse = F.mse_loss(x_hat, x)
        soft_l0 = soft_mask.sum(dim=-1).mean()
        loss = mse + l0_coefficient * soft_l0

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            sae.normalize_decoder()

        if step % log_every == 0 or step == steps - 1:
            sae.eval()
            with torch.no_grad():
                vx = val_norm.to(device)
                vhat, _, vhard, vsoft = sae(vx)
                val_mse = F.mse_loss(vhat, vx).item()
                hard_l0 = vhard.sum(dim=-1).mean().item()
                soft_l0_v = vsoft.sum(dim=-1).mean().item()
                dead = (vhard.sum(dim=0) == 0).float().mean().item()
                res_ss = (vx - vhat).pow(2).sum()
                tot_ss = (vx - vx.mean(dim=0)).pow(2).sum().clamp_min(1e-8)
                expl_var = (1 - res_ss / tot_ss).item()
                l0_ratio = soft_l0_v / max(hard_l0, 1e-8)

                # NMSE = MSE / Var(x), matching the TopK arm's Section 3.3 definition,
                # computed on pooled per-element statistics (not per-dimension).
                var_x = (vx - vx.mean(dim=0)).pow(2).mean().clamp_min(1e-8)
                mse_per_elem = (vx - vhat).pow(2).mean()
                nmse = (mse_per_elem / var_x).item()

            logs.append(
                {
                    "step": step,
                    "val_mse": val_mse,
                    "nmse": nmse,
                    "hard_l0": hard_l0,
                    "soft_l0": soft_l0_v,
                    "l0_ratio": l0_ratio,
                    "expl_var": expl_var,
                    "dead_frac": dead,
                }
            )

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_state = {k: v.clone() for k, v in sae.state_dict().items()}

            sae.train()

    sae.load_state_dict(best_state)
    sae.eval()
    return sae, logs


def best_qualifying_checkpoint(logs, hard_l0_cutoff: float = 300.0, weights=(0.45, 0.25, 0.20)):
    """Select the best logged checkpoint under a sparsity cutoff, scored by a
    weighted combination of explained variance, val MSE, and l0_ratio (how
    close the soft L0 used in training tracks the hard L0 actually achieved).

    Mirrors the manual scoring formula used during the layer 21/23 sweeps:
    45% expl_var, 25% -val_mse, 20% -(l0_ratio - 1). Any checkpoint with
    hard_l0 > hard_l0_cutoff is disqualified. Returns None if no checkpoint
    in `logs` qualifies -- callers should handle this explicitly rather than
    silently falling back to an arbitrary step.
    """
    w_ev, w_mse, w_ratio = weights

    def score(entry):
        if entry["hard_l0"] > hard_l0_cutoff:
            return float("-inf")
        return (
            w_ev * entry["expl_var"]
            - w_mse * entry["val_mse"]
            - w_ratio * abs(entry["l0_ratio"] - 1.0)
        )

    qualifying = [l for l in logs if l["hard_l0"] <= hard_l0_cutoff]
    if not qualifying:
        return None
    return max(qualifying, key=score)
