"""JumpReLU Sparse Autoencoder.

Architecture: log-parameterized threshold (always positive via exp()),
straight-through estimator for gradients through the hard threshold,
kaiming uniform init, unit-norm decoder columns.
"""
import math

import torch
import torch.nn as nn


class JumpReLUSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, init_threshold: float = 0.03, bandwidth: float = 0.1):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.bandwidth = bandwidth

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.log_threshold = nn.Parameter(torch.full((d_sae,), math.log(init_threshold)))

        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        self.normalize_decoder()

    @property
    def threshold(self) -> torch.Tensor:
        """Per-latent activation threshold. Always positive via exp() of a learned log value."""
        return self.log_threshold.exp()

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Project decoder rows back onto the unit sphere after each optimizer step."""
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec.div_(norms)

    def encode(self, x: torch.Tensor):
        """Returns (soft-thresholded activations, hard mask, soft mask).

        hard_mask is the true JumpReLU output (pre > threshold). soft_mask is a
        sigmoid relaxation used only to route gradients through the threshold via
        a straight-through estimator: forward pass uses hard_mask, backward pass
        uses soft_mask's gradient.
        """
        pre = x @ self.W_enc + self.b_enc
        threshold = self.threshold
        hard_mask = (pre > threshold).float()
        soft_mask = torch.sigmoid((pre - threshold) / self.bandwidth)
        st_mask = hard_mask.detach() - soft_mask.detach() + soft_mask
        acts = pre * st_mask
        return acts, hard_mask, soft_mask

    def decode(self, sae_acts: torch.Tensor) -> torch.Tensor:
        return sae_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        sae_acts, hard_mask, soft_mask = self.encode(x)
        x_hat = self.decode(sae_acts)
        return x_hat, sae_acts, hard_mask, soft_mask
