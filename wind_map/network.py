"""
network.py — LatentModel: the full Attentive Neural Process
    for wind prediction.

    model = LatentModel(num_hidden, x_dim=3, y_dim=3)
    mu, sigma, kl, loss = model(context_x, context_y, target_x, target_y)
"""

import torch as t
import torch.nn as nn
from torch.distributions import kl_divergence

from wind_map.module import (
    LatentEncoder, DeterministicEncoder, Decoder,
)


class LatentModel(nn.Module):
    """
    Attentive Neural Process for wind vector prediction.

    Inputs:
        context_x  [B, N_ctx, 3]  — lat_norm, lon_norm, alt_norm
        context_y  [B, N_ctx, 3]  — wind_dir_sin, wind_dir_cos, wind_speed_norm
        target_x   [B, N_tgt, 3]  — query positions
        target_y   [B, N_tgt, 3]  — ground truth (training only)

    Outputs:
        mu    [B, N_tgt, 3]  — predicted mean
        sigma [B, N_tgt, 3]  — predicted std (positive)
        kl    [B, N_tgt]     — KL divergence (training only, else None)
        loss  scalar          — negative ELBO (training only, else None)
    """

    def __init__(self, num_hidden, x_dim=3, y_dim=3, num_heads=4,
                 num_layers=4, dropout=0.0,
                 free_bits=0.01):
        super(LatentModel, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.free_bits = free_bits

        assert num_hidden % num_heads == 0, (
            f"num_hidden ({num_hidden}) must be divisible"
            f" by num_heads ({num_heads})")

        num_latents = num_hidden
        decoder_output_sizes = [num_hidden] * 2 + [2 * y_dim]

        self.latent_encoder = LatentEncoder(
            num_hidden, num_latents,
            x_dim=x_dim, y_dim=y_dim,
            num_heads=num_heads, num_layers=num_layers,
            dropout=dropout)

        self.deterministic_encoder = DeterministicEncoder(
            num_hidden,
            x_dim=x_dim, y_dim=y_dim,
            num_heads=num_heads, num_layers=num_layers,
            dropout=dropout)

        representation_size = num_hidden + num_latents
        self.decoder = Decoder(
            x_dim, representation_size, decoder_output_sizes,
            target_hidden=num_hidden, dropout=dropout
        )

    def forward(self, context_x, context_y, target_x, target_y=None,
                context_mask=None, target_mask=None, kl_weight=1.0):
        num_targets = target_x.size(1)

        # Prior from context
        prior = self.latent_encoder(context_x, context_y, mask=context_mask)

        if target_y is not None:
            # Posterior from full target set (training only)
            posterior = self.latent_encoder(
                target_x, target_y, mask=target_mask
            )
            latent_rep = posterior.rsample()
        else:
            posterior = None
            latent_rep = prior.rsample()

        # Broadcast z to every target point
        latent_rep = latent_rep.unsqueeze(1).expand(-1, num_targets, -1)

        # Deterministic path: cross-attention context -> target
        deterministic_rep = self.deterministic_encoder(
            context_x, context_y, target_x, context_mask=context_mask)

        # Decode to wind distribution parameters
        representation = t.cat([deterministic_rep, latent_rep], dim=-1)
        dist, mu, sigma = self.decoder(representation, target_x)

        if target_y is not None:
            log_p = dist.log_prob(target_y).sum(dim=-1)
            kl_per_dim = kl_divergence(posterior, prior)
            kl_per_dim = t.clamp(kl_per_dim, min=self.free_bits)
            kl_per_sample = kl_per_dim.sum(dim=-1)
            kl = kl_per_sample.unsqueeze(1).expand(-1, num_targets)

            if target_mask is not None:
                mask_f = target_mask.to(log_p.dtype)
                n_valid = mask_f.sum(dim=1).clamp(min=1.0)
                log_p_sum = (log_p * mask_f).sum(dim=1)
                kl = kl * mask_f
            else:
                n_valid = t.full(
                    (log_p.size(0),), float(num_targets),
                    device=log_p.device
                )
                log_p_sum = log_p.sum(dim=1)

            recon = -t.mean(log_p_sum / n_valid)
            kl_term = kl_weight * t.mean(kl_per_sample)
            loss = recon + kl_term
        else:
            kl = None
            loss = None

        return mu, sigma, kl, loss
