"""
module.py — Attentive Neural Process building blocks.
Based on DeepMind ANP design.

Pipeline: concat(x, y) -> Linear -> self-attention -> aggregate -> ...
  Latent path:   self-attention -> mean -> mu/log_sigma -> z
  Deterministic: self-attention + cross-attention -> r*
  Decoder:       concat(r*, z, target_x) -> MLP -> mu, sigma
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# Attention (matches DeepMind ANP: multihead + residual + LayerNorm)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multihead attention with residual connection and LayerNorm.

    DeepMind ANP design:
      key, value, query (d_in) -> Linear(d_in,d_in) -> multihead ->
      cat([residual, result]) -> Linear(2*d_in, d_in) -> dropout ->
      + residual -> LayerNorm -> output
    """

    def __init__(self, num_hidden, h=4, dropout=0.1):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h

        self.key = nn.Linear(num_hidden, num_hidden, bias=False)
        self.value = nn.Linear(num_hidden, num_hidden, bias=False)
        self.query = nn.Linear(num_hidden, num_hidden, bias=False)

        self.residual_dropout = nn.Dropout(p=dropout)
        self.final_linear = nn.Linear(num_hidden * 2, num_hidden)
        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, key, value, query, mask=None):
        residual = query
        B, seq_q, _ = query.shape
        seq_k = key.size(1)
        H, hs = self.h, self.num_hidden_per_attn

        k = self.key(key).view(B, seq_k, H, hs).transpose(1, 2)
        v = self.value(value).view(B, seq_k, H, hs).transpose(1, 2)
        q = self.query(query).view(B, seq_q, H, hs).transpose(1, 2)

        scale = hs ** 0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, None, :], float('-inf'))
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, seq_q, H * hs)

        result = torch.cat([residual, out], dim=-1)
        result = self.final_linear(result)
        result = self.residual_dropout(result)
        result = result + residual
        result = self.layer_norm(result)
        return result, attn


# ---------------------------------------------------------------------------
# Encoders & Decoder
# ---------------------------------------------------------------------------

class DeterministicEncoder(nn.Module):
    """Self-attention on context -> cross-attention context->target -> r*.

    Keys/values come from self-attended context representations.
    Queries come from target_x encoded through same projection (zeros for y).
    """

    def __init__(self, num_hidden, x_dim=3, y_dim=3,
                 num_heads=4, num_layers=4, dropout=0.0):
        super().__init__()
        self.y_dim = y_dim
        self.input_projection = nn.Linear(x_dim + y_dim, num_hidden)
        self.self_attentions = nn.ModuleList([
            Attention(num_hidden, h=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.cross_attentions = nn.ModuleList([
            Attention(num_hidden, h=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, context_x, context_y, target_x, context_mask=None):
        # Context: concat(x,y) -> projection -> self-attention
        encoder_input = torch.cat([context_x, context_y], dim=-1)
        hidden = self.input_projection(encoder_input)
        for attn in self.self_attentions:
            hidden, _ = attn(hidden, hidden, hidden, mask=context_mask)

        # Target queries: target_x through same projection (zeros for y)
        B, N_tgt = target_x.shape[:2]
        dummy_y = torch.zeros(B, N_tgt, self.y_dim,
                              dtype=target_x.dtype, device=target_x.device)
        tgt_input = torch.cat([target_x, dummy_y], dim=-1)
        query = self.input_projection(tgt_input)

        # Cross-attention: keys/values from context, queries from targets
        for attn in self.cross_attentions:
            query, _ = attn(hidden, hidden, query, mask=context_mask)

        return query


class LatentEncoder(nn.Module):
    """concat(x,y) -> projection -> self-attention
    -> mean -> mu/log_sigma -> z"""

    def __init__(self, num_hidden, num_latents, x_dim=3, y_dim=3,
                 num_heads=4, num_layers=4, dropout=0.0):
        super().__init__()
        self.input_projection = nn.Linear(x_dim + y_dim, num_hidden)
        self.self_attentions = nn.ModuleList([
            Attention(num_hidden, h=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        d = num_hidden
        penultimate_size = int((d + num_latents) / 2)
        self.penultimate_layer = nn.Linear(d, penultimate_size)
        self.mean_layer = nn.Linear(penultimate_size, num_latents)
        self.std_layer = nn.Linear(penultimate_size, num_latents)

    def forward(self, x, y, mask=None):
        encoder_input = torch.cat([x, y], dim=-1)
        hidden = self.input_projection(encoder_input)
        for attn in self.self_attentions:
            hidden, _ = attn(hidden, hidden, hidden, mask=mask)

        if mask is None:
            hidden = hidden.mean(dim=1)
        else:
            mask_f = mask.unsqueeze(-1).to(hidden.dtype)
            counts = mask_f.sum(dim=1).clamp(min=1.0)
            hidden = (hidden * mask_f).sum(dim=1) / counts

        hidden = F.relu(self.penultimate_layer(hidden))
        mu = self.mean_layer(hidden)
        log_sigma = self.std_layer(hidden)
        sigma = F.softplus(log_sigma) + 1e-6

        return Normal(loc=mu, scale=sigma)


class Decoder(nn.Module):
    """concat(representation, projected_target_x) -> MLP -> mu, sigma"""

    def __init__(self, x_size, representation_size, output_sizes,
                 target_hidden=None, dropout=0.0):
        super().__init__()
        if target_hidden is None:
            target_hidden = output_sizes[0]
        self.target_projection = nn.Linear(x_size, target_hidden)
        layers = []
        current = representation_size + target_hidden
        for size in output_sizes[:-1]:
            layers.append(nn.Linear(current, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current = size
        layers.append(nn.Linear(current, output_sizes[-1]))
        self.decoder_mlp = nn.Sequential(*layers)

    def forward(self, representation, target_x):
        target_x = self.target_projection(target_x)
        hidden = torch.cat([representation, target_x], dim=-1)
        hidden = self.decoder_mlp(hidden)
        mu, log_sigma = hidden.chunk(2, dim=-1)
        # sigma floor of 0.1 prevents -inf NLL
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)
        dist = Normal(loc=mu, scale=sigma)
        return dist, mu, sigma
