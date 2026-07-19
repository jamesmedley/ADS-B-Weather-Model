"""
module.py — Attentive Neural Process building blocks.

Pipeline: concat(x, y) -> BatchMLP -> self-attention -> aggregate -> ...
  Latent path:   self-attention -> mean -> mu/log_sigma -> z
  Deterministic: self-attention + cross-attention -> r*
  Decoder:       concat(r*, z, target_x) -> MLP -> mu, sigma
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class BatchMLP(nn.Module):
    """MLP applied to the last axis of a [B, n, d_in] tensor."""

    def __init__(self, in_size, output_sizes):
        super().__init__()
        layers = []
        current = in_size
        for size in output_sizes[:-1]:
            layers.append(nn.Linear(current, size))
            layers.append(nn.ReLU())
            current = size
        layers.append(nn.Linear(current, output_sizes[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        B, n, _ = x.shape
        out = self.net(x.reshape(B * n, -1))
        return out.reshape(B, n, -1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

def dot_product_attention(q, k, v, normalise, mask=None):
    """Scaled dot-product attention. mask=True means real data."""
    d_k = q.shape[-1]
    scale = math.sqrt(d_k)
    unnorm_weights = torch.einsum('bmd,bnd->bmn', q, k) / scale

    if mask is not None:
        key_mask = mask.unsqueeze(1)
        if normalise:
            unnorm_weights = unnorm_weights.masked_fill(
                ~key_mask, float('-inf')
            )
        else:
            unnorm_weights = unnorm_weights.masked_fill(~key_mask, -1e9)

    if normalise:
        weights = torch.softmax(unnorm_weights, dim=-1)
    else:
        weights = torch.sigmoid(unnorm_weights)
        if mask is not None:
            weights = weights * key_mask

    rep = torch.einsum('bmn,bnd->bmd', weights, v)
    return rep


class MultiheadAttentionModule(nn.Module):
    """Multi-head cross-attention with per-head 1x1 conv projections
    (summed, not concatenated)."""

    def __init__(self, d_k, d_v, num_heads=8):
        super().__init__()
        self._num_heads = num_heads
        self._d_v = d_v
        self.head_size = d_v // num_heads
        self.wq = nn.Linear(d_k, d_v, bias=False)
        self.wk = nn.Linear(d_k, d_v, bias=False)
        self.wv = nn.Linear(d_v, d_v, bias=False)
        self.wo = nn.Linear(d_v, d_v, bias=False)

    def forward(self, q, k, v, mask=None):
        B, m, _ = q.shape
        n = k.shape[1]
        H, hs = self._num_heads, self.head_size

        q_h = self.wq(q).view(B, m, H, hs).transpose(1, 2)
        k_h = self.wk(k).view(B, n, H, hs).transpose(1, 2)
        v_h = self.wv(v).view(B, n, H, hs).transpose(1, 2)

        scale = hs ** 0.5
        attn = torch.einsum('bhmd,bhnd->bhmn', q_h, k_h) / scale
        if mask is not None:
            attn = attn.masked_fill(~mask[:, None, None, :], float('-inf'))
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum('bhmn,bhnd->bhmd', attn, v_h)
        out = out.transpose(1, 2).reshape(B, m, H * hs)
        return self.wo(out)


# ---------------------------------------------------------------------------
# Self-attention blocks
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: self-attention + position-wise FFN."""

    def __init__(self, d, num_heads=8, ffn_expansion=2, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = MultiheadAttentionModule(d_k=d, d_v=d, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * ffn_expansion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d * ffn_expansion, d),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        normed = self.norm1(x)
        x = x + self.drop(self.attn(q=normed, k=normed, v=normed, mask=mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class SelfAttentionStack(nn.Module):
    """Stack of TransformerBlocks with a final LayerNorm."""

    def __init__(self, d, num_layers=2, num_heads=8,
                 ffn_expansion=2, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d, num_heads=num_heads,
                             ffn_expansion=ffn_expansion,
                             dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d)

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Attention wrapper
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Wraps key/query representation and attention type choice."""

    def __init__(self, x_size, rep, output_sizes, att_type, d_v=None,
                 scale=1., normalise=True, num_heads=8):
        super().__init__()
        self._rep = rep
        self._type = att_type
        self._scale = scale
        self._normalise = normalise
        self._num_heads = num_heads

        if rep == 'mlp':
            assert output_sizes is not None
            self.mlp = BatchMLP(x_size, output_sizes)
            d_k = output_sizes[-1]
        elif rep == 'identity':
            self.mlp = None
            d_k = x_size
        else:
            raise NameError("'rep' not among ['identity', 'mlp']")

        if att_type == 'multihead':
            assert d_v is not None
            self.multihead = MultiheadAttentionModule(d_k, d_v, num_heads)
        else:
            self.multihead = None

    def forward(self, x1, x2, r, mask=None):
        """x1=context x, x2=target x, r=context representations.
        mask=True for real keys."""
        if self._rep == 'identity':
            k, q = x1, x2
        elif self._rep == 'mlp':
            k = self.mlp(x1)
            q = self.mlp(x2)
        else:
            raise NameError("'rep' not among ['identity', 'mlp']")

        if self._type == 'dot_product':
            rep = dot_product_attention(q, k, r, self._normalise, mask=mask)
        elif self._type == 'multihead':
            rep = self.multihead(q, k, r, mask=mask)
        else:
            raise NameError("'att_type' not among ['uniform','laplace',"
                            "'dot_product','multihead']")

        return rep


# ---------------------------------------------------------------------------
# Encoders & Decoder
# ---------------------------------------------------------------------------

class DeterministicEncoder(nn.Module):
    """concat(x,y) -> MLP -> self-attention -> cross-attention -> r*"""

    def __init__(self, x_size, y_size, output_sizes, attention,
                 num_heads=8, num_layers=2, ffn_expansion=2, dropout=0.0):
        super().__init__()
        self._attention = attention
        self.encoder_mlp = BatchMLP(x_size + y_size, output_sizes)
        self.self_attn = SelfAttentionStack(
            output_sizes[-1], num_layers=num_layers, num_heads=num_heads,
            ffn_expansion=ffn_expansion, dropout=dropout)

    def forward(self, context_x, context_y, target_x, context_mask=None):
        encoder_input = torch.cat([context_x, context_y], dim=-1)
        hidden = self.encoder_mlp(encoder_input)
        hidden = self.self_attn(hidden, mask=context_mask)
        return self._attention(context_x, target_x, hidden, mask=context_mask)


class LatentEncoder(nn.Module):
    """concat(x,y) -> MLP -> self-attention -> mean -> mu/log_sigma -> z"""

    def __init__(self, x_size, y_size, output_sizes, num_latents,
                 num_heads=8, num_layers=2, ffn_expansion=2, dropout=0.0):
        super().__init__()
        self.encoder_mlp = BatchMLP(x_size + y_size, output_sizes)
        self.self_attn = SelfAttentionStack(
            output_sizes[-1], num_layers=num_layers, num_heads=num_heads,
            ffn_expansion=ffn_expansion, dropout=dropout)

        d = output_sizes[-1]
        penultimate_size = int((d + num_latents) / 2)
        self.penultimate_layer = nn.Linear(d, penultimate_size)
        self.mean_layer = nn.Linear(penultimate_size, num_latents)
        self.std_layer = nn.Linear(penultimate_size, num_latents)

    def forward(self, x, y, mask=None):
        encoder_input = torch.cat([x, y], dim=-1)
        hidden = self.encoder_mlp(encoder_input)
        hidden = self.self_attn(hidden, mask=mask)

        if mask is None:
            hidden = hidden.mean(dim=1)
        else:
            mask_f = mask.unsqueeze(-1).to(hidden.dtype)
            counts = mask_f.sum(dim=1).clamp(min=1.0)
            hidden = (hidden * mask_f).sum(dim=1) / counts

        hidden = F.relu(self.penultimate_layer(hidden))
        mu = self.mean_layer(hidden)
        log_sigma = self.std_layer(hidden)
        sigma = 0.05 + F.softplus(log_sigma)

        return Normal(loc=mu, scale=sigma)


class Decoder(nn.Module):
    """concat(representation, target_x) -> MLP -> mu, sigma"""

    def __init__(self, x_size, representation_size, output_sizes, dropout=0.0):
        super().__init__()
        layers = []
        current = x_size + representation_size
        for size in output_sizes[:-1]:
            layers.append(nn.Linear(current, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current = size
        layers.append(nn.Linear(current, output_sizes[-1]))
        self.decoder_mlp = nn.Sequential(*layers)

    def forward(self, representation, target_x):
        hidden = torch.cat([representation, target_x], dim=-1)
        hidden = self.decoder_mlp(hidden)
        mu, log_sigma = hidden.chunk(2, dim=-1)
        # sigma floor of 0.1 prevents -inf NLL
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)
        dist = Normal(loc=mu, scale=sigma)
        return dist, mu, sigma
