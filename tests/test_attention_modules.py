"""
Attention building block tests: Attention, LatentEncoder,
DeterministicEncoder, Decoder.

Focus: masking correctness (padded context must never influence
outputs), positive scales, the decoder sigma floor, and finite outputs.
"""

import pytest
import torch

from wind_map.module import (
    Attention, Decoder, DeterministicEncoder, LatentEncoder,
)

pytestmark = pytest.mark.unit


def make_inputs(b=2, n=6, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(b, n, d, generator=g),
            torch.randn(b, n, d, generator=g),
            torch.randn(b, n, d, generator=g))


def test_attention_output_and_attn_shapes():
    k, v, q = make_inputs()
    attn = Attention(num_hidden=8, h=4, dropout=0.0)
    out, w = attn(k, v, q)
    assert out.shape == q.shape
    assert w.shape == (2, 4, 6, 6)


def test_attention_mask_blocks_padded_keys():
    k, v, q = make_inputs()
    attn = Attention(num_hidden=8, h=4, dropout=0.0)
    mask = torch.tensor([
        [True] * 5 + [False],
        [True] * 3 + [False] * 3,
    ])
    out, w = attn(k, v, q, mask=mask)
    assert torch.isfinite(out).all()
    for b, n_valid in [(0, 5), (1, 3)]:
        blocked = w[b, :, :, n_valid:]
        assert torch.all(blocked == 0)
        kept = w[b, :, :, :n_valid].sum(dim=-1)
        assert torch.allclose(
            kept, torch.ones_like(kept), atol=1e-5)


def test_attention_dist_bias_changes_result():
    k, v, q = make_inputs()
    attn = Attention(num_hidden=8, h=4, dropout=0.0,
                     use_dist_bias=True)
    attn.eval()
    # A uniform bias cancels in the softmax; only a bias that varies
    # across keys can reweight attention.
    key_varying = torch.linspace(0.0, 20.0, 6).expand(2, 6, 6)
    with torch.no_grad():
        base, _ = attn(k, v, q)
        zero_bias = torch.zeros(2, 6, 6)
        same, _ = attn(k, v, q, dist_bias=zero_bias)
        biased, w_biased = attn(k, v, q,
                                dist_bias=key_varying)
    assert torch.allclose(base, same, atol=1e-6)
    assert not torch.allclose(base, biased, atol=1e-4)


def test_latent_encoder_returns_positive_scale():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 5, 3, generator=g)
    y = torch.randn(2, 5, 3, generator=g)
    enc = LatentEncoder(num_hidden=8, num_latents=4,
                        num_heads=4, layers=1, dropout=0.0)
    enc.eval()
    dist = enc(x, y)
    assert dist.loc.shape == (2, 4)
    assert dist.scale.shape == (2, 4)
    assert (dist.scale > 0).all()


def test_latent_encoder_mask_ignores_padded_rows():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(1, 5, 3, generator=g)
    y = torch.randn(1, 5, 3, generator=g)
    enc = LatentEncoder(num_hidden=8, num_latents=4,
                        num_heads=4, layers=1, dropout=0.0)
    enc.eval()
    mask = torch.tensor([[True] * 4 + [False]])
    d1 = enc(x, y, mask=mask)

    x_polluted = x.clone()
    y_polluted = y.clone()
    x_polluted[0, 4] = 123.0
    y_polluted[0, 4] = -321.0
    d2 = enc(x_polluted, y_polluted, mask=mask)

    assert torch.allclose(d1.loc, d2.loc, atol=1e-5)
    assert torch.allclose(d1.scale, d2.scale, atol=1e-5)


def test_deterministic_encoder_shape_and_context_masking():
    g = torch.Generator().manual_seed(3)
    cx = torch.randn(2, 5, 3, generator=g)
    cy = torch.randn(2, 5, 3, generator=g)
    tx = torch.randn(2, 7, 3, generator=g)
    enc = DeterministicEncoder(num_hidden=8, num_heads=4,
                               layers=1, dropout=0.0)
    enc.eval()
    r = enc(cx, cy, tx)
    assert r.shape == (2, 7, 8)

    mask = torch.tensor([[True] * 3 + [False] * 2] * 2)
    r1 = enc(cx, cy, tx, context_mask=mask)
    cx2, cy2 = cx.clone(), cy.clone()
    cx2[:, 3:] = 999.0
    cy2[:, 3:] = -999.0
    r2 = enc(cx2, cy2, tx, context_mask=mask)
    assert torch.allclose(r1, r2, atol=1e-5)


def test_decoder_sigma_floor():
    g = torch.Generator().manual_seed(4)
    rep = torch.zeros(2, 4, 12)
    tx = torch.randn(2, 4, 3, generator=g) * 100.0
    dec = Decoder(x_size=3, representation_size=12,
                  output_sizes=[8, 6], target_hidden=8,
                  dropout=0.0)
    dec.eval()
    dist, mu, sigma = dec(rep, tx)
    assert mu.shape == sigma.shape == (2, 4, 3)
    assert (sigma >= 0.1).all()
    assert torch.isfinite(mu).all()
    assert torch.isfinite(sigma).all()
