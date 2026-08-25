"""
LatentModel (full ANP) tests.

Pinned behaviours: output shapes, sigma floor, finite losses, gradient
flow to every parameter, padding invariance of the masked ELBO, free
bits clamping, and KL-weight linearity.
"""

import pytest
import torch

from wind_map.network import LatentModel

pytestmark = pytest.mark.unit

SEED = 7


def rand_inputs(b=2, n_ctx=5, n_tgt=6, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    cx = torch.randn(b, n_ctx, 3, generator=g)
    cy = torch.randn(b, n_ctx, 2, generator=g)
    tx = torch.randn(b, n_tgt, 3, generator=g)
    ty = torch.randn(b, n_tgt, 2, generator=g)
    return cx, cy, tx, ty


def test_inference_forward_shapes_no_loss(tiny_model):
    cx, cy, tx, _ = rand_inputs()
    mu, sigma, kl, kl_dim, loss = tiny_model(cx, cy, tx)
    assert mu.shape == (2, 6, 2)
    assert sigma.shape == (2, 6, 2)
    assert kl is None and kl_dim is None and loss is None
    assert (sigma >= 0.1).all()
    assert torch.isfinite(mu).all()
    assert torch.isfinite(sigma).all()


def test_training_loss_finite_scalar(tiny_model):
    cx, cy, tx, ty = rand_inputs()
    *_, loss = tiny_model(cx, cy, tx, ty)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_gradients_flow_to_every_parameter(tiny_model):
    cx, cy, tx, ty = rand_inputs()
    *_, loss = tiny_model(cx, cy, tx, ty)
    loss.backward()
    no_grad = [n for n, p in tiny_model.named_parameters()
               if p.grad is None]
    zero_or_nan = [
        n for n, p in tiny_model.named_parameters()
        if p.grad is not None
        and (not torch.isfinite(p.grad).all()
             or p.grad.abs().sum() == 0)
    ]
    assert no_grad == [], f"detached parameters: {no_grad}"
    assert zero_or_nan == [], f"dead gradients: {zero_or_nan}"


def test_padded_targets_do_not_affect_valid_outputs(tiny_model):
    """Masked-loss invariance: garbage in padded rows must change
    neither the valid-region predictions nor the loss."""
    g = torch.Generator().manual_seed(SEED + 1)
    b, nt_true, nt_pad = 2, 4, 3
    cx = torch.randn(b, 5, 3, generator=g)
    cy = torch.randn(b, 5, 2, generator=g)
    tx = torch.randn(b, nt_true + nt_pad, 3, generator=g)
    ty = torch.randn(b, nt_true + nt_pad, 2, generator=g)

    mask = torch.zeros(b, nt_true + nt_pad, dtype=torch.bool)
    mask[:, :nt_true] = True

    outs = []
    for pad_value in (0.0, 500.0):
        ty_variant = ty.clone()
        ty_variant[:, nt_true:] = pad_value
        tx_variant = tx.clone()
        tx_variant[:, nt_true:] = -pad_value
        torch.manual_seed(123)
        mu, _, _, _, loss = tiny_model(
            cx, cy, tx_variant, ty_variant,
            target_mask=mask)
        outs.append((mu, loss))

    (mu_a, loss_a), (mu_b, loss_b) = outs
    assert torch.allclose(mu_a[:, :nt_true], mu_b[:, :nt_true],
                          atol=1e-5)
    assert torch.allclose(loss_a, loss_b, atol=1e-5)


def test_padded_context_does_not_change_predictions(tiny_model):
    g = torch.Generator().manual_seed(SEED + 2)
    cx = torch.randn(1, 4 + 3, 3, generator=g)
    cy = torch.randn(1, 4 + 3, 2, generator=g)
    tx = torch.randn(1, 6, 3, generator=g)
    cmask = torch.tensor([[True] * 4 + [False] * 3])
    torch.manual_seed(5)
    mu_ref, _, _, _, _ = tiny_model(cx, cy, tx,
                                    context_mask=cmask)
    cx_bad, cy_bad = cx.clone(), cy.clone()
    cx_bad[:, 4:] = 777.0
    cy_bad[:, 4:] = -777.0
    torch.manual_seed(5)
    mu_bad, _, _, _, _ = tiny_model(cx_bad, cy_bad, tx,
                                    context_mask=cmask)
    assert torch.allclose(mu_ref, mu_bad, atol=1e-5)


def test_free_bits_clamps_kl_per_dimension():
    model = LatentModel(num_hidden=8, num_latents=4, num_heads=4,
                        latent_layers=1, deterministic_layers=1,
                        num_decoder_layers=1, free_bits=0.5)
    cx, cy, tx, ty = rand_inputs()
    torch.manual_seed(SEED)
    _, _, kl, kl_per_dim, _ = model(cx, cy, tx, ty)
    assert (kl >= 0.5 - 1e-6).all()


def test_kl_weight_scales_loss_linearly(tiny_model):
    cx, cy, tx, ty = rand_inputs()

    def fwd(weight):
        torch.manual_seed(SEED)
        return tiny_model(cx, cy, tx, ty,
                          kl_weight=weight)[-1].detach()

    l0, l_half, l_one = fwd(0.0), fwd(0.5), fwd(1.0)
    assert l_one > l0
    assert (l0 + l_one) / 2 == pytest.approx(l_half.item(),
                                             abs=1e-5)


def test_same_seed_forward_is_reproducible():
    """Latent z is sampled stochastically even in eval mode, so the
    pinned determinism contract is: identical seed -> identical
    outputs (this is what checkpoint regression checks rely on)."""
    torch.manual_seed(0)
    model = LatentModel(num_hidden=8, num_heads=4,
                        latent_layers=1, deterministic_layers=1,
                        num_decoder_layers=1, dropout=0.5)
    cx, cy, tx, _ = rand_inputs()
    torch.manual_seed(11)
    a = model(cx, cy, tx)[0]
    torch.manual_seed(12)
    b = model(cx, cy, tx)[0]
    assert not torch.equal(a, b), \
        "stochastic latents should vary with seed"
    torch.manual_seed(11)
    c = model(cx, cy, tx)[0]
    assert torch.equal(a, c)

    # Eval mode must be equally reproducible run-to-run.
    model.eval()
    torch.manual_seed(11)
    d = model(cx, cy, tx)[0]
    torch.manual_seed(11)
    e = model(cx, cy, tx)[0]
    assert torch.equal(d, e)


def test_num_hidden_must_be_divisible_by_heads():
    with pytest.raises(AssertionError):
        LatentModel(num_hidden=7, num_heads=4)
