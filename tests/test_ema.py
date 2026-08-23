"""
EMA (flat-buffer exponential moving average) tests.
"""

import pytest
import torch

from wind_map.train import EMA

pytestmark = pytest.mark.unit


def small_model(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.Tanh(),
        torch.nn.Linear(4, 2))


def flat_params(model):
    return torch.cat(
        [p.data.reshape(-1) for p in model.parameters()])


def test_shadow_starts_as_copy_of_parameters():
    m = small_model()
    ema = EMA(m, decay=0.9)
    assert torch.equal(ema.shadow, flat_params(m))
    assert ema.shadow.numel() == sum(
        p.numel() for p in m.parameters())


def test_decay_zero_tracks_parameters_exactly():
    m = small_model()
    ema = EMA(m, decay=0.0)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    ema.update(m)
    assert torch.equal(ema.shadow, flat_params(m))


def test_decay_one_keeps_shadow_frozen():
    m = small_model()
    ema = EMA(m, decay=1.0)
    before = ema.shadow.clone()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(generator=torch.Generator().manual_seed(3))
    ema.update(m)
    assert torch.equal(ema.shadow, before)


def test_update_matches_manual_ema_formula():
    decay = 0.9
    m = small_model(seed=1)
    ema = EMA(m, decay=decay)
    s0 = flat_params(m).clone()

    with torch.no_grad():
        for p in m.parameters():
            p.fill_(0.5)
    ema.update(m)
    expected1 = decay * s0 + (1 - decay) * flat_params(m)

    with torch.no_grad():
        for p in m.parameters():
            p.fill_(-0.25)
    ema.update(m)
    expected2 = decay * expected1 + (1 - decay) * flat_params(m)

    assert torch.allclose(ema.shadow, expected2, atol=1e-6)


def test_apply_shadow_restore_roundtrip():
    m = small_model(seed=2)
    ema = EMA(m, decay=0.99)

    with torch.no_grad():
        for p in m.parameters():
            p.normal_()
    modified = flat_params(m).clone()

    ema.apply_shadow(m)
    assert torch.equal(flat_params(m), ema.shadow)
    assert not torch.equal(flat_params(m), modified)

    ema.restore(m)
    assert torch.equal(flat_params(m), modified)
