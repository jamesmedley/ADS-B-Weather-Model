"""
Checkpoint saving/loading tests.

A checkpoint must round-trip: weights identical after reload,
hyperparameters recovered automatically, and a helpful error when
hyperparameters are missing.
"""

import os

import pytest
import torch

from wind_map.infer import load_model_checkpoint
from wind_map.network import LatentModel
from wind_map.train import _save_checkpoint

pytestmark = pytest.mark.unit

SEED = 17


def build_model():
    return LatentModel(
        num_hidden=8, num_latents=4, num_heads=4,
        latent_layers=1, deterministic_layers=1,
        num_decoder_layers=1, dropout=0.1,
        free_bits=0.01, use_dist_bias=True)


def hparams_dict():
    return {
        'num_hidden': 8, 'num_latents': 4,
        'latent_layers': 1, 'deterministic_layers': 1,
        'dropout': 0.1, 'use_dist_bias': True,
        'num_decoder_layers': 1, 'lr': 0.001,
        'batch_size': 2, 'seed': SEED,
    }


def test_atomic_save_writes_file_and_leaves_no_tmp(tmp_path):
    model = build_model()
    path = str(tmp_path / "best_model.pth.tar")
    ckpt = {'epoch': 3, 'model': model.state_dict(),
            'hparams': hparams_dict()}
    _save_checkpoint(path, ckpt)
    assert os.path.exists(path)
    assert not os.path.exists(path + '.tmp')
    loaded = torch.load(path, weights_only=False)
    assert loaded['epoch'] == 3


def test_checkpoint_roundtrip_reproduces_predictions(tmp_path):
    model = build_model()
    model.eval()
    path = str(tmp_path / "ckpt.pth.tar")
    _save_checkpoint(path, {
        'epoch': 1, 'model': model.state_dict(),
        'val_loss': 0.25, 'hparams': hparams_dict(),
        'norm_params': {
            'centre_lat': 51.4, 'centre_lon': -1.2,
            'range_km': 120.0, 'max_alt_ft': 40000.0,
            'wind_speed_kt_mean': 30.0,
            'wind_speed_kt_std': 15.0,
            'log_speed_mean': 3.4, 'log_speed_std': 0.5,
        },
    })

    reloaded, ckpt = load_model_checkpoint(path, torch.device('cpu'))
    assert ckpt['val_loss'] == pytest.approx(0.25)
    reloaded.eval()

    g = torch.Generator().manual_seed(SEED)
    cx, cy, tx, ty = [
        torch.randn(*shape, generator=g)
        for shape in ((2, 5, 3), (2, 5, 3), (2, 6, 3),
                      (2, 6, 3))
    ]
    # Latent sampling is stochastic, so each forward must start
    # from the same RNG state to be comparable.
    torch.manual_seed(SEED)
    with torch.no_grad():
        out_a = model(cx, cy, tx, ty)
    torch.manual_seed(SEED)
    with torch.no_grad():
        out_b = reloaded(cx, cy, tx, ty)
    mu_a, sig_a, _, _, loss_a = out_a
    mu_b, sig_b, _, _, loss_b = out_b
    assert torch.allclose(mu_a, mu_b, atol=1e-6)
    assert torch.allclose(sig_a, sig_b, atol=1e-6)
    assert torch.allclose(loss_a, loss_b, atol=1e-6)


def test_explicit_overrides_beat_checkpoint_hparams(tmp_path):
    model = build_model()
    path = str(tmp_path / "ckpt.pth.tar")
    _save_checkpoint(path, {
        'epoch': 1, 'model': model.state_dict(),
        'hparams': hparams_dict(),
    })
    _, ckpt = load_model_checkpoint(
        path, torch.device('cpu'), dropout=0.5)
    assert ckpt['hparams']['dropout'] == 0.1


def test_missing_hparams_raises_helpful_error(tmp_path):
    model = build_model()
    path = str(tmp_path / "no_hparams.pth.tar")
    torch.save({'model': model.state_dict()}, path)
    with pytest.raises(ValueError, match="num_hidden"):
        load_model_checkpoint(path, torch.device('cpu'))
