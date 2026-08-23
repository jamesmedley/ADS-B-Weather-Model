"""
Inference tests: WindPredictor end-to-end on a synthetic checkpoint,
uncertainty decomposition, and tensor conversion helpers.
"""

import math

import numpy as np
import pytest
import torch

from wind_map.infer import (
    WindPredictor, compute_uncertainty_components,
    load_model_checkpoint,
)
from wind_map.network import LatentModel
from wind_map.preprocess import LEGACY_PARAMS

pytestmark = pytest.mark.integration

SEED = 23


@pytest.fixture()
def checkpoint(tmp_path):
    torch.manual_seed(SEED)
    model = LatentModel(
        num_hidden=8, num_latents=4, num_heads=4,
        latent_layers=1, deterministic_layers=1,
        num_decoder_layers=1, dropout=0.0,
        use_dist_bias=False)
    path = tmp_path / "best_model.pth.tar"
    torch.save({
        'epoch': 4,
        'model': model.state_dict(),
        'val_loss': 0.75,
        'hparams': {
            'num_hidden': 8, 'num_latents': 4,
            'latent_layers': 1, 'deterministic_layers': 1,
            'dropout': 0.0, 'use_dist_bias': False,
            'num_decoder_layers': 1,
        },
        'norm_params': {
            'centre_lat': 51.4, 'centre_lon': -1.2,
            'range_km': 120.0, 'max_alt_ft': 40000.0,
            'wind_speed_kt_mean': 30.0,
            'wind_speed_kt_std': 15.0,
            'log_speed_mean': math.log(31.0),
            'log_speed_std': 0.5,
        },
    }, path)
    return str(path), model


def make_context(n=6, seed=1):
    rng = np.random.default_rng(seed)
    return [{"lat": float(51.4 + rng.uniform(-0.5, 0.5)),
             "lon": float(-1.2 + rng.uniform(-0.5, 0.5)),
             "alt_ft": float(rng.uniform(1000, 39000)),
             "wind_dir": float(rng.uniform(0, 360)),
             "wind_speed": float(rng.uniform(2, 70))}
            for _ in range(n)]


def make_queries(n=4):
    return [{"lat": 51.4 + 0.02 * i,
             "lon": -1.2 + 0.03 * i,
             "alt_ft": 5000.0 * (i + 1)}
            for i in range(n)]


def test_predictor_output_contract(checkpoint):
    ckpt_path, _ = checkpoint
    predictor = WindPredictor(ckpt_path, device='cpu')
    result = predictor.predict(make_context(), make_queries(),
                               n_samples=3)
    expected_keys = {
        'wind_dir_deg', 'wind_speed_kt',
        'wind_dir_std', 'wind_speed_std',
        'epistemic_dir_std', 'epistemic_speed_std',
        'aleatoric_dir_std', 'aleatoric_speed_std',
    }
    assert set(result) == expected_keys
    dirs = np.asarray(result['wind_dir_deg'])
    speeds = np.asarray(result['wind_speed_kt'])
    assert ((dirs >= 0) & (dirs < 360)).all()
    assert (speeds > -1e-6).all()
    for key in ('wind_dir_std', 'wind_speed_std'):
        std = np.asarray(result[key])
        assert (std >= 0).all()
        assert np.isfinite(std).all()


def test_predictor_total_uncertainty_combines_components(checkpoint):
    ckpt_path, _ = checkpoint
    predictor = WindPredictor(ckpt_path, device='cpu')
    result = predictor.predict(make_context(), make_queries(),
                               n_samples=3)
    for total, epi, ale in [
        ('wind_dir_std', 'epistemic_dir_std',
         'aleatoric_dir_std'),
        ('wind_speed_std', 'epistemic_speed_std',
         'aleatoric_speed_std'),
    ]:
        combined = np.sqrt(np.asarray(result[epi]) ** 2
                           + np.asarray(result[ale]) ** 2)
        assert np.allclose(result[total], combined, atol=1e-8)


def test_predictor_deterministic_given_seed(checkpoint):
    ckpt_path, _ = checkpoint
    predictor = WindPredictor(ckpt_path, device='cpu')
    ctx, qry = make_context(), make_queries()
    torch.manual_seed(99)
    a = predictor.predict(ctx, qry, n_samples=5)
    torch.manual_seed(99)
    b = predictor.predict(ctx, qry, n_samples=5)
    for key in a:
        assert np.allclose(a[key], b[key])


def test_predictor_falls_back_to_legacy_params(tmp_path):
    torch.manual_seed(SEED)
    model = LatentModel(num_hidden=8, num_latents=4, num_heads=4,
                        latent_layers=1, deterministic_layers=1,
                        num_decoder_layers=1)
    path = tmp_path / "legacy.pth.tar"
    torch.save({'model': model.state_dict(),
                'hparams': {
                    'num_hidden': 8, 'num_latents': 4,
                    'latent_layers': 1,
                    'deterministic_layers': 1,
                    'dropout': 0.0, 'use_dist_bias': False,
                    'num_decoder_layers': 1,
                }}, path)
    predictor = WindPredictor(str(path), device='cpu')
    assert predictor.params is LEGACY_PARAMS


def test_compute_uncertainty_components_values(params):
    n_z, n_pts = 3, 5
    mu_stack = np.zeros((n_z, n_pts, 3))
    mu_stack[:, :, 0] = 0.6   # sin component identical across draws
    mu_stack[:, :, 1] = 0.8
    sigma_stack = np.full((n_z, n_pts, 3), 0.2)

    out = compute_uncertainty_components(
        mu_stack, sigma_stack, params, mc_samples=5, seed=0)

    for key, val in out.items():
        assert val.shape == (n_pts,)
        assert (np.asarray(val) >= 0).all()

    # Identical z-draw means -> zero epistemic spread.
    assert np.allclose(out['epistemic_speed_std'], 0.0, atol=1e-9)

    # Aleatoric speed follows the exact Jacobian of the log transform.
    jac = params.log_speed_std * math.exp(
        params.log_speed_mean + 0.0 * params.log_speed_std)
    expected = math.sqrt(((sigma_stack[..., 2] * jac) ** 2).mean())
    assert np.allclose(out['aleatoric_speed_std'], expected,
                       atol=1e-9)


def test_log_speed_negative_logits_warn(capsys, params):
    from wind_map.infer import _log_speed_to_kt
    vals = _log_speed_to_kt(np.array([-50.0, 0.0]), params)
    captured = capsys.readouterr().out
    assert "Warning" in captured
    assert vals[0] < 0


def test_observations_to_tensors_shape_and_values(params):
    from wind_map.infer import observations_to_tensors
    obs = [{"lat": params.centre_lat, "lon": params.centre_lon,
            "alt_ft": 0.0, "wind_dir": 90.0, "wind_speed": 20.0}]
    x, y = observations_to_tensors(obs, torch.device('cpu'), params)
    assert x.shape == (1, 1, 3)
    assert y.shape == (1, 1, 3)
    assert x[0, 0].tolist() == [pytest.approx(0.0),
                                pytest.approx(0.0),
                                pytest.approx(0.0)]
    assert y[0, 0, 0] == pytest.approx(1.0, abs=1e-6)


def test_loaded_model_matches_saved_predictions(checkpoint):
    ckpt_path, model = checkpoint
    loaded, _ = load_model_checkpoint(ckpt_path,
                                      torch.device('cpu'))
    g = torch.Generator().manual_seed(SEED)
    cx = torch.randn(1, 4, 3, generator=g)
    cy = torch.randn(1, 4, 3, generator=g)
    tx = torch.randn(1, 3, 3, generator=g)
    torch.manual_seed(SEED)
    with torch.no_grad():
        mu_a = model(cx, cy, tx)[0]
    torch.manual_seed(SEED)
    with torch.no_grad():
        mu_b = loaded(cx, cy, tx)[0]
    assert torch.allclose(mu_a, mu_b, atol=1e-7)
