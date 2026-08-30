"""
Exact GP baseline tests: kernel maths, regressor fit/predict,
hyperparameter fitting, and coordinate conversion.
"""

import math

import numpy as np
import pytest

from wind_map.gp import (
    GaussianProcessPredictor, GaussianProcessRegressor,
    _matern52_corr, cache_x_to_gp_x, fit_hyperparameters,
    norm_params_from_dict,
)
from wind_map.preprocess import KM_PER_DEG_LAT, encode_wind

pytestmark = pytest.mark.unit


def gp_inputs(n=8, seed=0):
    rng = np.random.default_rng(seed)
    x = np.column_stack([
        rng.uniform(-60, 60, n),
        rng.uniform(-60, 60, n),
        rng.uniform(0.1, 10.0, n),
    ])
    y = np.column_stack([
        np.sin(x[:, 0] / 40.0),
        np.cos(x[:, 1] / 50.0),
    ])
    return x, y


def make_reg(noise=1e-4):
    return GaussianProcessRegressor(
        lengthscales=[50.0, 50.0, 2.0],
        amplitudes=[0.8, 0.9],
        noises=[noise, noise])


def test_matern52_correlation_properties():
    assert _matern52_corr(0.0) == pytest.approx(1.0)
    r = np.linspace(0.01, 5.0, 50)
    corr = _matern52_corr(r ** 2)
    assert (np.diff(corr) < 0).all()
    assert _matern52_corr(25.0) < 1e-3
    expected = ((1 + math.sqrt(5) + 5.0 / 3.0)
                * math.exp(-math.sqrt(5)))
    assert _matern52_corr(1.0) == pytest.approx(expected)


def test_gpr_predict_before_fit_raises():
    reg = make_reg()
    with pytest.raises(RuntimeError):
        reg.predict(np.zeros((3, 3)))


def test_gpr_interpolates_training_points_with_low_noise():
    x, y = gp_inputs()
    reg = make_reg(noise=1e-6).fit(x, y)
    mean, var_total, var_latent = reg.predict(x)
    assert mean.shape == (8, 2)
    assert np.allclose(mean, y, atol=1e-3)


def test_gpr_variance_bounds():
    x, y = gp_inputs()
    reg = make_reg(noise=0.05).fit(x, y)
    _, var_total, var_latent = reg.predict(x)
    assert (var_latent >= 0).all()
    assert (var_total >= var_latent - 1e-12).all()
    expected_extra = 0.05 ** 2
    assert np.allclose(var_total - var_latent,
                       expected_extra, atol=1e-9)


def test_ard_distance_symmetry():
    from wind_map.gp import _ard_squared_dist
    x, _ = gp_inputs()
    inv_l2 = np.array([1.0, 1.0, 1.0])
    d = _ard_squared_dist(x, x, inv_l2)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)


def test_norm_params_from_dict_new_and_legacy_conventions(params):
    new_style = {
        "centre_lat": params.centre_lat,
        "centre_lon": params.centre_lon,
        "range_km": params.range_km,
        "max_alt_ft": params.max_alt_ft,
        "u_mean": params.u_mean,
        "u_std": params.u_std,
        "v_mean": params.v_mean,
        "v_std": params.v_std,
    }
    p_new = norm_params_from_dict(new_style)
    assert p_new.u_mean == params.u_mean
    assert p_new.v_std == params.v_std

    legacy_a = dict(new_style)
    del legacy_a["u_mean"], legacy_a["u_std"]
    del legacy_a["v_mean"], legacy_a["v_std"]
    legacy_a["wind_speed_kt_mean"] = 30.0
    legacy_a["wind_speed_kt_std"] = 15.0
    legacy_b = dict(new_style)
    del legacy_b["u_mean"], legacy_b["u_std"]
    del legacy_b["v_mean"], legacy_b["v_std"]
    legacy_b["wind_speed_mean_kt"] = 30.0
    legacy_b["wind_speed_std_kt"] = 15.0
    p_la = norm_params_from_dict(legacy_a)
    p_lb = norm_params_from_dict(legacy_b)
    assert p_la.u_mean == p_lb.u_mean
    assert p_la.u_std == p_lb.u_std

    with pytest.raises(KeyError):
        norm_params_from_dict({"centre_lat": 1.0})


def test_cache_x_to_gp_x_inverts_normalisation(cache):
    from wind_map.preprocess import normalise_coords
    obs = cache.raw[cache.ids[0]][0]
    lat_n, lon_n, alt_n = normalise_coords(
        obs["lat"], obs["lon"], obs["alt_ft"], cache.params)
    km = cache_x_to_gp_x(
        np.array([[lat_n, lon_n, alt_n]]), cache.params)
    lat_km = (obs["lat"] - cache.params.centre_lat) * KM_PER_DEG_LAT
    lon_km = ((obs["lon"] - cache.params.centre_lon)
              * cache.params.km_per_deg_lon)
    alt_km = obs["alt_ft"] * 0.0003048
    assert km[0, 0] == pytest.approx(lat_km, abs=1e-4)
    assert km[0, 1] == pytest.approx(lon_km, abs=1e-4)
    assert km[0, 2] == pytest.approx(alt_km, rel=1e-5)


@pytest.mark.integration
def test_fit_hyperparameters_short_run_returns_sane_values():
    rng = np.random.default_rng(5)
    snapshots = []
    for _ in range(3):
        x, y = gp_inputs(n=5, seed=int(rng.integers(1000)))
        snapshots.append((x.astype(np.float64),
                          y.astype(np.float64)))
    ls, amps, noises, mll = fit_hyperparameters(
        snapshots, n_steps=5, lr=0.02, verbose=False)
    assert ls.shape == (3,)
    assert amps.shape == noises.shape == (2,)
    assert (ls > 0).all() and (amps > 0).all()
    assert (noises >= 0.01).all()
    assert np.isfinite(mll)


@pytest.mark.integration
def test_gp_predictor_api_parity_and_determinism(tmp_path, cache):
    hp = {
        "lengthscales": [50.0, 50.0, 2.0],
        "amplitudes": [0.6, 0.9],
        "noises": [0.05, 0.05],
        "norm_params": cache.params.to_dict(),
    }
    path = tmp_path / "gp_params.json"
    import json
    path.write_text(json.dumps(hp))
    predictor = GaussianProcessPredictor(str(path))

    sid = cache.ids[0]
    context = [
        {k: v for k, v in obs.items()} for obs in cache.raw[sid]]
    queries = [{"lat": cache.params.centre_lat + 0.1,
                "lon": cache.params.centre_lon,
                "alt_ft": 10000.0},
               {"lat": cache.params.centre_lat - 0.05,
                "lon": cache.params.centre_lon + 0.05,
                "alt_ft": 30000.0}]
    out_a = predictor.predict(context, queries)
    out_b = predictor.predict(context, queries)

    expected_keys = {
        "wind_dir_deg", "wind_speed_kt",
        "wind_dir_std", "wind_speed_std",
        "wind_u_std", "wind_v_std", "combined_vector_std",
        "epistemic_dir_std", "epistemic_speed_std",
        "aleatoric_dir_std", "aleatoric_speed_std",
        "epistemic_u_std", "epistemic_v_std",
        "aleatoric_u_std", "aleatoric_v_std",
    }
    assert set(out_a) == expected_keys
    for key in expected_keys:
        assert np.allclose(out_a[key], out_b[key])
    assert ((out_a["wind_dir_deg"] >= 0)
            & (out_a["wind_dir_deg"] < 360)).all()
    assert (out_a["wind_speed_kt"] > 0).all()

    # Combined vector std must equal sqrt(u^2 + v^2) of the component stds.
    comb = np.sqrt(out_a["wind_u_std"] ** 2 + out_a["wind_v_std"] ** 2)
    assert np.allclose(out_a["combined_vector_std"], comb, atol=1e-8)


def test_encode_wind_matches_gp_target_encoding(params):
    u, v = encode_wind(90.0, 20.0, params)
    assert u == pytest.approx(-1.0, abs=1e-6)
    assert v == pytest.approx(0.0, abs=1e-6)
