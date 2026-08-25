"""
Feature engineering, encoding and normalisation parameter tests.
"""

import dataclasses
import json
import math

import pytest

from wind_map.preprocess import (
    KM_PER_DEG_LAT, NormParams, decode_wind, encode_wind,
    normalise_coords,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "wind_dir,speed",
    [
        (0.0, 0.0),
        (1.0, 5.0),
        (44.9, 12.3),
        (90.0, 33.3),
        (179.999, 80.0),
        (180.0, 150.0),
        (270.0, 42.0),
        (359.0, 7.5),
        (359.99, 64.2),
    ],
)
def test_encode_decode_roundtrip(params, wind_dir, speed):
    u, v = encode_wind(wind_dir, speed, params)
    deg, kt = decode_wind(u, v, params)
    err = abs(deg - wind_dir) % 360
    err = min(err, 360 - err)
    assert err < 1e-6
    assert kt == pytest.approx(speed, rel=1e-9)


def test_encode_speed_zero(params):
    u, v = encode_wind(45.0, 0.0, params)
    assert u == pytest.approx(0.0, abs=1e-12)
    assert v == pytest.approx(0.0, abs=1e-12)


def test_normalise_coords_centre_maps_to_zero(params):
    lat_n, lon_n, alt_n = normalise_coords(
        params.centre_lat, params.centre_lon, 0.0, params)
    assert lat_n == 0.0
    assert lon_n == 0.0
    assert alt_n == 0.0


def test_normalise_coords_lat_offset_hand_calculated(params):
    # 60 km north of centre: lat_km / range_km = 60 / 120 = 0.5
    lat = params.centre_lat + 60.0 / KM_PER_DEG_LAT
    lat_n, lon_n, _ = normalise_coords(lat, params.centre_lon, 0.0,
                                       params)
    assert lat_n == pytest.approx(0.5, rel=1e-12)
    assert lon_n == 0.0


def test_normalise_coords_altitude_log_scaling(params):
    _, _, alt_lo = normalise_coords(
        params.centre_lat, params.centre_lon, 0.0, params)
    _, _, alt_hi = normalise_coords(
        params.centre_lat, params.centre_lon,
        params.max_alt_ft, params)
    assert alt_lo == 0.0
    assert alt_hi == pytest.approx(1.0, rel=1e-12)


def test_km_per_deg_lon_cosine_correction(params):
    expected = 111.0 * math.cos(math.radians(params.centre_lat))
    assert params.km_per_deg_lon == pytest.approx(expected)


def test_norm_params_meta_roundtrip(params, tmp_path):
    meta = {"normalisation": params.to_dict()}
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(meta))
    loaded = NormParams.from_meta(str(path))
    for field in dataclasses.fields(NormParams):
        name = field.name
        assert getattr(loaded, name) == getattr(params, name)


def test_to_dict_key_names_match_meta_contract(params):
    d = params.to_dict()
    assert d["u_mean"] == params.u_mean
    assert d["u_std"] == params.u_std
    assert d["v_mean"] == params.v_mean
    assert d["v_std"] == params.v_std
    assert set(d) == {
        "centre_lat", "centre_lon", "range_km", "max_alt_ft",
        "u_mean", "u_std", "v_mean", "v_std",
    }


def test_from_meta_missing_key_raises(params, tmp_path):
    bad = {"normalisation": {"centre_lat": 1.0}}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(KeyError):
        NormParams.from_meta(str(path))
