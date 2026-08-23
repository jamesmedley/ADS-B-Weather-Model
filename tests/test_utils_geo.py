"""
Geospatial/statistical helper tests from wind_map.utils.
"""

import math

import pytest

from wind_map.preprocess import KM_PER_DEG_LAT
from wind_map.utils import (
    circular_mean, circular_std, format_snapshot_time,
    load_snapshot, lonlat_to_mercator, make_grid,
    pick_snapshot, wind_to_uv,
)

pytestmark = pytest.mark.unit


def test_format_snapshot_time_iso():
    assert format_snapshot_time(
        "2024-06-01T09:30:00") == "01/06/24 09:30"


def test_format_snapshot_time_passthrough_and_none():
    assert format_snapshot_time(None) is None
    assert format_snapshot_time("weird-stamp") == "weird-stamp"


def test_lonlat_to_mercator_origin_and_scale():
    x0, y0 = lonlat_to_mercator(0.0, 0.0)
    assert x0 == pytest.approx(0.0, abs=1e-6)
    assert y0 == pytest.approx(0.0, abs=1e-6)
    x45, y45 = lonlat_to_mercator(0.0, 45.0)
    expected_y = 6378137.0 * math.log(
        math.tan(math.pi / 4 + math.radians(45.0) / 2))
    assert y45 == pytest.approx(expected_y)
    assert y45 > 0
    _, y_neg = lonlat_to_mercator(0.0, -45.0)
    assert y_neg == pytest.approx(-y45)


def test_wind_to_uv_directions_and_magnitude():
    u, v = wind_to_uv(0.0, 10.0)
    assert (u, v) == pytest.approx((0.0, -10.0))
    u, v = wind_to_uv(90.0, 10.0)
    assert (u, v) == pytest.approx((-10.0, 0.0))
    u, v = wind_to_uv(180.0, 10.0)
    assert (u, v) == pytest.approx((0.0, 10.0))
    u, v = wind_to_uv(35.0, 42.0)
    assert math.hypot(u, v) == pytest.approx(42.0)


def _circ_dist(a, b):
    """Shortest angular distance between two headings, degrees."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def test_circular_mean_identical_and_wraparound():
    assert circular_mean([27.0] * 5) == pytest.approx(27.0)
    assert _circ_dist(circular_mean([359.0, 1.0]), 0.0) < 1e-4
    assert _circ_dist(circular_mean([350.0, 10.0]), 0.0) < 1e-6


def test_circular_mean_simple_average():
    assert circular_mean([0.0, 90.0]) == pytest.approx(45.0)


def test_circular_std_zero_for_identical():
    # The R clip (1 - 1e-9) floors the result near ~0.0026 deg.
    std = circular_std([123.4] * 8)
    assert 0.0 <= std < 0.01


def test_circular_std_grows_with_spread():
    tight = circular_std([0.0, 5.0])
    wide = circular_std([0.0, 60.0])
    assert wide > tight > 0


def test_make_grid_shapes_and_extent(params):
    n_lat, n_lon = 5, 7
    queries, lat_grid, lon_grid, lats, lons = make_grid(
        alt_ft=10000.0, n_lat=n_lat, n_lon=n_lon, params=params)
    assert len(queries) == n_lat * n_lon
    assert lat_grid.shape == (n_lat, n_lon)
    assert lon_grid.shape == (n_lat, n_lon)
    lat_span = params.range_km / KM_PER_DEG_LAT
    assert lats[0] == pytest.approx(
        params.centre_lat - lat_span)
    assert lats[-1] == pytest.approx(
        params.centre_lat + lat_span)
    assert all(q["alt_ft"] == 10000.0 for q in queries)


def test_load_snapshot_roundtrips_raw_values(cache):
    sid = cache.ids[0]
    obs, t = load_snapshot(str(cache.path), sid)
    raw = cache.raw[sid]
    assert t == cache.times[sid]
    assert len(obs) == len(raw)
    for got, src in zip(obs, raw):
        assert got["lat"] == pytest.approx(src["lat"], abs=1e-3)
        assert got["lon"] == pytest.approx(src["lon"], abs=1e-3)
        assert got["alt_ft"] == pytest.approx(
            src["alt_ft"], rel=1e-3)
        assert got["wind_speed"] == pytest.approx(
            src["wind_speed"], rel=1e-3)
        assert 0.0 <= got["wind_dir"] < 360.0


def test_pick_snapshot_explicit_id(cache):
    sid = cache.ids[-1]
    obs, t, chosen = pick_snapshot(str(cache.path),
                                   snapshot_id=sid)
    assert chosen == sid
    assert len(obs) == len(cache.raw[sid])


def test_pick_snapshot_val_split_member(cache):
    from wind_map.preprocess import day_grouped_split
    obs, t, chosen = pick_snapshot(str(cache.path))
    tr, va, te = day_grouped_split(str(cache.path))
    pool = va if va else te
    assert chosen in pool
