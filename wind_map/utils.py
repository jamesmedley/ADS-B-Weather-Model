"""
utils.py — Shared helpers for visualisation scripts.

Functions here are used by visualise_wind.py, animate_wind.py,
and visualise_uncertainty.py to avoid duplication.
"""

import random
import math

import numpy as np

from wind_map.preprocess import (
    _parse_datetime,
    load_params, KM_PER_DEG_LAT,
    day_grouped_split, decode_wind, WindSnapshotDataset,
)


def format_snapshot_time(time_str):
    if time_str is None:
        return None
    try:
        dt = _parse_datetime(time_str)
        return dt.strftime("%d/%m/%y %H:%M")
    except ValueError:
        return str(time_str).strip()


def lonlat_to_mercator(lon, lat):
    R = 6_378_137.0
    x = np.radians(lon) * R
    y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * R
    return x, y


def load_snapshot(cache_dir, snapshot_id):
    ds = WindSnapshotDataset(cache_dir, snapshot_ids=[snapshot_id])
    x, y = ds[0]
    x, y = x.numpy(), y.numpy()
    params = load_params(cache_dir)

    km_per_deg_lon = 111.0 * math.cos(math.radians(params.centre_lat))
    lat = x[:, 0] * params.range_km / KM_PER_DEG_LAT + params.centre_lat
    lon = x[:, 1] * params.range_km / km_per_deg_lon + params.centre_lon
    alt = np.exp(x[:, 2] * np.log(1 + params.max_alt_ft)) - 1

    obs = []
    for i in range(len(x)):
        wd, ws = decode_wind(y[i, 0], y[i, 1], y[i, 2], params)
        obs.append({
            "lat": float(lat[i]),
            "lon": float(lon[i]),
            "alt_ft": float(alt[i]),
            "wind_dir": float(wd),
            "wind_speed": float(ws)
        })

    return obs, ds.snapshot_times[0]


def pick_snapshot(cache_dir, snapshot_id=None, split="val"):
    if snapshot_id is not None:
        obs, t = load_snapshot(cache_dir, snapshot_id)
        return obs, t, snapshot_id

    train_ids, val_ids, test_ids = day_grouped_split(cache_dir)
    ids = val_ids if split == "val" else test_ids
    if not ids:
        raise RuntimeError("No snapshots found.")

    for _ in range(10):
        sid = random.choice(ids)
        obs, t = load_snapshot(cache_dir, sid)
        if len(obs) >= 2:
            return obs, t, sid

    raise RuntimeError("Could not find valid snapshot.")


def make_grid(alt_ft, n_lat=30, n_lon=40, params=None,
              lat_range_deg=None, lon_range_deg=None):
    if params is not None:
        if lat_range_deg is None:
            lat_range_deg = params.range_km / KM_PER_DEG_LAT
        if lon_range_deg is None:
            lon_range_deg = params.range_km / params.km_per_deg_lon
    else:
        if lat_range_deg is None:
            lat_range_deg = 0.63
        if lon_range_deg is None:
            lon_range_deg = 1.0

    centre_lat = params.centre_lat if params is not None else 51.071066
    centre_lon = params.centre_lon if params is not None else -1.042441

    lats = np.linspace(centre_lat - lat_range_deg,
                       centre_lat + lat_range_deg, n_lat)
    lons = np.linspace(centre_lon - lon_range_deg,
                       centre_lon + lon_range_deg, n_lon)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    queries = [{"lat": float(lat_grid[i, j]),
                "lon": float(lon_grid[i, j]),
                "alt_ft": alt_ft}
               for i in range(n_lat) for j in range(n_lon)]

    return queries, lat_grid, lon_grid, lats, lons


def wind_to_uv(dir_deg, speed):
    rad = np.radians(dir_deg)
    return -speed * np.sin(rad), -speed * np.cos(rad)


def circular_mean(angles_deg, axis=0):
    rad = np.radians(angles_deg)
    return np.degrees(np.arctan2(
        np.sin(rad).mean(axis=axis),
        np.cos(rad).mean(axis=axis))) % 360


def circular_std(angles_deg, axis=0):
    rad = np.radians(angles_deg)
    R = np.sqrt(np.sin(rad).mean(axis=axis) ** 2 +
                np.cos(rad).mean(axis=axis) ** 2)
    R = np.clip(R, 0, 1 - 1e-9)
    return np.degrees(np.sqrt(-2 * np.log(R)))
