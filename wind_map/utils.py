"""
utils.py — Shared helpers for visualisation scripts.

Functions here are used by visualise_wind.py, animate_wind.py,
and visualise_uncertainty.py to avoid duplication.
"""

import random
from datetime import datetime

import numpy as np

from wind_map.preprocess import (
    CENTRE_LAT, CENTRE_LON, LAT_RANGE_DEG, LON_RANGE_DEG,
    MAX_ALT_FT, day_grouped_split, decode_wind, WindSnapshotDataset,
)


def format_snapshot_time(time_str):
    if time_str is None:
        return None
    s = str(time_str).strip()
    try:
        epoch = float(s)
        return datetime.fromtimestamp(epoch).strftime("%d/%m/%y %H:%M")
    except ValueError:
        pass
    iso = s.replace('T', ' ')
    try:
        return datetime.fromisoformat(iso).strftime("%d/%m/%y %H:%M")
    except ValueError:
        pass
    return s


def lonlat_to_mercator(lon, lat):
    R = 6_378_137.0
    x = np.radians(lon) * R
    y = np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) * R
    return x, y


def load_snapshot(cache_dir, snapshot_id):
    ds = WindSnapshotDataset(cache_dir, snapshot_ids=[snapshot_id])
    x, y = ds[0]
    x, y = x.numpy(), y.numpy()

    lat = x[:, 0] * LAT_RANGE_DEG + CENTRE_LAT
    lon = x[:, 1] * LON_RANGE_DEG + CENTRE_LON
    alt = x[:, 2] * MAX_ALT_FT

    obs = []
    for i in range(len(x)):
        wd, ws = decode_wind(y[i, 0], y[i, 1], y[i, 2])
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


def make_grid(alt_ft, n_lat=30, n_lon=40,
              lat_range_deg=LAT_RANGE_DEG, lon_range_deg=LON_RANGE_DEG):
    lats = np.linspace(CENTRE_LAT - lat_range_deg,
                       CENTRE_LAT + lat_range_deg, n_lat)
    lons = np.linspace(CENTRE_LON - lon_range_deg,
                       CENTRE_LON + lon_range_deg, n_lon)
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
