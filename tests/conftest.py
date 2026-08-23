"""
conftest.py — shared fixtures for the wind_map test suite.

The central fixture is a synthetic .npy cache built programmatically in
tmp_path using the exact on-disk layout written by scripts/convert_db.py
(x.npy, y.npy, snapshot_ids.npy, snapshot_times.npy, offsets.npy,
meta.json). All dataset / split / dataloader / training / inference
tests consume the same deterministic synthetic data, so no real
ADS-B data is ever needed.
"""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from wind_map.network import LatentModel
from wind_map.preprocess import (
    NormParams, WindSnapshotDataset, encode_wind, normalise_coords,
)

TEST_SEED = 1234

PARAMS = NormParams(
    centre_lat=51.4,
    centre_lon=-1.2,
    range_km=120.0,
    max_alt_ft=40_000.0,
    wind_speed_mean_kt=30.0,
    wind_speed_std_kt=15.0,
    log_speed_mean=math.log(31.0),
    log_speed_std=0.5,
)


@dataclasses.dataclass
class SyntheticCache:
    """Generated cache plus the raw observations used to build it."""
    path: Path
    params: NormParams
    raw: dict            # snapshot_id -> list of observation dicts
    times: dict          # snapshot_id -> ISO time string

    @property
    def ids(self):
        return sorted(self.raw.keys())


def _make_observations(rng, n_obs):
    obs = []
    for _ in range(n_obs):
        lat = PARAMS.centre_lat + (
            rng.uniform(-0.8, 0.8) * PARAMS.range_km) / 111.0
        lon = PARAMS.centre_lon + (
            rng.uniform(-0.8, 0.8) * PARAMS.range_km
            / PARAMS.km_per_deg_lon)
        rec = {
            "lat": float(lat),
            "lon": float(lon),
            "alt_ft": float(rng.uniform(500.0, PARAMS.max_alt_ft)),
            "wind_dir": float(rng.uniform(0.0, 360.0)),
            "wind_speed": float(rng.uniform(1.0, 80.0)),
        }
        obs.append(rec)
    return obs


@pytest.fixture()
def params():
    return PARAMS


@pytest.fixture()
def make_cache(tmp_path):
    """Factory building a deterministic synthetic npy cache.

    Returns a SyntheticCache. Snapshots span >= 3 calendar days so the
    day-grouped splitter accepts them; each snapshot has >= 3
    observations so every collate path is exercised.

    The default day sizes (one 26-snapshot day plus eight 2-snapshot
    days) are chosen so the production 0.8/0.1/0.1 subset-sum always
    lands a valid train/val/test partition: train takes exactly five
    days (26+8=34 of 42 snapshots), leaving four days from which the
    exact test target (4) takes two days — val keeps two days no
    matter how ties inside the DP resolve.
    """
    def _make(day_sizes=(26, 2, 2, 2, 2, 2, 2, 2, 2),
              seed=TEST_SEED, tag="cache"):
        rng = np.random.default_rng(seed)
        cache_root = tmp_path / f"{tag}"
        cache_root.mkdir(parents=True, exist_ok=True)

        xs, ys, offsets = [], [], [0]
        raw, times = {}, {}
        sid = 101
        total = 0
        for day, n_snaps in enumerate(day_sizes):
            for snap in range(n_snaps):
                time_str = (
                    f"2024-06-{day + 1:02d}"
                    f"T{9 + 6 * (snap % 2):02d}:00:00"
                )
                recs = _make_observations(
                    rng, int(rng.integers(3, 13)))
                for r in recs:
                    lat_n, lon_n, alt_n = normalise_coords(
                        r["lat"], r["lon"], r["alt_ft"], PARAMS)
                    s, c, sp = encode_wind(
                        r["wind_dir"], r["wind_speed"], PARAMS)
                    xs.append([lat_n, lon_n, alt_n])
                    ys.append([s, c, sp])
                total += len(recs)
                offsets.append(total)
                raw[sid] = recs
                times[sid] = time_str
                sid += 1

        np.save(cache_root / "x.npy",
                np.asarray(xs, dtype=np.float32))
        np.save(cache_root / "y.npy",
                np.asarray(ys, dtype=np.float32))
        np.save(cache_root / "snapshot_ids.npy",
                np.asarray(sorted(raw.keys()), dtype=np.int64))
        np.save(cache_root / "snapshot_times.npy",
                np.asarray([times[i] for i in sorted(raw.keys())],
                           dtype="<U32"))
        np.save(cache_root / "offsets.npy",
                np.asarray(offsets, dtype=np.int64))
        with open(cache_root / "meta.json", "w") as f:
            json.dump({"normalisation": PARAMS.to_dict()}, f)
        return SyntheticCache(path=cache_root, params=PARAMS,
                              raw=raw, times=times)
    return _make


@pytest.fixture()
def cache(make_cache):
    return make_cache()


@pytest.fixture()
def cache_dir(cache):
    return str(cache.path)


@pytest.fixture()
def tiny_model():
    torch.manual_seed(TEST_SEED)
    return LatentModel(
        num_hidden=8,
        num_latents=4,
        num_heads=4,
        latent_layers=1,
        deterministic_layers=1,
        num_decoder_layers=1,
        dropout=0.0,
        free_bits=0.01,
        use_dist_bias=False,
    )


@pytest.fixture()
def train_batch(cache_dir):
    ds = WindSnapshotDataset(cache_dir)
    torch.manual_seed(TEST_SEED)
    from wind_map.preprocess import collate_fn
    items = [ds[i] for i in range(min(4, len(ds)))]
    return collate_fn(items)


@pytest.fixture()
def val_batch(cache_dir):
    ds = WindSnapshotDataset(cache_dir)
    from wind_map.preprocess import collate_fn_val
    g = torch.Generator().manual_seed(TEST_SEED)
    k = min(4, len(ds))
    partitions = {}
    for i in range(k):
        n = ds[i][0].size(0)
        perm = torch.randperm(n, generator=g)
        half = n // 2
        partitions[i] = (perm[:half], perm[half:])
    items = [(i, *ds[i]) for i in range(k)]
    return collate_fn_val(items, partitions=partitions)
