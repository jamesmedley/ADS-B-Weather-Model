"""
convert_db.py integration tests.

Builds a synthetic SQLite database with the production schema, runs
the full conversion, and validates the resulting .npy cache against
hand-computed normalisation statistics.
"""

import importlib.util
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from wind_map.preprocess import (
    KM_PER_DEG_LAT, WindSnapshotDataset, load_params,
    normalise_coords, encode_wind,
)

pytestmark = pytest.mark.integration

_SCRIPT = (Path(__file__).resolve().parents[1]
           / "scripts" / "convert_db.py")
_spec = importlib.util.spec_from_file_location(
    "convert_db_under_test", _SCRIPT)
convert_db = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(convert_db)

CENTRE_LAT, CENTRE_LON = 51.4, -1.2


def write_db(path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE snapshots (
            snapshot_id INTEGER PRIMARY KEY,
            snapshot_time TEXT,
            aircraft_count INTEGER
        )
    """)
    cur.execute("""
        CREATE TABLE aircraft_observations (
            snapshot_id INTEGER,
            lat REAL,
            lon REAL,
            altitude_ft REAL,
            wind_direction_deg REAL,
            wind_speed_kt REAL
        )
    """)
    sid = 0
    for day in range(1, 7):          # 6 calendar days
        for hour in ("06", "18"):    # 2 snapshots per day
            sid += 1
            time_str = f"2024-06-{day:02d}T{hour}:00:00"
            rows = []
            for k in range(4):
                rows.append((
                    CENTRE_LAT + 0.05 * k * (1 if k % 2 else -1),
                    CENTRE_LON + 0.07 * k,
                    5000.0 * (k + 1),
                    45.0 * k,
                    10.0 + 5.0 * k,
                ))
            # one row with NULL wind must be dropped
            rows.append((51.5, -1.3, 20000.0, None, None))
            cur.execute(
                "INSERT INTO snapshots VALUES (?, ?, ?)",
                (sid, time_str, len(rows)))
            for r in rows:
                cur.execute(
                    "INSERT INTO aircraft_observations"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (sid, *r))

    # filtered out: aircraft_count below the minimum
    sid += 1
    cur.execute("INSERT INTO snapshots VALUES (?, ?, ?)",
                (sid, "2024-06-07T06:00:00", 2))
    cur.execute(
        "INSERT INTO aircraft_observations VALUES"
        " (?, ?, ?, ?, ?, ?)",
        (sid, 51.5, -1.3, 10000.0, 90.0, 20.0))
    cur.execute(
        "INSERT INTO aircraft_observations VALUES"
        " (?, ?, ?, ?, ?, ?)",
        (sid, 51.55, -1.25, 11000.0, 91.0, 21.0))

    # passes count filter but every wind field NULL -> empty snapshot
    sid += 1
    cur.execute("INSERT INTO snapshots VALUES (?, ?, ?)",
                (sid, "2024-06-08T06:00:00", 4))
    cur.execute(
        "INSERT INTO aircraft_observations VALUES"
        " (?, ?, ?, ?, ?, ?)",
        (sid, 51.5, -1.3, 9000.0, None, None))

    con.commit()
    con.close()


@pytest.fixture()
def converted(tmp_path):
    db_path = tmp_path / "adsb.db"
    out_dir = tmp_path / "npy_cache"
    write_db(str(db_path))
    convert_db.convert(str(db_path), str(out_dir),
                       min_aircraft=3)
    return out_dir


def test_cache_files_and_dtypes(converted):
    x = np.load(converted / "x.npy")
    y = np.load(converted / "y.npy")
    ids = np.load(converted / "snapshot_ids.npy")
    times = np.load(converted / "snapshot_times.npy")
    offsets = np.load(converted / "offsets.npy")
    assert x.dtype == np.float32 and y.dtype == np.float32
    assert ids.dtype == np.int64 and offsets.dtype == np.int64
    assert times.dtype == np.dtype("<U32")
    assert x.ndim == 2 and x.shape[1] == 3
    assert y.ndim == 2 and y.shape[1] == 2
    assert y.shape[0] == x.shape[0]
    assert len(ids) == len(times)
    assert len(offsets) == len(ids) + 1
    assert offsets[0] == 0
    assert (np.diff(offsets) >= 0).all()
    assert offsets[-1] == len(x)


def test_aircraft_count_filter_excludes_snapshot(converted):
    ids = np.load(converted / "snapshot_ids.npy").tolist()
    n_expected = 12 + 1   # 12 normal + 1 all-NULL snapshot
    assert len(ids) == n_expected
    excluded_id = 13      # the two-row, aircraft_count=2 snapshot
    assert excluded_id not in ids


def test_null_wind_rows_dropped(converted):
    meta = json.loads((converted / "meta.json").read_text())
    # 4 valid rows per normal snapshot, 0 for the all-NULL one
    assert meta["num_observations"] == 48
    assert meta["num_empty_snapshots"] == 1


def test_meta_normalisation_block(converted):
    params = load_params(str(converted))
    raw = json.loads(
        (converted / "meta.json").read_text())["normalisation"]
    assert params.centre_lat == pytest.approx(
        raw["centre_lat"])
    assert params.centre_lat == pytest.approx(51.425)
    assert params.range_km > 0
    assert params.max_alt_ft > 19000.0
    assert np.isfinite(params.u_mean)
    assert np.isfinite(params.u_std) and params.u_std > 0
    assert np.isfinite(params.v_mean)
    assert np.isfinite(params.v_std) and params.v_std > 0


def test_norm_fit_scope_is_train_days_only(converted):
    meta = json.loads((converted / "meta.json").read_text())
    fit = meta["norm_fit"]
    assert fit["scope"] == "train_days_only"
    assert fit["split_seed"] == 42
    assert fit["n_train_snapshots"] > 0


def test_rows_match_manual_encoding(converted):
    params = load_params(str(converted))
    ds = WindSnapshotDataset(str(converted))
    x, y = ds[0]
    _, _, alt_n = normalise_coords(
        CENTRE_LAT, CENTRE_LON, 5000.0, params)
    u, v = encode_wind(0.0, 10.0, params)
    assert x.shape == (4, 3)
    assert y.shape[1] == 2
    assert float(y[0, 0]) == pytest.approx(u, abs=1e-5)
    assert float(y[0, 1]) == pytest.approx(v, abs=1e-5)
    assert float(x[0, 2]) == pytest.approx(float(alt_n),
                                           abs=1e-5)


def test_chronological_ordering(converted):
    ids = np.load(converted / "snapshot_ids.npy")
    times = np.load(converted / "snapshot_times.npy")
    order = np.argsort(times, kind="stable")
    assert (order == np.arange(len(ids))).all()


def test_convert_empty_db_raises(tmp_path):
    db_path = tmp_path / "empty.db"
    con = sqlite3.connect(db_path)
    con.execute("""
        CREATE TABLE snapshots (
            snapshot_id INTEGER PRIMARY KEY,
            snapshot_time TEXT,
            aircraft_count INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE aircraft_observations (
            snapshot_id INTEGER, lat REAL, lon REAL,
            altitude_ft REAL, wind_direction_deg REAL,
            wind_speed_kt REAL
        )
    """)
    con.commit()
    con.close()
    with pytest.raises(RuntimeError):
        convert_db.convert(str(db_path), str(tmp_path / "out"),
                           min_aircraft=3)


def test_km_per_deg_lon_consistent_with_preprocess():
    from wind_map.preprocess import NormParams
    p = NormParams(centre_lat=60.0, centre_lon=0.0, range_km=1.0,
                   max_alt_ft=1.0, u_mean=0.0, u_std=1.0,
                   v_mean=0.0, v_std=1.0)
    assert p.km_per_deg_lon == pytest.approx(
        KM_PER_DEG_LAT * math.cos(math.radians(60.0)))
