"""
convert_db.py — Convert the ADS-B wind
SQLite database into a flat .npy cache.

SQLite is fine for ingestion but too slow for DataLoader random-access reads.
This script does the scan and normalisation once, writing memory-mappable
numpy arrays so training never touches SQLite.

Normalisation statistics are fitted on train days only: the same
deterministic day-grouped split used during training (seed 42 by default)
is computed first, and centre/range/speed statistics come from those
snapshots alone. Val/test weather therefore never influences the input
encoding or target scaling.

Cache layout (written to --out):
    x.npy            float32 [N, 3]  lat_norm, lon_norm, alt_norm
    y.npy            float32 [N, 3]  wind_dir_sin, wind_dir_cos,
                                      wind_speed_norm
    snapshot_ids.npy int64   [S]     snapshot_id per snapshot
    snapshot_times.npy <U32  [S]     snapshot_time, aligned with snapshot_ids
    offsets.npy      int64   [S+1]   row range in x/y per snapshot
    meta.json                  conversion metadata

Usage:
    python convert_db.py --db data/adsb.db --out data/npy_cache
"""

import os
import json
import math
import time
import sqlite3
import argparse
import numpy as np

from wind_map.preprocess import (
    normalise_coords, encode_wind, MIN_AIRCRAFT,
    NormParams, KM_PER_DEG_LAT, split_snapshot_ids,
)


def _ordered_snapshot_ids(con, min_aircraft):
    """Get chronological (snapshot_id, snapshot_time) pairs
    passing the aircraft count filter."""
    cur = con.cursor()
    cur.execute("""
        SELECT snapshot_id, snapshot_time
        FROM snapshots
        WHERE aircraft_count >= ?
        ORDER BY snapshot_time ASC
    """, (min_aircraft,))
    rows = cur.fetchall()
    return [row[0] for row in rows], [row[1] for row in rows]


def _all_valid_rows_by_snapshot(con):
    """Single full-table scan for every row with non-null
    wind fields. Returns
    {snapshot_id: [(lat, lon, alt, dir, spd), ...]}."""
    cur = con.cursor()
    cur.execute("""
        SELECT snapshot_id, lat, lon,
               altitude_ft, wind_direction_deg,
               wind_speed_kt
        FROM aircraft_observations
        WHERE lat IS NOT NULL
          AND lon IS NOT NULL
          AND altitude_ft IS NOT NULL
          AND wind_direction_deg IS NOT NULL
          AND wind_speed_kt IS NOT NULL
        ORDER BY snapshot_id ASC
    """)
    by_snapshot = {}
    for sid, lat, lon, alt_ft, wind_dir, wind_speed in cur.fetchall():
        by_snapshot.setdefault(sid, []).append(
            (lat, lon, alt_ft, wind_dir, wind_speed)
        )
    return by_snapshot


def convert(db_path, out_dir, min_aircraft=MIN_AIRCRAFT,
            split_seed=42, train_frac=0.8, test_frac=0.1, val_frac=0.1):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    con = sqlite3.connect(db_path)
    print(f"Reading snapshot list from {db_path} ...")
    snapshot_ids, snapshot_times = _ordered_snapshot_ids(con, min_aircraft)
    if not snapshot_ids:
        raise RuntimeError("No snapshots passed the aircraft_count filter.")
    if any(t is None for t in snapshot_times):
        raise RuntimeError("One or more snapshots have NULL snapshot_time.")

    print(f"Scanning aircraft_observations for"
          f" {len(snapshot_ids)} snapshots ...")
    rows_by_snapshot = _all_valid_rows_by_snapshot(con)
    con.close()

    # --- Day-grouped split (same algorithm/seed as training) ---
    # Fitted on train days only so aggregate statistics of val/test
    # weather never shape the input encoding or target scaling.
    train_ids_split, _, _ = split_snapshot_ids(
        snapshot_ids, snapshot_times,
        train_frac=train_frac, test_frac=test_frac,
        val_frac=val_frac, seed=split_seed, verbose=True)
    train_day_ids = set(train_ids_split)
    n_train_snapshots = len(train_day_ids)

    # --- Compute data-driven normalisation parameters (train days only) ---
    train_rows = [
        row for sid in train_day_ids
        for row in rows_by_snapshot.get(sid, [])
    ]
    if not train_rows:
        raise RuntimeError("No valid observations on train days.")
    train_arr = np.array(train_rows, dtype=np.float64)
    all_lats = train_arr[:, 0]
    all_lons = train_arr[:, 1]
    all_alts = train_arr[:, 2]
    all_ws = train_arr[:, 4]

    centre_lat = float(np.median(all_lats))
    centre_lon = float(np.median(all_lons))
    km_per_deg_lon = 111.0 * math.cos(math.radians(centre_lat))
    lat_km = (all_lats - centre_lat) * KM_PER_DEG_LAT
    lon_km = (all_lons - centre_lon) * km_per_deg_lon
    distances_km = np.sqrt(lat_km ** 2 + lon_km ** 2)
    range_km = float(np.percentile(distances_km, 99))
    max_alt_ft = float(np.percentile(all_alts, 99))
    log_speeds = np.log(1 + all_ws)
    params = NormParams(
        centre_lat=centre_lat,
        centre_lon=centre_lon,
        range_km=range_km,
        max_alt_ft=max_alt_ft,
        wind_speed_mean_kt=float(all_ws.mean()),
        wind_speed_std_kt=float(all_ws.std()),
        log_speed_mean=float(log_speeds.mean()),
        log_speed_std=float(log_speeds.std()),
    )

    # --- Normalise and write ---
    x_chunks, y_chunks = [], []
    offsets = np.zeros(len(snapshot_ids) + 1, dtype=np.int64)
    total_rows = 0

    for i, sid in enumerate(snapshot_ids):
        rows = rows_by_snapshot.get(sid, [])
        if rows:
            xs = np.empty((len(rows), 3), dtype=np.float32)
            ys = np.empty((len(rows), 3), dtype=np.float32)
            for j, (lat, lon, alt_ft, wind_dir, wind_speed) in enumerate(rows):
                lat_n, lon_n, alt_n = normalise_coords(
                    lat, lon, alt_ft, params)
                sin_w, cos_w, spd_n = encode_wind(
                    wind_dir, wind_speed, params)
                xs[j] = (lat_n, lon_n, alt_n)
                ys[j] = (sin_w, cos_w, spd_n)
            x_chunks.append(xs)
            y_chunks.append(ys)
            total_rows += len(rows)
        offsets[i + 1] = total_rows

    x_all = (
        np.concatenate(x_chunks, axis=0)
        if x_chunks
        else np.empty((0, 3), dtype=np.float32)
    )
    y_all = (
        np.concatenate(y_chunks, axis=0)
        if y_chunks
        else np.empty((0, 3), dtype=np.float32)
    )
    snapshot_ids_arr = np.asarray(snapshot_ids, dtype=np.int64)
    snapshot_times_arr = np.asarray(
        [str(t) for t in snapshot_times], dtype='<U32'
    )

    np.save(os.path.join(out_dir, 'x.npy'), x_all)
    np.save(os.path.join(out_dir, 'y.npy'), y_all)
    np.save(os.path.join(out_dir, 'snapshot_ids.npy'), snapshot_ids_arr)
    np.save(os.path.join(out_dir, 'snapshot_times.npy'), snapshot_times_arr)
    np.save(os.path.join(out_dir, 'offsets.npy'), offsets)

    empty_snapshots = int(np.sum(np.diff(offsets) == 0))
    meta = {
        'source_db': os.path.abspath(db_path),
        'min_aircraft': min_aircraft,
        'num_snapshots': len(snapshot_ids),
        'num_empty_snapshots': empty_snapshots,
        'num_observations': int(total_rows),
        'normalisation': params.to_dict(),
        'norm_fit': {
            'scope': 'train_days_only',
            'split_seed': split_seed,
            'train_frac': train_frac,
            'test_frac': test_frac,
            'val_frac': val_frac,
            'n_train_snapshots': n_train_snapshots,
        },
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    dt = time.time() - t0
    print(f"  {len(snapshot_ids)} snapshots  ({empty_snapshots} empty)")
    print(f"  {total_rows} observations written")
    print(f"  normalisation fitted on {n_train_snapshots} train-day "
          f"snapshots (seed {split_seed})")
    print(f"  centre: ({centre_lat:.4f}, {centre_lon:.4f})  "
          f"range_km: {range_km:.1f}  max_alt: {max_alt_ft:.0f} ft")
    print(f"  wind speed: mean={params.wind_speed_mean_kt:.2f} kt  "
          f"std={params.wind_speed_std_kt:.2f} kt")
    print(f"  log speed: mean={params.log_speed_mean:.4f}  "
          f"std={params.log_speed_std:.4f}")
    print(f"  cache written to {out_dir}/ in {dt:.1f}s")
    print(
        f"    x.npy: {x_all.shape} {x_all.dtype}"
        f"   y.npy: {y_all.shape} {y_all.dtype}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Convert the wind ADS-B SQLite database'
            ' into a flat .npy cache.'
        )
    )
    parser.add_argument(
        '--db', required=True,
        help='Path to source SQLite database'
    )
    parser.add_argument(
        '--out', default='./data/npy_cache',
        help='Output cache directory'
             ' (default: ./data/npy_cache)'
    )
    parser.add_argument(
        '--min_aircraft', type=int,
        default=MIN_AIRCRAFT,
        help=(
            f'Minimum aircraft_count per snapshot'
            f' (default: {MIN_AIRCRAFT})'
        )
    )
    parser.add_argument(
        '--split_seed', type=int, default=42,
        help=(
            'Seed for the day-grouped split used to select the '
            'train days that normalisation statistics are fitted '
            'on (must match training; default: 42)'
        )
    )
    parser.add_argument(
        '--train_frac', type=float, default=0.8,
        help='Train fraction for the norm-fit split (default: 0.8)'
    )
    parser.add_argument(
        '--test_frac', type=float, default=0.1,
        help='Test fraction for the norm-fit split (default: 0.1)'
    )
    parser.add_argument(
        '--val_frac', type=float, default=0.1,
        help='Val fraction for the norm-fit split (default: 0.1)'
    )
    args = parser.parse_args()

    convert(args.db, args.out, min_aircraft=args.min_aircraft,
            split_seed=args.split_seed, train_frac=args.train_frac,
            test_frac=args.test_frac, val_frac=args.val_frac)
