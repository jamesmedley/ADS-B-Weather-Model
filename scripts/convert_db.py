"""
convert_db.py — Convert the ADS-B wind
SQLite database into a flat .npy cache.

SQLite is fine for ingestion but too slow for DataLoader random-access reads.
This script does the scan and normalisation once, writing memory-mappable
numpy arrays so training never touches SQLite.

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
import time
import sqlite3
import argparse
import numpy as np

from wind_map.preprocess import (
    normalise_coords, encode_wind, MIN_AIRCRAFT,
    CENTRE_LAT, CENTRE_LON, LAT_RANGE_DEG, LON_RANGE_DEG,
    MAX_ALT_FT,
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


def convert(db_path, out_dir, min_aircraft=MIN_AIRCRAFT):
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

    x_chunks, y_chunks = [], []
    offsets = np.zeros(len(snapshot_ids) + 1, dtype=np.int64)
    total_rows = 0
    all_wind_speeds = []

    for i, sid in enumerate(snapshot_ids):
        rows = rows_by_snapshot.get(sid, [])
        if rows:
            xs = np.empty((len(rows), 3), dtype=np.float32)
            ys = np.empty((len(rows), 3), dtype=np.float32)
            for j, (lat, lon, alt_ft, wind_dir, wind_speed) in enumerate(rows):
                lat_n, lon_n, alt_n = normalise_coords(lat, lon, alt_ft)
                sin_w, cos_w, spd_n = encode_wind(wind_dir, wind_speed)
                xs[j] = (lat_n, lon_n, alt_n)
                ys[j] = (sin_w, cos_w, spd_n)
                all_wind_speeds.append(wind_speed)
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
    # Stringify snapshot_time and store as fixed-width
    # unicode (no allow_pickle needed)
    snapshot_times_arr = np.asarray(
        [str(t) for t in snapshot_times], dtype='<U32'
    )

    np.save(os.path.join(out_dir, 'x.npy'), x_all)
    np.save(os.path.join(out_dir, 'y.npy'), y_all)
    np.save(os.path.join(out_dir, 'snapshot_ids.npy'), snapshot_ids_arr)
    np.save(os.path.join(out_dir, 'snapshot_times.npy'), snapshot_times_arr)
    np.save(os.path.join(out_dir, 'offsets.npy'), offsets)

    empty_snapshots = int(np.sum(np.diff(offsets) == 0))
    wind_speeds_arr = np.array(all_wind_speeds)
    computed_mean = float(wind_speeds_arr.mean())
    computed_std = float(wind_speeds_arr.std())
    meta = {
        'source_db': os.path.abspath(db_path),
        'min_aircraft': min_aircraft,
        'num_snapshots': len(snapshot_ids),
        'num_empty_snapshots': empty_snapshots,
        'num_observations': int(total_rows),
        'normalisation': {
            'centre_lat': CENTRE_LAT,
            'centre_lon': CENTRE_LON,
            'lat_range_deg': LAT_RANGE_DEG,
            'lon_range_deg': LON_RANGE_DEG,
            'max_alt_ft': MAX_ALT_FT,
            'wind_speed_mean_kt': computed_mean,
            'wind_speed_std_kt': computed_std,
        },
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    dt = time.time() - t0
    print(f"  {len(snapshot_ids)} snapshots  ({empty_snapshots} empty)")
    print(f"  {total_rows} observations written")
    print(f"  wind speed stats: mean={computed_mean:.2f} kt"
          f"  std={computed_std:.2f} kt")
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
        '--min-aircraft', type=int,
        default=MIN_AIRCRAFT,
        help=(
            f'Minimum aircraft_count per snapshot'
            f' (default: {MIN_AIRCRAFT})'
        )
    )
    args = parser.parse_args()

    convert(args.db, args.out, min_aircraft=args.min_aircraft)
