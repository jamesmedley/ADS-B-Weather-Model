"""
heatmap_aircraft.py â€” Strava-style density heatmap of aircraft
flight tracks, built directly from the raw ADS-B SQLite database.

Reconstructs per-aircraft polylines from consecutive position reports,
rasterises them onto a pixel grid with per-pixel accumulation, then
renders a glowing intensity heatmap. Produces one heatmap per
flight-level altitude band plus an all-altitude combined map.

Usage:
    python heatmap_aircraft.py --db data/adsb.db \
        --out-dir outputs/imgs
    python heatmap_aircraft.py --db data/adsb.db \
        --bands 0,10000,20000,30000,40000,100000 --grid
"""

import os
import sqlite3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.ndimage

from wind_map.utils import lonlat_to_mercator


MERCATOR = None  # set in __main__


def load_observations(db_path, start=None, end=None, min_rssi=None):
    """Load (hex, t, lat, lon, alt_ft) for all valid observations."""
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    sql = """
        SELECT hex, snapshot_time, lat, lon, altitude_ft
        FROM aircraft_observations
        WHERE lat IS NOT NULL
          AND lon IS NOT NULL
          AND altitude_ft IS NOT NULL
    """
    params = []
    if start is not None:
        sql += " AND snapshot_time >= ?"
        params.append(start)
    if end is not None:
        sql += " AND snapshot_time <= ?"
        params.append(end)
    if min_rssi is not None:
        sql += " AND rssi >= ?"
        params.append(min_rssi)
    sql += " ORDER BY hex, snapshot_time ASC"
    cur.execute(sql, params)
    rows = cur.fetchall()
    con.close()

    hexes = [r[0] for r in rows]
    times = np.array([r[1] for r in rows], dtype=np.float64)
    lats = np.array([r[2] for r in rows], dtype=np.float64)
    lons = np.array([r[3] for r in rows], dtype=np.float64)
    alts = np.array([r[4] for r in rows], dtype=np.float64)
    return hexes, times, lats, lons, alts


def build_tracks(hexes, times, lats, lons, alts, gap_s):
    """Group observations into per-aircraft tracks.

    A track is broken when consecutive observations of the same hex are
    more than gap_s apart (so out-and-return flights aren't joined). We
    deliberately do NOT break on altitude jumps: aircraft climb/descend
    thousands of ft between snapshots.

    Returns a list of tracks; each track is a dict with coords.
    """
    n = len(hexes)
    tracks = []
    i = 0
    while i < n:
        j = i + 1
        while j < n and hexes[j] == hexes[i]:
            j += 1

        idx = np.arange(i, j)
        t = times[idx]
        lat = lats[idx]
        lon = lons[idx]
        alt = alts[idx]

        # Identify segments to break on (time gaps only)
        breaks = np.zeros(j - i, dtype=bool)
        breaks[1:] = (t[1:] - t[:-1]) > gap_s

        start = 0
        for k in np.where(breaks)[0].tolist() + [j - i]:
            if k - start >= 2:
                tracks.append({
                    "hex": hexes[i],
                    "lat": lat[start:k],
                    "lon": lon[start:k],
                    "alt": alt[start:k],
                })
            start = k
        i = j

    return tracks


def split_at_band_edges(track, band_edges):
    """Return line segments as (lat1, lon1, alt1, lat2, lon2, alt2).

    Segments are split at every band edge crossed, so each returned
    segment lies entirely within one band.
    """
    lat, lon, alt = track["lat"], track["lon"], track["alt"]
    edges = np.array(band_edges, dtype=np.float64)
    segs = []
    for p in range(len(lat) - 1):
        la0, lo0, a0 = lat[p], lon[p], alt[p]
        la1, lo1, a1 = lat[p + 1], lon[p + 1], alt[p + 1]

        # Band edges strictly between the two altitudes
        lo_edge, hi_edge = min(a0, a1), max(a0, a1)
        cross = edges[(edges > lo_edge) & (edges < hi_edge)]

        if cross.size == 0:
            segs.append((la0, lo0, a0, la1, lo1, a1))
            continue

        cuts = np.concatenate(([a0], cross, [a1]))
        la_prev, lo_prev, a_prev = la0, lo0, a0
        for c in cuts[1:]:
            f = (c - a0) / (a1 - a0)
            la_c = la0 + f * (la1 - la0)
            lo_c = lo0 + f * (lo1 - lo0)
            segs.append((la_prev, lo_prev, a_prev, la_c, lo_c, c))
            la_prev, lo_prev, a_prev = la_c, lo_c, c
    return segs


def rasterise(segs, W, H, extent, gauss_sigma):
    """Rasterise segments to a blurred density image (float32).

    `segs` are (lat1, lon1, alt1, lat2, lon2, alt2); positions are
    converted to Mercator metres internally so they match `extent`.
    """
    xmin, xmax, ymin, ymax = extent
    sx = W / (xmax - xmin)
    sy = H / (ymax - ymin)

    if not segs:
        return np.zeros((H, W), dtype=np.float32)

    seg_arr = np.asarray(segs, dtype=np.float64)
    lats = np.concatenate([seg_arr[:, 0], seg_arr[:, 3]])
    lons = np.concatenate([seg_arr[:, 1], seg_arr[:, 4]])
    mx, my = lonlat_to_mercator(lons, lats)
    x = (mx - xmin) * sx
    y = (my - ymin) * sy
    x0, x1 = x[:seg_arr.shape[0]], x[seg_arr.shape[0]:]
    y0, y1 = y[:seg_arr.shape[0]], y[seg_arr.shape[0]:]

    num = len(seg_arr)
    lengths = np.hypot(x1 - x0, y1 - y0)
    # Target ~1 px spacing along each segment
    steps = np.maximum(2, np.ceil(lengths)).astype(np.int64)

    idx0 = np.repeat(np.arange(num), steps)
    t = np.concatenate([
        np.linspace(0.0, 1.0, s, endpoint=True) for s in steps
    ])
    xs = x0[idx0] + t * (x1[idx0] - x0[idx0])
    ys = y0[idx0] + t * (y1[idx0] - y0[idx0])

    xi = np.clip(np.floor(xs).astype(np.int64), 0, W - 1)
    yi = np.clip(np.floor(ys).astype(np.int64), 0, H - 1)
    flat = np.bincount(yi * W + xi, minlength=W * H)

    grid = flat.reshape(H, W).astype(np.float32)
    if gauss_sigma > 0:
        grid = scipy.ndimage.gaussian_filter(grid, sigma=gauss_sigma)
    return grid


def accumulate_points(lats, lons, W, H, extent, gauss_sigma):
    """Rasterise single-point observations (short tracks)."""
    xmin, xmax, ymin, ymax = extent
    sx = W / (xmax - xmin)
    sy = H / (ymax - ymin)

    mx, my = lonlat_to_mercator(lons, lats)
    xi = np.clip(np.floor((mx - xmin) * sx).astype(np.int64), 0, W - 1)
    yi = np.clip(np.floor((my - ymin) * sy).astype(np.int64), 0, H - 1)
    flat = np.bincount(yi * W + xi, minlength=W * H)
    grid = flat.reshape(H, W).astype(np.float32)
    if gauss_sigma > 0:
        grid = scipy.ndimage.gaussian_filter(grid, sigma=gauss_sigma)
    return grid


def all_segments(tracks, band_edges):
    """Flatten every track into (lat, lon, alt, lat, lon, alt) segments."""
    segs = []
    for tr in tracks:
        segs.extend(split_at_band_edges(tr, band_edges))
    return segs


def to_intensity(grid, pmin=0.1, pmax=99.9):
    """Log-stretched normalised intensity in [0, 1]."""
    vmin, vmax = np.percentile(grid, pmin), np.percentile(grid, pmax)
    if vmax <= vmin:
        return np.zeros_like(grid)
    loggrid = np.log1p(grid)
    logvmin = np.log1p(vmin)
    logvmax = np.log1p(vmax)
    if logvmax <= logvmin:
        return np.zeros_like(grid)
    return np.clip(
        (loggrid - logvmin) / (logvmax - logvmin), 0, 1)


CMA = {
    "blue": mcolors.LinearSegmentedColormap.from_list(
        "strava_blue",
        ["#000000", "#001a33", "#004080", "#0099ff", "#66ccff", "#ffffff"]),
    "white": mcolors.LinearSegmentedColormap.from_list(
        "strava_white",
        ["#000000", "#1a1a1a", "#555555", "#aaaaaa", "#ffffff"]),
    "heat": "inferno",
    "mag": mcolors.LinearSegmentedColormap.from_list(
        "strava_mag",
        ["#000000", "#2b0066", "#7a00b3", "#ff33cc", "#ffcc00", "#ffffff"]),
}


def render_heatmap(grid, cmap, extent, out_path,
                   title=None, dpi=150, figsize=(10, 10)):
    """Draw the intensity grid as a heatmap on a black background."""
    intensity = to_intensity(grid)
    cm = CMA[cmap]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    ax.imshow(
        intensity, origin="lower",
        extent=extent, cmap=cm, vmin=0, vmax=1,
        interpolation="nearest", zorder=2)

    if title:
        ax.set_title(
            title, color="white", fontsize=14,
            loc="left", pad=12)

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor="black")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="data/adsb.db",
                   help=("Path to ADS-B SQLite database"
                         " (default: data/adsb.db)"))
    p.add_argument("--out-dir", default="outputs/imgs",
                   help="Output directory for images (default: outputs/imgs)")
    p.add_argument("--bands", type=str,
                   default="0,10000,20000,30000,40000,100000",
                   help="Comma-separated altitude band edges (ft)")
    p.add_argument("--gap_s", type=float, default=300.0,
                   help="Break a track when observations are > N s apart")
    p.add_argument("--pixels", type=int, default=2000,
                   help="Longest grid axis in pixels")
    p.add_argument("--blur", type=float, default=1.5,
                   help="Gaussian blur sigma (px)")
    p.add_argument("--cmap", choices=sorted(CMA), default="blue")
    p.add_argument("--grid", action="store_true",
                   help="Also write a single montage of all plates")
    p.add_argument("--start", default=None,
                   help=("Include observations at or after"
                         " this epoch/ISO time"))
    p.add_argument("--end", default=None,
                   help=("Include observations at or before"
                         " this epoch/ISO time"))
    p.add_argument("--min_rssi", type=float, default=None,
                   help="Drop observations with rssi below this value")

    args = p.parse_args()

    bands = [float(x) for x in args.bands.split(",")]
    if len(bands) < 3:
        raise SystemExit("Need at least 3 band edges.")

    print(f"Loading observations from {args.db} ...")
    hexes, times, lats, lons, alts = load_observations(
        args.db, start=args.start, end=args.end, min_rssi=args.min_rssi)
    print(f"  {len(lats)} observations, "
          f"{len(set(hexes))} aircraft")

    if len(lats) < 2:
        print("  No observations to plot.")
        return

    print("Reconstructing tracks ...")
    tracks = build_tracks(hexes, times, lats, lons, alts, args.gap_s)
    print(f"  {len(tracks)} tracks")

    # Bounding extent (Mercator metres) with small margin
    margin = 0.03
    mx, my = lonlat_to_mercator(lons, lats)
    xmin, xmax = mx.min(), mx.max()
    ymin, ymax = my.min(), my.max()
    padx = (xmax - xmin) * margin
    pady = (ymax - ymin) * margin
    extent = (xmin - padx, xmax + padx, ymin - pady, ymax + pady)
    span_x = xmax - xmin
    span_y = ymax - ymin
    W = args.pixels
    H = max(int(args.pixels * span_y / span_x), 64)

    short_tracks = [t for t in tracks
                    if len(t["lat"]) < 2]
    short_lat = np.concatenate([t["lat"] for t in short_tracks]) \
        if short_tracks else np.zeros(0)
    short_lon = np.concatenate([t["lon"] for t in short_tracks]) \
        if short_tracks else np.zeros(0)

    band_labels = {}
    for i in range(len(bands) - 1):
        lo, hi = bands[i], bands[i + 1]
        if lo == 0:
            label = f"0-{(hi/1000):.0f}k"
        else:
            label = f"{(lo/1000):.0f}k-{(hi/1000):.0f}k"
        band_labels[(lo, hi)] = label

    print("Rasterising all-altitude heatmap ...")
    all_segs = all_segments(tracks, bands)
    grid_all = rasterise(all_segs, W, H, extent, args.blur)
    if short_tracks:
        grid_all += accumulate_points(
            short_lat, short_lon, W, H, extent, args.blur)

    print("Rasterising per-band heatmaps ...")
    band_grids = {}
    for i in range(len(bands) - 1):
        lo, hi = bands[i], bands[i + 1]
        band_segs = [
            s for s in all_segs
            if lo <= min(s[2], s[5]) and max(s[2], s[5]) <= hi
        ]
        grid = rasterise(band_segs, W, H, extent, args.blur)
        if short_tracks:
            grid += accumulate_points(
                short_lat, short_lon, W, H, extent, args.blur)
        band_grids[(lo, hi)] = grid

    os.makedirs(args.out_dir, exist_ok=True)

    for (lo, hi), grid in band_grids.items():
        out = os.path.join(
            args.out_dir,
            f"aircraft_heatmap_{band_labels[(lo, hi)]}.png")
        render_heatmap(
            grid, args.cmap, extent, out, title=band_labels[(lo, hi)])

    out_all = os.path.join(args.out_dir, "aircraft_heatmap_all.png")
    render_heatmap(grid_all, args.cmap, extent, out_all,
                   title="All Altitudes")

    if args.grid:
        n = len(band_grids) + 1
        fig, axes = plt.subplots(
            1, n, figsize=(4.4 * n, 4.4),
            facecolor="black")
        cm = CMA[args.cmap]
        order = list(band_grids.keys())
        for ax, (lo, hi) in zip(axes[:-1], order):
            ax.imshow(
                to_intensity(band_grids[(lo, hi)]), origin="lower",
                extent=extent, cmap=cm, vmin=0, vmax=1)
            ax.set_title(band_labels[(lo, hi)], color="white",
                         fontsize=11)
            ax.set_axis_off()
        axes[-1].imshow(
            to_intensity(grid_all), origin="lower",
            extent=extent, cmap=cm, vmin=0, vmax=1)
        axes[-1].set_title("All", color="white", fontsize=11)
        axes[-1].set_axis_off()

        out_grid = os.path.join(
            args.out_dir, "aircraft_heatmap_grid.png")
        plt.savefig(out_grid, dpi=150, bbox_inches="tight",
                    facecolor="black")
        plt.close(fig)
        print(f"Saved -> {out_grid}")


if __name__ == "__main__":
    main()
