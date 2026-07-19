"""
analyse_variability.py — Raw-data diagnostics for the
north/south uncertainty gradient.

No model loaded — only looks at the .npy cache to check:
  1. Leave-one-out local wind variability by latitude
  2. Lat/altitude joint density (is there a confound?)

Usage:
    python analyse_variability.py --cache data/npy_cache \
        --alt_ft 36000 --output outputs/imgs/wind_variability.png
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt

from wind_map.preprocess import (
    CENTRE_LAT, LAT_RANGE_DEG, MAX_ALT_FT, WIND_SPEED_MEAN, WIND_SPEED_STD
)


def load_cache(cache_dir):
    x = np.load(f"{cache_dir}/x.npy")
    y = np.load(f"{cache_dir}/y.npy")
    offsets = np.load(f"{cache_dir}/offsets.npy")
    return x, y, offsets


def to_physical(x, y):
    """Convert normalised arrays back to physical units."""
    lat_deg = x[:, 0] * LAT_RANGE_DEG + CENTRE_LAT
    alt_ft = x[:, 2] * MAX_ALT_FT
    dir_deg = np.degrees(np.arctan2(y[:, 0], y[:, 1])) % 360
    speed_kt = y[:, 2] * WIND_SPEED_STD + WIND_SPEED_MEAN
    return lat_deg, alt_ft, dir_deg, speed_kt


def leave_one_out_residuals(x, y, offsets):
    """
    For every obs in every snapshot with >= 3 rows, compute the direction
    residual (deg) and speed residual (kt) against the leave-one-out mean
    of the other observations.
    """
    lat_list, alt_list, dir_resid_list, speed_resid_list = [], [], [], []

    sin_all, cos_all, spd_all = y[:, 0], y[:, 1], y[:, 2]

    for s in range(len(offsets) - 1):
        start, end = offsets[s], offsets[s + 1]
        n = end - start
        if n < 3:
            continue

        sin_s = sin_all[start:end]
        cos_s = cos_all[start:end]
        spd_s = spd_all[start:end] * WIND_SPEED_STD + WIND_SPEED_MEAN
        lat_s = x[start:end, 0] * LAT_RANGE_DEG + CENTRE_LAT
        alt_s = x[start:end, 2] * MAX_ALT_FT

        total_sin, total_cos, total_spd = sin_s.sum(), cos_s.sum(), spd_s.sum()

        for i in range(n):
            loo_sin = (total_sin - sin_s[i]) / (n - 1)
            loo_cos = (total_cos - cos_s[i]) / (n - 1)
            loo_spd = (total_spd - spd_s[i]) / (n - 1)

            obs_dir = np.degrees(np.arctan2(sin_s[i], cos_s[i]))
            loo_dir = np.degrees(np.arctan2(loo_sin, loo_cos))
            diff = ((obs_dir - loo_dir + 180) % 360) - 180

            lat_list.append(lat_s[i])
            alt_list.append(alt_s[i])
            dir_resid_list.append(abs(diff))
            speed_resid_list.append(spd_s[i] - loo_spd)

    return (np.array(lat_list), np.array(alt_list),
            np.array(dir_resid_list), np.array(speed_resid_list))


def bin_by_latitude(lat_deg, values, n_bins, lat_min, lat_max):
    edges = np.linspace(lat_min, lat_max, n_bins + 1)
    idx = np.digitize(lat_deg, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    means, stds, counts, centres = [], [], [], []
    for b in range(n_bins):
        v = values[idx == b]
        centres.append(0.5 * (edges[b] + edges[b + 1]))
        counts.append(len(v))
        means.append(v.mean() if len(v) else np.nan)
        stds.append(v.std() if len(v) else np.nan)
    return np.array(centres), np.array(means), np.array(stds), np.array(counts)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", required=True)
    p.add_argument(
        "--alt_ft", type=float, default=36000,
        help="Altitude for the confound check "
             "(should match uncertainty visualisation)")
    p.add_argument("--alt_tol", type=float, default=2000,
                   help="+/- band (ft) around --alt_ft")
    p.add_argument("--bins", type=int, default=15)
    p.add_argument("--output", default="outputs/imgs/wind_variability.png")
    args = p.parse_args()

    print(f"Loading cache from {args.cache} ...")
    x, y, offsets = load_cache(args.cache)
    n_snapshots = len(offsets) - 1
    print(f"  {len(x)} observations across {n_snapshots} snapshots")

    lat_min, lat_max = CENTRE_LAT - LAT_RANGE_DEG, CENTRE_LAT + LAT_RANGE_DEG

    # Part 1: leave-one-out local variability by latitude
    print("Computing leave-one-out residuals "
          "(snapshots with >=3 obs)...")
    (lat_r, alt_r, dir_resid,
     speed_resid) = leave_one_out_residuals(x, y, offsets)
    print(f"  {len(lat_r)} residuals computed")

    dir_centres, dir_means, dir_stds, dir_counts = bin_by_latitude(
        lat_r, dir_resid, args.bins, lat_min, lat_max)
    spd_centres, spd_means, spd_stds, spd_counts = bin_by_latitude(
        lat_r, np.abs(speed_resid), args.bins, lat_min, lat_max)

    dir_corr = np.corrcoef(lat_r, dir_resid)[0, 1]
    speed_corr = np.corrcoef(lat_r, np.abs(speed_resid))[0, 1]
    print(f"  corr(latitude, |direction residual|) = {dir_corr:+.3f}")
    print(f"  corr(latitude, |speed residual|)      = {speed_corr:+.3f}")
    print("  (near 0 => no real lat-dependent variability; "
          "positive => more variable north)")

    # Part 2: lat/altitude joint density
    lat_all, alt_all, _, _ = to_physical(x, y)
    band = np.abs(alt_all - args.alt_ft) <= args.alt_tol
    print(f"  {band.sum()} of {len(alt_all)} observations within "
          f"+/-{args.alt_tol:.0f} ft of {args.alt_ft:.0f} ft")

    lat_alt_corr = np.corrcoef(lat_all, alt_all)[0, 1]
    print(f"  corr(latitude, altitude) = {lat_alt_corr:+.3f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.bar(
        dir_centres, dir_means,
        width=(lat_max - lat_min) / args.bins * 0.9,
        yerr=dir_stds / np.sqrt(np.maximum(dir_counts, 1)),
        capsize=2)
    ax.set_title(
        f"Local direction variability by latitude "
        f"(corr={dir_corr:+.2f})")
    ax.set_xlabel("Latitude (deg)")
    ax.set_ylabel("Mean |direction residual| (deg)")

    ax = axes[0, 1]
    ax.bar(
        spd_centres, spd_means,
        width=(lat_max - lat_min) / args.bins * 0.9,
        yerr=spd_stds / np.sqrt(np.maximum(spd_counts, 1)),
        capsize=2, color="tab:orange")
    ax.set_title(
        f"Local speed variability by latitude "
        f"(corr={speed_corr:+.2f})")
    ax.set_xlabel("Latitude (deg)")
    ax.set_ylabel("Mean |speed residual| (kt)")

    ax = axes[1, 0]
    h = ax.hist2d(lat_all, alt_all, bins=[args.bins, 30], cmap="viridis")
    fig.colorbar(h[3], ax=ax, label="observation count")
    ax.axhline(
        args.alt_ft, color="red", linestyle="--",
        linewidth=1,
        label=f"query alt = {args.alt_ft:.0f} ft")
    ax.set_title(f"Lat/altitude joint density (corr={lat_alt_corr:+.2f})")
    ax.set_xlabel("Latitude (deg)")
    ax.set_ylabel("Altitude (ft)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1, 1]
    ax.hist(
        lat_all, bins=args.bins,
        range=(lat_min, lat_max),
        alpha=0.5, label="all altitudes", density=True)
    ax.hist(
        lat_all[band], bins=args.bins,
        range=(lat_min, lat_max),
        alpha=0.5,
        label=f"{args.alt_ft:.0f} ft +/- {args.alt_tol:.0f} ft",
        density=True)
    ax.set_title("Latitude coverage: all data vs query-altitude band")
    ax.set_xlabel("Latitude (deg)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
