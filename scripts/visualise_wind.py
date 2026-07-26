"""
visualise_wind.py — Plot predicted wind field as a quiver
with uncertainty heatmap and basemap variants.

Saves two images from each run:
  - wind_field_uncertainty.png  — inferno uncertainty heatmap
                                  behind the quiver (grey arrows)
  - wind_field_basemap.png      — geographical basemap behind the
                                  quiver (speed-coloured arrows)

The basemap image is saved immediately after the wind field is
predicted; the high-resolution heatmap grid is only computed
afterwards, so the basemap output is not delayed.

Usage:
    python visualise_wind.py --checkpoint checkpoint/best_model.pth.tar \
        --cache data/npy_cache --alt_ft 35000 \
        --output outputs/imgs/wind_field.png
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patheffects as pe
import contextily as cx

from wind_map.infer import WindPredictor
from wind_map.preprocess import (
    load_params, KM_PER_DEG_LAT,
)
from wind_map.utils import (
    format_snapshot_time, lonlat_to_mercator, pick_snapshot,
    make_grid, wind_to_uv
)


def plot_wind(checkpoint, alt_ft, context, n_samples,
              n_lat, n_lon, output_uncertainty, output_basemap,
              num_hidden, layers, num_decoder_layers=None,
              snapshot_id=None, snapshot_time=None,
              params=None,
              lat_range_deg=None, lon_range_deg=None):
    predictor = WindPredictor(
        checkpoint, num_hidden=num_hidden,
        layers=layers,
        num_decoder_layers=num_decoder_layers,
        params=params)

    if params is not None:
        default_lat_range = params.range_km / KM_PER_DEG_LAT
        default_lon_range = params.range_km / params.km_per_deg_lon
    else:
        default_lat_range = 0.63
        default_lon_range = 1.0

    eff_lat_range = (lat_range_deg if lat_range_deg is not None
                     else default_lat_range)
    eff_lon_range = (lon_range_deg if lon_range_deg is not None
                     else default_lon_range)

    too_big = (eff_lat_range > 1.5 * default_lat_range
               or eff_lon_range > 1.5 * default_lon_range)
    if too_big:
        print(
            f"Warning: view (+-{eff_lat_range:.3f} lat "
            f"/ +-{eff_lon_range:.3f} lon) extends beyond "
            f"training range (+-{default_lat_range:.3f} lat "
            f"/ +-{default_lon_range:.3f} lon). "
            f"Predictions outside that box are extrapolated.")

    # Main grid (arrows)
    queries, lat_grid, lon_grid, _, _ = make_grid(
        alt_ft, n_lat, n_lon, params=params,
        lat_range_deg=eff_lat_range, lon_range_deg=eff_lon_range)
    result = predictor.predict(context, queries, n_samples=n_samples)

    dirs = result["wind_dir_deg"].reshape(n_lat, n_lon)
    speed = result["wind_speed_kt"].reshape(n_lat, n_lon)
    u, v = wind_to_uv(dirs, speed)

    # Mercator coords for main grid
    mx, my = lonlat_to_mercator(lon_grid, lat_grid)

    # Scale arrows
    max_spd = speed.max() + 1e-9
    u_scaled = u / max_spd
    v_scaled = v / max_spd

    # Observations
    obs_lat = np.array([o["lat"] for o in context])
    obs_lon = np.array([o["lon"] for o in context])
    obs_dir = np.array([o["wind_dir"] for o in context])
    obs_spd = np.array([o["wind_speed"] for o in context])
    obs_alt = np.array([o["alt_ft"] for o in context])

    obs_mx, obs_my = lonlat_to_mercator(obs_lon, obs_lat)
    ou, ov = wind_to_uv(obs_dir, obs_spd)
    ou_s = ou / max_spd
    ov_s = ov / max_spd

    c_lat = params.centre_lat if params is not None else 51.071066
    c_lon = params.centre_lon if params is not None else -1.042441
    x_min, y_min = lonlat_to_mercator(c_lon - eff_lon_range,
                                      c_lat - eff_lat_range)
    x_max, y_max = lonlat_to_mercator(c_lon + eff_lon_range,
                                      c_lat + eff_lat_range)

    # ------------------------------------------------------------------
    # 1) Basemap image  — saved immediately, no heatmap computation
    # ------------------------------------------------------------------
    _save_basemap(
        output_basemap, mx, my, u_scaled, v_scaled, speed,
        obs_mx, obs_my, ou_s, ov_s, obs_spd, obs_alt,
        x_min, x_max, y_min, y_max,
        alt_ft, snapshot_time, snapshot_id,
    )

    # ------------------------------------------------------------------
    # 2) High-res heatmap grid  — computed *after* basemap is saved
    # ------------------------------------------------------------------
    heat_queries, heat_lat, heat_lon, _, _ = make_grid(
        alt_ft, n_lat * 3, n_lon * 3, params=params,
        lat_range_deg=eff_lat_range, lon_range_deg=eff_lon_range)
    heat_result = predictor.predict(context, heat_queries, n_samples=n_samples)

    heat_dir_std = heat_result["wind_dir_std"].reshape(n_lat * 3, n_lon * 3)
    heat_spd_std = heat_result["wind_speed_std"].reshape(n_lat * 3, n_lon * 3)

    def _pct_normalise(arr, lo=2, hi=98):
        vmin = np.percentile(arr, lo)
        vmax = np.percentile(arr, hi)
        return np.clip((arr - vmin) / (vmax - vmin + 1e-9), 0, 1)

    heat_unc = 0.5 * (
        _pct_normalise(heat_dir_std) + _pct_normalise(heat_spd_std))

    heat_mx, heat_my = lonlat_to_mercator(heat_lon, heat_lat)

    # ------------------------------------------------------------------
    # 3) Uncertainty image  — with heatmap background
    # ------------------------------------------------------------------
    _save_uncertainty(
        output_uncertainty, mx, my, u_scaled, v_scaled,
        heat_unc, heat_mx, heat_my,
        obs_mx, obs_my, ou_s, ov_s, obs_spd, obs_alt,
        x_min, x_max, y_min, y_max,
        alt_ft, snapshot_time, snapshot_id,
    )


# ---------------------------------------------------------------------------
# Internal rendering helpers
# ---------------------------------------------------------------------------

def _save_basemap(path, mx, my, u_scaled, v_scaled, speed,
                  obs_mx, obs_my, ou_s, ov_s, obs_spd, obs_alt,
                  x_min, x_max, y_min, y_max,
                  alt_ft, snapshot_time, snapshot_id):
    fig, ax = plt.subplots(figsize=(11, 11))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    cx.add_basemap(ax, crs="EPSG:3857",
                   source=cx.providers.CartoDB.Positron,
                   zoom="auto", attribution=False)

    # Wind arrows coloured by speed (turbo)
    q = ax.quiver(mx, my, u_scaled, v_scaled, speed,
                  cmap="turbo", scale=12, width=0.003,
                  headwidth=3, headlength=3.5, headaxislength=3,
                  alpha=0.9, zorder=3)

    cb = fig.colorbar(q, ax=ax, fraction=0.03, pad=0.02, shrink=0.6)
    cb.set_label("Wind Speed (kt)", color="white")
    cb.ax.tick_params(colors="white")

    # Observations
    ax.quiver(obs_mx, obs_my, ou_s, ov_s,
              color="#ff4444", scale=12, width=0.004, zorder=5)
    ax.scatter(obs_mx, obs_my, marker="^", s=80,
               color="#ff4444", edgecolors="white", linewidths=0.6)

    for mx_o, my_o, spd, alt in zip(obs_mx, obs_my, obs_spd, obs_alt):
        ax.text(
            mx_o + 800, my_o + 800,
            f"{spd:.0f} kt | FL{int(alt / 100):03d}",
            fontsize=7, color="#ff8888",
            path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

    ax.set_axis_off()

    title = f"Predicted Wind — FL{int(alt_ft / 100):03d}"
    if snapshot_time:
        title += f" | {format_snapshot_time(snapshot_time)}"
    elif snapshot_id:
        title += f" | snapshot {snapshot_id}"
    ax.set_title(title, color="white", fontsize=13)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved -> {path}")


def _save_uncertainty(path, mx, my, u_scaled, v_scaled,
                      heat_unc, heat_mx, heat_my,
                      obs_mx, obs_my, ou_s, ov_s, obs_spd, obs_alt,
                      x_min, x_max, y_min, y_max,
                      alt_ft, snapshot_time, snapshot_id):
    fig, ax = plt.subplots(figsize=(11, 11))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Uncertainty heatmap
    ax.imshow(
        heat_unc, origin="lower",
        extent=[heat_mx.min(), heat_mx.max(), heat_my.min(), heat_my.max()],
        cmap="inferno", alpha=1.0, interpolation="bilinear", zorder=2)

    # Wind arrows (grey)
    ax.quiver(mx, my, u_scaled, v_scaled,
              color="#cccccc", scale=12, width=0.003,
              headwidth=3, headlength=3.5, headaxislength=3,
              alpha=0.9, zorder=3)

    # Observations
    ax.quiver(obs_mx, obs_my, ou_s, ov_s,
              color="#ff4444", scale=12, width=0.004, zorder=5)
    ax.scatter(obs_mx, obs_my, marker="^", s=80,
               color="#ff4444", edgecolors="white", linewidths=0.6)

    for mx_o, my_o, spd, alt in zip(obs_mx, obs_my, obs_spd, obs_alt):
        ax.text(
            mx_o + 800, my_o + 800,
            f"{spd:.0f} kt | FL{int(alt / 100):03d}",
            fontsize=7, color="#ff8888",
            path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

    # Colourbar
    sm = cm.ScalarMappable(
        cmap="inferno", norm=mcolors.Normalize(vmin=0, vmax=1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, shrink=0.6)
    cb.set_label("Prediction Uncertainty", color="white")
    cb.ax.tick_params(colors="white")

    ax.set_axis_off()

    title = f"Predicted Wind Uncertainty — FL{int(alt_ft / 100):03d}"
    if snapshot_time:
        title += f" | {format_snapshot_time(snapshot_time)}"
    elif snapshot_id:
        title += f" | snapshot {snapshot_id}"
    ax.set_title(title, color="white", fontsize=13)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved -> {path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--alt_ft", type=float, default=35000)
    p.add_argument("--output", default="outputs/imgs/wind_field.png",
                   help=(
                       "Base output path; saves"
                       " _uncertainty and _basemap variants"))
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument('--decoder_layers', type=int, default=None,
                   help='Number of hidden layers in the decoder MLP')
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--grid_lat", type=int, default=25)
    p.add_argument("--grid_lon", type=int, default=25)

    p.add_argument("--radius_km", type=float, default=None,
                   help="Zoom: half-width of plotted area in km from centre")
    p.add_argument("--lat_range_deg", type=float, default=None,
                   help="Zoom: half-width in degrees latitude")
    p.add_argument("--lon_range_deg", type=float, default=None,
                   help="Zoom: half-width in degrees longitude")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cache")
    src.add_argument("--context_json")

    p.add_argument("--snapshot_id", type=int)
    p.add_argument("--split", choices=["val", "test"], default="val")

    args = p.parse_args()

    params = None
    if args.cache:
        context, snapshot_time, sid = pick_snapshot(
            args.cache, snapshot_id=args.snapshot_id, split=args.split)
        params = load_params(args.cache)
    else:
        with open(args.context_json) as f:
            context = json.load(f)
        snapshot_time, sid = None, None

    lat_range_deg = args.lat_range_deg
    lon_range_deg = args.lon_range_deg
    if args.radius_km is not None:
        if lat_range_deg is None:
            lat_range_deg = args.radius_km / KM_PER_DEG_LAT
        if lon_range_deg is None and params is not None:
            lon_range_deg = args.radius_km / params.km_per_deg_lon
        elif lon_range_deg is None:
            lon_range_deg = (args.radius_km
                             / (111.0 * np.cos(np.radians(51.07))))

    if lat_range_deg is None and params is not None:
        lat_range_deg = params.range_km / KM_PER_DEG_LAT
    if lon_range_deg is None and params is not None:
        lon_range_deg = params.range_km / params.km_per_deg_lon

    root, ext = os.path.splitext(args.output)
    uncertainty_path = f"{root}_uncertainty{ext}"
    basemap_path = f"{root}_basemap{ext}"

    plot_wind(
        checkpoint=args.checkpoint,
        alt_ft=args.alt_ft,
        context=context,
        n_samples=args.samples,
        n_lat=args.grid_lat,
        n_lon=args.grid_lon,
        output_uncertainty=uncertainty_path,
        output_basemap=basemap_path,
        num_hidden=args.hidden,
        layers=args.layers,
        num_decoder_layers=args.decoder_layers,
        snapshot_id=sid,
        snapshot_time=snapshot_time,
        params=params,
        lat_range_deg=lat_range_deg,
        lon_range_deg=lon_range_deg,
    )
