"""
visualise_wind_gp.py — Same quiver + speed/direction uncertainty maps
as visualise_wind.py, but using the GP baseline instead of the ANP.

Reuses make_grid / wind_to_uv / lonlat_to_mercator from wind_map.utils and the
array-level renderers _save_basemap / _save_uncertainty from visualise_wind.py,
so the output looks identical and is directly comparable.

Saves three images from each run:
  - wind_field_gp_uncertainty_speed.png     — inferno speed-uncertainty
                                  heatmap behind the quiver (grey arrows)
  - wind_field_gp_uncertainty_direction.png — inferno direction-uncertainty
                                  heatmap behind the quiver (grey arrows)
  - wind_field_gp_basemap.png               — geographical basemap behind the
                                  quiver (speed-coloured arrows)

Usage:
    python scripts/visualise_wind_gp.py --params checkpoint/gp_params.json \
        --cache data/npy_cache --alt_ft 35000 \
        --output outputs/imgs/wind_field_gp.png
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from wind_map.gp import GaussianProcessPredictor  # noqa: E402
from wind_map.preprocess import load_params, KM_PER_DEG_LAT  # noqa: E402
from wind_map.utils import (  # noqa: E402
    pick_snapshot, make_grid, lonlat_to_mercator, wind_to_uv,
)
from scripts.visualise_wind import (  # noqa: E402
    _save_basemap, _save_uncertainty,
)


def plot_wind_gp(params_path, alt_ft, context, n_lat, n_lon,
                 output_uncertainty_speed, output_uncertainty_direction,
                 output_basemap,
                 snapshot_id=None, snapshot_time=None,
                 norm_params=None, lat_range_deg=None, lon_range_deg=None):
    predictor = GaussianProcessPredictor(params_path, params=norm_params)

    if norm_params is not None:
        default_lat_range = norm_params.range_km / KM_PER_DEG_LAT
        default_lon_range = norm_params.range_km / norm_params.km_per_deg_lon
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
        print("Warning: view extends beyond training range; "
              "predictions outside that box are extrapolated.")

    # Main grid (arrows)
    queries, lat_grid, lon_grid, _, _ = make_grid(
        alt_ft, n_lat, n_lon, params=norm_params,
        lat_range_deg=eff_lat_range, lon_range_deg=eff_lon_range)
    result = predictor.predict(context, queries, n_samples=1)

    dirs = result["wind_dir_deg"].reshape(n_lat, n_lon)
    speed = result["wind_speed_kt"].reshape(n_lat, n_lon)
    u, v = wind_to_uv(dirs, speed)

    mx, my = lonlat_to_mercator(lon_grid, lat_grid)

    max_spd = speed.max() + 1e-9
    u_scaled = u / max_spd
    v_scaled = v / max_spd

    obs_lat = np.array([o["lat"] for o in context])
    obs_lon = np.array([o["lon"] for o in context])
    obs_dir = np.array([o["wind_dir"] for o in context])
    obs_spd = np.array([o["wind_speed"] for o in context])
    obs_alt = np.array([o["alt_ft"] for o in context])

    obs_mx, obs_my = lonlat_to_mercator(obs_lon, obs_lat)
    ou, ov = wind_to_uv(obs_dir, obs_spd)
    ou_s = ou / max_spd
    ov_s = ov / max_spd

    c_lat = norm_params.centre_lat if norm_params is not None else 51.071066
    c_lon = norm_params.centre_lon if norm_params is not None else -1.042441
    x_min, y_min = lonlat_to_mercator(c_lon - eff_lon_range,
                                      c_lat - eff_lat_range)
    x_max, y_max = lonlat_to_mercator(c_lon + eff_lon_range,
                                      c_lat + eff_lat_range)

    # Basemap image
    _save_basemap(
        output_basemap, mx, my, u_scaled, v_scaled, speed,
        obs_mx, obs_my, ou_s, ov_s, obs_spd, obs_alt,
        x_min, x_max, y_min, y_max,
        alt_ft, snapshot_time, snapshot_id,
    )

    # High-res heatmap grid
    heat_queries, heat_lat, heat_lon, _, _ = make_grid(
        alt_ft, n_lat * 3, n_lon * 3, params=norm_params,
        lat_range_deg=eff_lat_range, lon_range_deg=eff_lon_range)
    heat_result = predictor.predict(context, heat_queries, n_samples=1)

    heat_dir_std = heat_result["wind_dir_std"].reshape(n_lat * 3, n_lon * 3)
    heat_spd_std = heat_result["wind_speed_std"].reshape(n_lat * 3, n_lon * 3)

    def _pct_normalise(arr, lo=2, hi=98):
        vmin = np.percentile(arr, lo)
        vmax = np.percentile(arr, hi)
        return np.clip((arr - vmin) / (vmax - vmin + 1e-9), 0, 1)

    heat_dir_unc = _pct_normalise(heat_dir_std)
    heat_spd_unc = _pct_normalise(heat_spd_std)

    heat_mx, heat_my = lonlat_to_mercator(heat_lon, heat_lat)

    # Uncertainty images
    _save_uncertainty(
        output_uncertainty_speed, mx, my, u_scaled, v_scaled,
        heat_spd_unc, heat_mx, heat_my,
        obs_mx, obs_my, ou_s, ov_s, obs_spd, obs_alt,
        x_min, x_max, y_min, y_max,
        alt_ft, snapshot_time, snapshot_id,
        component="Speed")
    _save_uncertainty(
        output_uncertainty_direction, mx, my, u_scaled, v_scaled,
        heat_dir_unc, heat_mx, heat_my,
        obs_mx, obs_my, ou_s, ov_s, obs_spd, obs_alt,
        x_min, x_max, y_min, y_max,
        alt_ft, snapshot_time, snapshot_id,
        component="Direction")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Plot predicted wind field from the GP baseline.")
    p.add_argument("--params", required=True,
                   help="Path to gp_params.json from train_gp.py")
    p.add_argument("--alt_ft", type=float, default=35000)
    p.add_argument("--output", default="outputs/imgs/wind_field_gp.png",
                   help="Base output path; saves _uncertainty_speed,"
                        " _uncertainty_direction and _basemap variants")
    p.add_argument("--samples", type=int, default=1000,
                   help="Accepted for parity with visualise_wind.py (ignored)")
    p.add_argument("--grid_lat", type=int, default=25)
    p.add_argument("--grid_lon", type=int, default=25)
    p.add_argument("--radius_km", type=float, default=None)
    p.add_argument("--lat_range_deg", type=float, default=None)
    p.add_argument("--lon_range_deg", type=float, default=None)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cache")
    src.add_argument("--context_json")

    p.add_argument("--snapshot_id", type=int)
    p.add_argument("--split", choices=["val", "test"], default="val")

    args = p.parse_args()

    norm_params = None
    if args.cache:
        context, snapshot_time, sid = pick_snapshot(
            args.cache, snapshot_id=args.snapshot_id, split=args.split)
        norm_params = load_params(args.cache)
    else:
        with open(args.context_json) as f:
            context = json.load(f)
        snapshot_time, sid = None, None

    lat_range_deg = args.lat_range_deg
    lon_range_deg = args.lon_range_deg
    if args.radius_km is not None:
        if lat_range_deg is None:
            lat_range_deg = args.radius_km / KM_PER_DEG_LAT
        if lon_range_deg is None and norm_params is not None:
            lon_range_deg = args.radius_km / norm_params.km_per_deg_lon
        elif lon_range_deg is None:
            lon_range_deg = (args.radius_km
                             / (111.0 * np.cos(np.radians(51.07))))

    if lat_range_deg is None and norm_params is not None:
        lat_range_deg = norm_params.range_km / KM_PER_DEG_LAT
    if lon_range_deg is None and norm_params is not None:
        lon_range_deg = norm_params.range_km / norm_params.km_per_deg_lon

    root, ext = os.path.splitext(args.output)
    uncertainty_speed_path = f"{root}_uncertainty_speed{ext}"
    uncertainty_direction_path = f"{root}_uncertainty_direction{ext}"
    basemap_path = f"{root}_basemap{ext}"

    plot_wind_gp(
        params_path=args.params,
        alt_ft=args.alt_ft,
        context=context,
        n_lat=args.grid_lat,
        n_lon=args.grid_lon,
        output_uncertainty_speed=uncertainty_speed_path,
        output_uncertainty_direction=uncertainty_direction_path,
        output_basemap=basemap_path,
        snapshot_id=sid,
        snapshot_time=snapshot_time,
        norm_params=norm_params,
        lat_range_deg=lat_range_deg,
        lon_range_deg=lon_range_deg,
    )
