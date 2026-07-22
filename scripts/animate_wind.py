"""
animate_wind.py — Animate predicted wind as advected particles,
saved as a GIF.

Same model/snapshot path as visualise_wind.py, but renders a
particle swarm advected by the predicted wind field instead of a
static quiver. Particle colour encodes local uncertainty
(inferno scale).

The wind field is predicted ONCE on a single grid. Particles are
advected between grid points via bilinear interpolation, so no
repeated model inference per frame.

Seamless looping: each particle gets a random phase offset into a
shared trajectory buffer. At frame f, particle p reads from row
(f + phase[p]) % n_frames. This guarantees bit-identical
wrap-around by construction.

Usage:
    python animate_wind.py --checkpoint checkpoint/best_model.pth.tar \
        --cache data/npy_cache --alt_ft 35000 \
        --output outputs/imgs/wind_flow.gif
"""

import argparse
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.cm as cm  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import contextily as cx  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.animation import (  # noqa: E402
    FuncAnimation, PillowWriter)
from scipy.interpolate import RegularGridInterpolator  # noqa: E402

from wind_map.infer import WindPredictor  # noqa: E402
from wind_map.preprocess import (  # noqa: E402
    CENTRE_LAT, CENTRE_LON, LAT_RANGE_DEG, LON_RANGE_DEG)
from wind_map.utils import (  # noqa: E402
    format_snapshot_time, lonlat_to_mercator, pick_snapshot,
    make_grid, wind_to_uv)

KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 111.0 * np.cos(np.radians(CENTRE_LAT))
KT_TO_KM_PER_S = 1.852 / 3600.0


# ---------------------------------------------------------------------------
# Particle simulation
# ---------------------------------------------------------------------------

class ParticleSwarm:
    """Particles in (lat, lon) degrees, advected by wind."""

    def __init__(self, n_particles, lat_bounds, lon_bounds,
                 u_interp, v_interp, unc_interp, rng,
                 max_age_frames=60, trail_len=8):
        self.n = n_particles
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        self.u_interp = u_interp
        self.v_interp = v_interp
        self.unc_interp = unc_interp
        self.rng = rng
        self.max_age_frames = max_age_frames
        self.trail_len = trail_len

        self.lat = rng.uniform(*lat_bounds, size=n_particles)
        self.lon = rng.uniform(*lon_bounds, size=n_particles)
        self.age = rng.integers(0, max_age_frames, size=n_particles)
        self.hist_lat = np.tile(self.lat, (trail_len, 1))
        self.hist_lon = np.tile(self.lon, (trail_len, 1))
        # Uncertainty sampled at each trail point (not just the head)
        init_unc = unc_interp(np.stack([self.lat, self.lon], axis=-1))
        self.hist_unc = np.tile(init_unc, (trail_len, 1))

    def _respawn(self, mask):
        k = mask.sum()
        if k == 0:
            return
        new_lat = self.rng.uniform(*self.lat_bounds, size=k)
        new_lon = self.rng.uniform(*self.lon_bounds, size=k)
        self.lat[mask] = new_lat
        self.lon[mask] = new_lon
        self.age[mask] = 0
        new_unc = self.unc_interp(np.stack([new_lat, new_lon], axis=-1))
        for h in range(self.trail_len):
            self.hist_lat[h, mask] = new_lat
            self.hist_lon[h, mask] = new_lon
            self.hist_unc[h, mask] = new_unc

    def step(self, dt_seconds):
        pts = np.stack([self.lat, self.lon], axis=-1)
        u_kt = self.u_interp(pts)
        v_kt = self.v_interp(pts)

        dlat = (v_kt * KT_TO_KM_PER_S * dt_seconds) / KM_PER_DEG_LAT
        dlon = (u_kt * KT_TO_KM_PER_S * dt_seconds) / KM_PER_DEG_LON

        self.lat = self.lat + dlat
        self.lon = self.lon + dlon
        self.age += 1

        self.hist_lat = np.roll(self.hist_lat, -1, axis=0)
        self.hist_lon = np.roll(self.hist_lon, -1, axis=0)
        self.hist_unc = np.roll(self.hist_unc, -1, axis=0)
        self.hist_lat[-1] = self.lat
        self.hist_lon[-1] = self.lon
        self.hist_unc[-1] = self.unc_interp(
            np.stack([self.lat, self.lon], axis=-1))

        out_of_bounds = (
            (self.lat < self.lat_bounds[0])
            | (self.lat > self.lat_bounds[1])
            | (self.lon < self.lon_bounds[0])
            | (self.lon > self.lon_bounds[1])
        )
        too_old = self.age >= self.max_age_frames
        self._respawn(out_of_bounds | too_old)

    def uncertainty_now(self):
        return self.hist_unc[-1]


# ---------------------------------------------------------------------------

def build_wind_gif(checkpoint, alt_ft, context, n_samples,
                   n_lat, n_lon, output,
                   num_hidden, num_layers,
                   n_particles, n_frames, fps, dt_seconds, trail_len,
                   snapshot_id=None, snapshot_time=None,
                   lat_range_deg=None, lon_range_deg=None, seed=0):
    predictor = WindPredictor(
        checkpoint, num_hidden=num_hidden,
        num_layers=num_layers)

    eff_lat_range = (lat_range_deg if lat_range_deg is not None
                     else LAT_RANGE_DEG)
    eff_lon_range = (lon_range_deg if lon_range_deg is not None
                     else LON_RANGE_DEG)

    if eff_lat_range > LAT_RANGE_DEG or eff_lon_range > LON_RANGE_DEG:
        print(
            f"Warning: view (+-{eff_lat_range:.3f} lat "
            f"/ +-{eff_lon_range:.3f} lon) extends beyond "
            f"training range. Predictions may be unreliable.")

    # Single grid prediction for wind + uncertainty fields
    queries, lat_grid, lon_grid, lat_axis, lon_axis = make_grid(
        alt_ft, n_lat, n_lon, eff_lat_range, eff_lon_range)
    result = predictor.predict(context, queries, n_samples=n_samples)

    dirs = result["wind_dir_deg"].reshape(n_lat, n_lon)
    speed = result["wind_speed_kt"].reshape(n_lat, n_lon)
    dir_std = result["wind_dir_std"].reshape(n_lat, n_lon)
    spd_std = result["wind_speed_std"].reshape(n_lat, n_lon)

    u, v = wind_to_uv(dirs, speed)

    unc = 0.5 * (
            dir_std / (dir_std.max() + 1e-9) +
            spd_std / (spd_std.max() + 1e-9))

    u_interp = RegularGridInterpolator((lat_axis, lon_axis), u,
                                       bounds_error=False, fill_value=0.0)
    v_interp = RegularGridInterpolator((lat_axis, lon_axis), v,
                                       bounds_error=False, fill_value=0.0)
    unc_interp = RegularGridInterpolator((lat_axis, lon_axis), unc,
                                         bounds_error=False, fill_value=0.0)

    lat_bounds = (lat_axis.min(), lat_axis.max())
    lon_bounds = (lon_axis.min(), lon_axis.max())

    rng = np.random.default_rng(seed)
    swarm = ParticleSwarm(n_particles, lat_bounds, lon_bounds,
                          u_interp, v_interp, unc_interp, rng,
                          max_age_frames=max(30, n_frames // 2),
                          trail_len=trail_len)

    # Precompute forward-only trajectory buffer, then loop
    # via per-particle phase offsets
    all_lat = np.empty((n_frames, n_particles))
    all_lon = np.empty((n_frames, n_particles))
    all_hist_lat = np.empty((n_frames, trail_len, n_particles))
    all_hist_lon = np.empty((n_frames, trail_len, n_particles))
    all_hist_unc = np.empty((n_frames, trail_len, n_particles))

    for f in range(n_frames):
        swarm.step(dt_seconds)
        all_lat[f] = swarm.lat
        all_lon[f] = swarm.lon
        all_hist_lat[f] = swarm.hist_lat
        all_hist_lon[f] = swarm.hist_lon
        all_hist_unc[f] = np.clip(swarm.hist_unc, 0, 1)

    phase = rng.integers(0, n_frames, size=n_particles)
    particle_idx = np.arange(n_particles)

    # Figure setup
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#0d1b2a")
    ax.set_facecolor("#0d1b2a")

    x_min, y_min = lonlat_to_mercator(CENTRE_LON - eff_lon_range,
                                      CENTRE_LAT - eff_lat_range)
    x_max, y_max = lonlat_to_mercator(CENTRE_LON + eff_lon_range,
                                      CENTRE_LAT + eff_lat_range)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_axis_off()

    cx.add_basemap(ax, crs="EPSG:3857",
                   source=cx.providers.CartoDB.Positron,
                   zoom="auto", attribution=False)

    norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = matplotlib.colormaps["inferno"]

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, shrink=0.6)
    cb.set_label("Prediction Uncertainty", color="white")

    # Observations
    obs_lat = np.array([o["lat"] for o in context])
    obs_lon = np.array([o["lon"] for o in context])
    obs_dir = np.array([o["wind_dir"] for o in context])
    obs_spd = np.array([o["wind_speed"] for o in context])
    obs_alt = np.array([o["alt_ft"] for o in context])

    obs_mx, obs_my = lonlat_to_mercator(obs_lon, obs_lat)
    ou, ov = wind_to_uv(obs_dir, obs_spd)
    max_spd = speed.max() + 1e-9
    ou_s = ou / max_spd
    ov_s = ov / max_spd

    ax.quiver(obs_mx, obs_my, ou_s, ov_s,
              color="#ff4444", scale=12, width=0.004, zorder=5)
    ax.scatter(obs_mx, obs_my, marker="^", s=80,
               color="#ff4444", edgecolors="white", linewidths=0.6, zorder=5)

    for mx_o, my_o, spd, alt in zip(obs_mx, obs_my, obs_spd, obs_alt):
        ax.text(
            mx_o + 800, my_o + 800,
            f"{spd:.0f} kt | FL{int(alt / 100):03d}",
            fontsize=7, color="#ff8888",
            path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
            zorder=6)

    title = f"Predicted Wind Flow — FL{int(alt_ft / 100):03d}"
    if snapshot_time:
        title += f" | {format_snapshot_time(snapshot_time)}"
    elif snapshot_id:
        title += f" | snapshot {snapshot_id}"
    ax.set_title(title, color="white", fontsize=13)

    trail_collection = LineCollection([], zorder=4)
    ax.add_collection(trail_collection)
    scatter = ax.scatter([], [], s=6, zorder=7, linewidths=0)

    trail_alphas = np.linspace(0.0, 0.9, trail_len)

    def init():
        trail_collection.set_segments([])
        scatter.set_offsets(np.empty((0, 2)))
        return trail_collection, scatter

    def update(f):
        row = (f + phase) % n_frames

        lat = all_lat[row, particle_idx]
        lon = all_lon[row, particle_idx]

        hist_row = np.broadcast_to(
            row[None, :], (trail_len, n_particles))
        hist_lat = np.take_along_axis(
            all_hist_lat, hist_row[None, :, :], axis=0)[0]
        hist_lon = np.take_along_axis(
            all_hist_lon, hist_row[None, :, :], axis=0)[0]
        hist_unc = np.take_along_axis(
            all_hist_unc, hist_row[None, :, :], axis=0)[0]

        head_colors = cmap(norm(hist_unc[-1]))

        mx, my = lonlat_to_mercator(lon, lat)
        scatter.set_offsets(np.column_stack([mx, my]))
        scatter.set_color(head_colors)

        # Trail segments coloured by uncertainty at each point
        hist_mx, hist_my = lonlat_to_mercator(hist_lon, hist_lat)
        seg_colors_full = cmap(norm(hist_unc))

        # Vectorised: build all segments at once
        # starts[i] = (hist_mx[i], hist_my[i])
        # ends[i] = (hist_mx[i+1], hist_my[i+1])
        starts = np.stack(
            [hist_mx[:-1], hist_my[:-1]], axis=-1)
        ends = np.stack(
            [hist_mx[1:], hist_my[1:]], axis=-1)
        segments = np.stack([starts, ends], axis=-2).reshape(-1, 2, 2)

        # Colors: RGB from end-point uncertainty,
        # alpha from trail position
        seg_colors = seg_colors_full[1:].copy()
        seg_colors[:, :, 3] = trail_alphas[:-1, None]
        seg_colors = seg_colors.reshape(-1, 4)

        trail_collection.set_segments(segments)
        trail_collection.set_color(seg_colors)
        trail_collection.set_linewidth(1.0)

        return trail_collection, scatter

    anim = FuncAnimation(fig, update, init_func=init, frames=n_frames,
                         interval=1000 / fps, blit=True)

    writer = PillowWriter(fps=fps)
    anim.save(
        output, writer=writer,
        savefig_kwargs={"facecolor": fig.get_facecolor()})
    plt.close(fig)

    print(f"Saved -> {output} ({n_frames} frames, seamless forward loop)")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--alt_ft", type=float, default=35000)
    p.add_argument("--output", default="outputs/imgs/wind_flow.gif")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--grid_lat", type=int, default=45)
    p.add_argument("--grid_lon", type=int, default=45)

    p.add_argument("--n_particles", type=int, default=1000)
    p.add_argument("--n_frames", type=int, default=500)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dt_seconds", type=float, default=45.0,
                   help="Simulated seconds of advection per frame")
    p.add_argument("--trail_len", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--radius_km", type=float, default=None)
    p.add_argument("--lat_range_deg", type=float, default=None)
    p.add_argument("--lon_range_deg", type=float, default=None)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cache")
    src.add_argument("--context_json")

    p.add_argument("--snapshot_id", type=int)
    p.add_argument("--split", choices=["val", "test"], default="val")

    args = p.parse_args()

    if args.cache:
        context, snapshot_time, sid = pick_snapshot(
            args.cache, snapshot_id=args.snapshot_id, split=args.split)
    else:
        context = json.load(open(args.context_json))
        snapshot_time, sid = None, None

    lat_range_deg = args.lat_range_deg
    lon_range_deg = args.lon_range_deg
    if args.radius_km is not None:
        if lat_range_deg is None:
            lat_range_deg = args.radius_km / KM_PER_DEG_LAT
        if lon_range_deg is None:
            lon_range_deg = args.radius_km / KM_PER_DEG_LON

    build_wind_gif(
        checkpoint=args.checkpoint,
        alt_ft=args.alt_ft,
        context=context,
        n_samples=args.samples,
        n_lat=args.grid_lat,
        n_lon=args.grid_lon,
        output=args.output,
        num_hidden=args.hidden,
        num_layers=args.num_layers,
        n_particles=args.n_particles,
        n_frames=args.n_frames,
        fps=args.fps,
        dt_seconds=args.dt_seconds,
        trail_len=args.trail_len,
        snapshot_id=sid,
        snapshot_time=snapshot_time,
        lat_range_deg=lat_range_deg,
        lon_range_deg=lon_range_deg,
        seed=args.seed,
    )
