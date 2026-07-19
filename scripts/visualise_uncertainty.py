"""
visualise_uncertainty_components.py — Side-by-side aleatoric
vs epistemic uncertainty maps.

Splits the model's total predictive uncertainty into two
components:
  - Aleatoric: E[sigma^2] — learned noise the model can't reduce
  - Epistemic: Var(mu) — uncertainty from latent space

Can run on a single snapshot or averaged over an entire dataset
split. Each panel is independently normalised 0-1 (compare shape,
not raw colour).

Usage:
    python visualise_uncertainty_components.py \
        --checkpoint best.pth.tar --cache data/npy_cache \
        --alt_ft 35000 --output uncertainty.png

    # Average over the val split:
    python visualise_uncertainty_components.py \
        --checkpoint best.pth.tar --cache data/npy_cache \
        --alt_ft 35000 --average --output avg_uncertainty.png
"""

import argparse
import json
import math

import numpy as np
import torch
import matplotlib.pyplot as plt

from wind_map.network import LatentModel
from wind_map.preprocess import (
    normalise_coords, MAX_WIND_KT, day_grouped_split
)
from wind_map.utils import (
    circular_mean, circular_std, pick_snapshot,
    make_grid, wind_to_uv, load_snapshot
)


# --- Model + tensor plumbing ---

def load_model(checkpoint_path, num_hidden, num_layers, device):
    model = LatentModel(num_hidden, num_layers=num_layers).to(device)
    ckpt = torch.load(
        checkpoint_path, map_location=device,
        weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val_loss={ckpt.get('val_loss', float('nan')):.4f})")
    return model


def obs_to_tensors(observations, device):
    xs, ys = [], []
    for obs in observations:
        lat_n, lon_n, alt_n = normalise_coords(
            obs['lat'], obs['lon'], obs['alt_ft'])
        rad = math.radians(obs['wind_dir'])
        xs.append([lat_n, lon_n, alt_n])
        ys.append([
            math.sin(rad), math.cos(rad),
            obs['wind_speed'] / MAX_WIND_KT])
    x = torch.FloatTensor(xs).unsqueeze(0).to(device)
    y = torch.FloatTensor(ys).unsqueeze(0).to(device)
    return x, y


def queries_to_tensor(queries, device):
    xs = [normalise_coords(q['lat'], q['lon'], q['alt_ft']) for q in queries]
    return torch.FloatTensor(xs).unsqueeze(0).to(device)


@torch.no_grad()
def predict_components(model, context, queries, n_samples, device):
    context_x, context_y = obs_to_tensors(context, device)
    target_x = queries_to_tensor(queries, device)

    mu_samples, sigma_samples = [], []
    for _ in range(n_samples):
        mu, sigma, _, _ = model(context_x, context_y, target_x, target_y=None)
        mu_samples.append(mu.squeeze(0).cpu())
        sigma_samples.append(sigma.squeeze(0).cpu())

    mu_stack = torch.stack(mu_samples, dim=0).numpy()
    sigma_stack = torch.stack(sigma_samples, dim=0).numpy()

    sin_mu = mu_stack[..., 0]
    cos_mu = mu_stack[..., 1]
    spd_mu = mu_stack[..., 2]

    sample_dirs = np.degrees(np.arctan2(sin_mu, cos_mu)) % 360
    sample_speeds = np.clip(spd_mu, 0, None) * MAX_WIND_KT

    mean_dirs = circular_mean(sample_dirs, axis=0)
    mean_speeds = sample_speeds.mean(axis=0)

    # Epistemic: spread of mu(z) across z-draws
    epistemic_dir_std = circular_std(sample_dirs, axis=0)
    epistemic_speed_std = sample_speeds.std(axis=0)

    # Aleatoric: delta-method propagation of sigma
    sin_sig = sigma_stack[..., 0]
    cos_sig = sigma_stack[..., 1]
    spd_sig = sigma_stack[..., 2]
    R2 = sin_mu ** 2 + cos_mu ** 2 + 1e-6
    aleatoric_dir_var_rad2 = (
        (cos_mu / R2) ** 2 * sin_sig ** 2
        + (sin_mu / R2) ** 2 * cos_sig ** 2)
    aleatoric_dir_var_deg2 = np.degrees(np.sqrt(aleatoric_dir_var_rad2)) ** 2
    aleatoric_dir_std = np.sqrt(aleatoric_dir_var_deg2.mean(axis=0))

    aleatoric_speed_var = (spd_sig * MAX_WIND_KT) ** 2
    aleatoric_speed_std = np.sqrt(aleatoric_speed_var.mean(axis=0))

    return {
        'wind_dir_deg': mean_dirs,
        'wind_speed_kt': mean_speeds,
        'aleatoric_dir_std': aleatoric_dir_std,
        'aleatoric_speed_std': aleatoric_speed_std,
        'epistemic_dir_std': epistemic_dir_std,
        'epistemic_speed_std': epistemic_speed_std,
    }


def split_snapshot_ids(cache_dir, split):
    train_ids, val_ids, test_ids = day_grouped_split(cache_dir)
    return {"train": train_ids, "val": val_ids, "test": test_ids}[split]


def combined_unc(dir_std, speed_std, n_lat, n_lon):
    """Normalised 0.5*(dir + speed) composite uncertainty map."""
    d = dir_std.reshape(n_lat, n_lon)
    s = speed_std.reshape(n_lat, n_lon)
    return 0.5 * (d / (d.max() + 1e-9) + s / (s.max() + 1e-9))


def compute_single(model, context, alt_ft, n_samples, n_lat, n_lon, device):
    queries, lat_grid, lon_grid, _, _ = make_grid(alt_ft, n_lat, n_lon)
    result = predict_components(model, context, queries, n_samples, device)
    return result, lat_grid, lon_grid


def compute_average(model, cache_dir, split, alt_ft, n_samples, n_lat, n_lon,
                    device, max_snapshots=None):
    ids = split_snapshot_ids(cache_dir, split)
    if max_snapshots is not None:
        ids = ids[:max_snapshots]

    queries, lat_grid, lon_grid, _, _ = make_grid(alt_ft, n_lat, n_lon)

    sums = None
    n_used = 0
    all_ctx_lat, all_ctx_lon = [], []

    for i, sid in enumerate(ids):
        obs, _ = load_snapshot(cache_dir, sid)
        if len(obs) < 2:
            continue

        result = predict_components(model, obs, queries, n_samples, device)
        if sums is None:
            sums = {k: np.zeros_like(v) for k, v in result.items()
                    if isinstance(v, np.ndarray)}
        for k in sums:
            sums[k] += result[k]
        n_used += 1
        all_ctx_lat.extend(o["lat"] for o in obs)
        all_ctx_lon.extend(o["lon"] for o in obs)

        if (i + 1) % 10 == 0 or (i + 1) == len(ids):
            print(f"  {i + 1}/{len(ids)} snapshots processed ({n_used} used)")

    if n_used == 0:
        raise RuntimeError(f"No usable snapshots in '{split}' split.")

    averaged = {k: v / n_used for k, v in sums.items()}
    averaged['n_used'] = n_used
    averaged['all_context_lat'] = np.array(all_ctx_lat)
    averaged['all_context_lon'] = np.array(all_ctx_lon)
    return averaged, lat_grid, lon_grid


# --- Plotting ---

def plot_components(result, lat_grid, lon_grid, alt_ft, n_lat, n_lon, output,
                    context=None, snapshot_time=None, snapshot_id=None,
                    averaged=False, n_used=None, all_context_lat=None,
                    all_context_lon=None):
    aleatoric_map = combined_unc(
        result['aleatoric_dir_std'],
        result['aleatoric_speed_std'], n_lat, n_lon)
    epistemic_map = combined_unc(
        result['epistemic_dir_std'],
        result['epistemic_speed_std'], n_lat, n_lon)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    titles = ["Aleatoric (E[sigma^2])", "Epistemic (Var(mu))"]
    maps = [aleatoric_map, epistemic_map]

    for ax, unc_map, title in zip(axes, maps, titles):
        ax.imshow(
            unc_map, origin="lower",
            extent=[lon_grid.min(), lon_grid.max(),
                    lat_grid.min(), lat_grid.max()],
            cmap="inferno", interpolation="bilinear",
            aspect="auto")

        if not averaged:
            dirs = result['wind_dir_deg'].reshape(n_lat, n_lon)
            speed = result['wind_speed_kt'].reshape(n_lat, n_lon)
            u, v = wind_to_uv(dirs, speed)
            max_spd = speed.max() + 1e-9
            u_s, v_s = u / max_spd, v / max_spd
            ax.quiver(lon_grid, lat_grid, u_s, v_s, color="#cccccc",
                      scale=12, width=0.003, alpha=0.8)
            obs_lat = np.array([o["lat"] for o in context])
            obs_lon = np.array([o["lon"] for o in context])
            ax.scatter(obs_lon, obs_lat, marker="^", s=80, color="#ff4444",
                       edgecolors="white", linewidths=0.6, zorder=5)
        else:
            ax.scatter(all_context_lon, all_context_lat, marker=".", s=3,
                       color="#00ffff", alpha=0.08, zorder=5)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    suptitle = f"Uncertainty decomposition — FL{int(alt_ft / 100):03d}"
    if averaged:
        suptitle += f" | averaged over {n_used} snapshots"
    elif snapshot_time:
        suptitle += f" | snapshot {snapshot_time}"
    elif snapshot_id:
        suptitle += f" | snapshot {snapshot_id}"
    suptitle += "\n(each panel independently normalised 0-1)"
    fig.suptitle(suptitle, fontsize=12)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved -> {output}")


# --- CLI ---

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--alt_ft", type=float, default=35000)
    p.add_argument(
        "--output",
        default="outputs/imgs/uncertainty_components.png")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--grid_lat", type=int, default=25)
    p.add_argument("--grid_lon", type=int, default=25)

    p.add_argument(
        "--average", action="store_true",
        help="Average over every snapshot in --split "
             "instead of one snapshot")
    p.add_argument("--max_snapshots", type=int, default=None,
                   help="Cap snapshots used in --average mode")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cache")
    src.add_argument("--context_json")

    p.add_argument("--snapshot_id", type=int)
    p.add_argument("--split", choices=["train", "val", "test"], default="val")
    args = p.parse_args()

    if args.average and not args.cache:
        p.error("--average requires --cache")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, args.hidden, args.num_layers, device)

    if args.average:
        print(f"Averaging over '{args.split}' split...")
        result, lat_grid, lon_grid = compute_average(
            model, args.cache, args.split, args.alt_ft,
            args.samples, args.grid_lat, args.grid_lon,
            device, max_snapshots=args.max_snapshots)
        plot_components(result, lat_grid, lon_grid, args.alt_ft, args.grid_lat,
                        args.grid_lon, args.output, averaged=True,
                        n_used=result['n_used'],
                        all_context_lat=result['all_context_lat'],
                        all_context_lon=result['all_context_lon'])
    else:
        if args.cache:
            context, snapshot_time, sid = pick_snapshot(
                args.cache, snapshot_id=args.snapshot_id, split=args.split)
        else:
            context = json.load(open(args.context_json))
            snapshot_time, sid = None, None

        result, lat_grid, lon_grid = compute_single(
            model, context, args.alt_ft, args.samples, args.grid_lat,
            args.grid_lon, device)
        plot_components(result, lat_grid, lon_grid, args.alt_ft, args.grid_lat,
                        args.grid_lon, args.output, context=context,
                        snapshot_time=snapshot_time, snapshot_id=sid)
