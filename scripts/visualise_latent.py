"""
visualise_latent.py — UMAP projection of the learned latent space,
coloured by physical conditions.

Encodes every snapshot's observations through the latent encoder to
obtain a 64-dim mu vector, projects to 2D via UMAP, and produces a
multi-panel figure with each panel coloured by a different snapshot
property (hour of day, wind speed, direction, etc.).

Usage:
    python scripts/visualise_latent.py \
        --checkpoint checkpoint/best_model.pth.tar \
        --cache data/npy_cache \
        --output outputs/imgs/latent_umap.png
"""

import argparse
from datetime import datetime, timezone

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from torch.utils.data import DataLoader
import umap

from wind_map.infer import load_model_checkpoint
from wind_map.preprocess import (
    WindSnapshotDataset, day_grouped_split,
    load_params,
)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _collate_pad(batch):
    """Minimal collate: pad (x, y) pairs and return context tensors."""
    xs, ys = [], []
    for x, y in batch:
        xs.append(x)
        ys.append(y)
    lens = [x.size(0) for x in xs]
    max_len = max(lens)
    x_dim = xs[0].size(-1)
    y_dim = ys[0].size(-1)
    B = len(xs)
    cx = xs[0].new_zeros(B, max_len, x_dim)
    cy = ys[0].new_zeros(B, max_len, y_dim)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i in range(B):
        n = lens[i]
        cx[i, :n] = xs[i]
        cy[i, :n] = ys[i]
        mask[i, :n] = True
    return cx, cy, mask


@torch.no_grad()
def encode_snapshots(model, dataset, batch_size=128, device='cpu',
                     max_snapshots=None):
    """Encode all snapshots through the latent encoder.

    Returns:
        mu_array  [N, num_latents]  — latent mean vectors
        meta      dict of per-snapshot metadata arrays
    """
    model.eval()
    if max_snapshots is not None and len(dataset) > max_snapshots:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(dataset), size=max_snapshots, replace=False)
        subset = torch.utils.data.Subset(dataset, indices)
    else:
        subset = dataset

    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=_collate_pad)

    all_mu = []
    all_n_obs = []
    all_speed_mean = []
    all_dir_mean = []
    all_speed_std = []
    all_lat_spread = []

    for cx, cy, mask in loader:
        cx = cx.to(device)
        cy = cy.to(device)
        mask = mask.to(device)

        dist = model.latent_encoder(cx, cy, mask=mask)
        mu = dist.loc.cpu().numpy()
        all_mu.append(mu)

        u_raw = cy[:, :, 0].numpy() * params.u_std + params.u_mean
        v_raw = cy[:, :, 1].numpy() * params.v_std + params.v_mean
        speed_kt = np.sqrt(u_raw**2 + v_raw**2)
        dir_deg = np.degrees(np.arctan2(-u_raw, -v_raw)) % 360

        valid = mask.numpy()
        B = cx.size(0)
        for i in range(B):
            v = valid[i]
            sp = speed_kt[i, v]
            dd = dir_deg[i, v]
            all_n_obs.append(int(v.sum()))
            all_speed_mean.append(float(np.mean(sp)))
            all_dir_mean.append(float(np.degrees(np.arctan2(
                np.sin(np.radians(dd)).mean(),
                np.cos(np.radians(dd)).mean())) % 360))
            all_speed_std.append(float(np.std(sp)) if len(sp) > 1 else 0.0)
            lat_vals = cx[i, v, 0].cpu().numpy()
            all_lat_spread.append(float(lat_vals.max() - lat_vals.min())
                                  if len(lat_vals) > 1 else 0.0)

    mu_array = np.concatenate(all_mu, axis=0)
    meta = {
        'n_obs': np.array(all_n_obs),
        'speed_mean': np.array(all_speed_mean),
        'dir_mean': np.array(all_dir_mean),
        'speed_std': np.array(all_speed_std),
        'lat_spread': np.array(all_lat_spread),
    }
    return mu_array, meta


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

FACECOLOR = '#0d1b2a'


def _add_panel(ax, embedding, values, cmap, label, vmin=None, vmax=None,
               cmap_type='sequential'):
    """Scatter one UMAP panel coloured by values."""
    if cmap_type == 'cyclic':
        norm = mcolors.Normalize(vmin=0, vmax=360)
    else:
        if vmin is None:
            vmin = np.percentile(values, 2)
        if vmax is None:
            vmax = np.percentile(values, 98)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    sc = ax.scatter(
        embedding[:, 0], embedding[:, 1],
        c=values, cmap=cmap, norm=norm,
        s=3, alpha=0.7, edgecolors='none', rasterized=True)

    cb = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cb.set_label(label, color='white', fontsize=9)
    cb.ax.tick_params(colors='white', labelsize=7)

    ax.set_facecolor(FACECOLOR)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('#333333')


def plot_latent(embedding, meta, snapshot_times, out_path, dpi=150):
    """Create the 2x3 panel UMAP figure."""
    hours = np.array([
        datetime.fromtimestamp(
            float(t), tz=timezone.utc).hour
        for t in snapshot_times
    ], dtype=float)

    doys = np.array([
        datetime.fromtimestamp(
            float(t), tz=timezone.utc).timetuple().tm_yday
        for t in snapshot_times
    ], dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor(FACECOLOR)
    fig.subplots_adjust(hspace=0.12, wspace=0.18)

    panels = [
        (axes[0, 0], hours, 'twilight', 'Hour of Day',
         'Cyclic', 0, 24),
        (axes[0, 1], meta['speed_mean'], 'viridis',
         'Mean Wind Speed (kt)', 'Sequential', None, None),
        (axes[0, 2], meta['dir_mean'], 'hsv',
         'Mean Wind Direction (deg)', 'Cyclic', 0, 360),
        (axes[1, 0], meta['n_obs'].astype(float), 'plasma',
         '# Observations', 'Sequential', None, None),
        (axes[1, 1], meta['speed_std'], 'inferno',
         'Wind Speed Std (kt)', 'Sequential', None, None),
        (axes[1, 2], doys, 'coolwarm',
         'Day of Year', 'Sequential', None, None),
    ]

    for ax, values, cmap, label, cmap_type, vmin, vmax in panels:
        _add_panel(ax, embedding, values, cmap, label,
                   vmin=vmin, vmax=vmax, cmap_type=cmap_type)

    fig.suptitle(
        'Latent Space UMAP — coloured by physical conditions',
        color='white', fontsize=14, y=0.97)

    plt.savefig(out_path, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='UMAP projection of latent space coloured by conditions')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--output', default='outputs/imgs/latent_umap.png')
    p.add_argument('--split', choices=['train', 'val', 'test', 'all'],
                   default='all')
    p.add_argument('--max_snapshots', type=int, default=5000,
                   help='Max snapshots to encode (0 = all)')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--n_neighbors', type=int, default=15,
                   help='UMAP n_neighbors parameter')
    p.add_argument('--min_dist', type=float, default=0.1,
                   help='UMAP min_dist parameter')
    p.add_argument('--dpi', type=int, default=150)
    p.add_argument('--hidden', type=int, default=None)
    p.add_argument('--num_latents', type=int, default=None)
    p.add_argument('--latent_layers', type=int, default=None)
    p.add_argument('--deterministic_layers', type=int, default=None)
    p.add_argument('--decoder_layers', type=int, default=None)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading model from {args.checkpoint} ...")
    model, ckpt = load_model_checkpoint(
        args.checkpoint, device,
        num_hidden=args.hidden,
        num_latents=args.num_latents,
        latent_layers=args.latent_layers,
        deterministic_layers=args.deterministic_layers,
        num_decoder_layers=args.decoder_layers)

    params = load_params(args.cache)

    train_ids, val_ids, test_ids = day_grouped_split(args.cache)
    if args.split == 'all':
        ids = train_ids + val_ids + test_ids
    else:
        ids = {'train': train_ids, 'val': val_ids,
               'test': test_ids}[args.split]
    print(f"Using {len(ids)} snapshots from '{args.split}' split")

    dataset = WindSnapshotDataset(args.cache, snapshot_ids=ids)
    snapshot_times = np.array(dataset.snapshot_times)

    max_n = args.max_snapshots if args.max_snapshots > 0 else None
    print(f"Encoding snapshots (max={max_n or 'all'}, "
          f"batch={args.batch_size}) ...")
    mu_array, meta = encode_snapshots(
        model, dataset, batch_size=args.batch_size, device=device,
        max_snapshots=max_n)

    # If subsampled, also subsample snapshot_times to match
    if max_n is not None and len(snapshot_times) > max_n:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(snapshot_times), size=max_n, replace=False)
        snapshot_times = snapshot_times[indices]

    print(f"Running UMAP (n_neighbors={args.n_neighbors}, "
          f"min_dist={args.min_dist}) on {mu_array.shape[0]} points ...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric='euclidean',
        random_state=42)
    embedding = reducer.fit_transform(mu_array)
    print(f"UMAP embedding shape: {embedding.shape}")

    plot_latent(embedding, meta, snapshot_times,
                args.output, dpi=args.dpi)
