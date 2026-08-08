"""
train_gp.py — Fit global GP hyperparameters for the wind GP baseline.

Learns the ARD Matérn-5/2 lengthscales, output amplitudes and observation
noises by maximising the sum of per-snapshot marginal log-likelihoods over a
subsample of TRAINING snapshots (the GP analogue of the ANP's shared weights).
Saves them to a JSON file consumed by test_gp.py and visualise_wind_gp.py.

Usage:
    python scripts/train_gp.py --cache data/npy_cache \
        --out checkpoint/gp_params.json --n-snapshots 3000
"""

import argparse
import json
import os
import time

import numpy as np

from wind_map.gp import fit_hyperparameters, cache_x_to_gp_x
from wind_map.preprocess import day_grouped_split, WindSnapshotDataset, load_params


def main():
    p = argparse.ArgumentParser(
        description="Fit global GP kernel hyperparameters for the wind GP baseline.")
    p.add_argument("--cache", required=True,
                   help="Path to the .npy cache built by convert_db.py")
    p.add_argument("--out", default="checkpoint/gp_params.json")
    p.add_argument("--n-snapshots", type=int, default=10000,
                   help="Subsample of training snapshots used for the MLE fit")
    p.add_argument("--max-snapshot-size", type=int, default=16,
                   help="Cap points per snapshot (randomly subsampled)")
    p.add_argument("--n-steps", type=int, default=800,
                   help="Adam optimisation steps")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--split-seed", type=int, default=42,
                   help="Seed for the day-grouped train/val/test split")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for snapshot subsampling and Adam")
    args = p.parse_args()

    params = load_params(args.cache)
    train_ids, _, _ = day_grouped_split(args.cache, seed=args.split_seed)
    ds = WindSnapshotDataset(args.cache, snapshot_ids=train_ids)
    print(f"{len(ds)} training snapshots available")

    snapshots = []
    for i in range(len(ds)):
        x, y = ds[i]
        m = x.shape[0]
        if m < 2:
            continue
        if m > args.max_snapshot_size:
            idx = np.random.default_rng(i).choice(
                m, size=args.max_snapshot_size, replace=False)
            x, y = x[idx], y[idx]
        snapshots.append((
            cache_x_to_gp_x(x.numpy(), params),
            y.numpy().astype(np.float64),
        ))

    rng = np.random.default_rng(args.seed)
    if args.n_snapshots and args.n_snapshots < len(snapshots):
        sel = rng.choice(len(snapshots), size=args.n_snapshots, replace=False)
        snapshots = [snapshots[i] for i in sel]

    n_obs = sum(x.shape[0] for x, _ in snapshots)
    print(f"Fitting on {len(snapshots)} snapshots "
          f"({n_obs} observations, input_space=km)")

    t0 = time.time()
    lengthscales, amplitudes, noises, mll_per_pt = fit_hyperparameters(
        snapshots, n_steps=args.n_steps, lr=args.lr, seed=args.seed)
    dt = time.time() - t0

    print(f"Fit in {dt:.1f}s  mll={mll_per_pt:+.4f} nats/point")
    print(f"  lengthscales (km): [{', '.join(f'{v:.1f}' for v in lengthscales)}]")
    print(f"  amplitudes:        [{', '.join(f'{v:.3f}' for v in amplitudes)}]")
    print(f"  noises:            [{', '.join(f'{v:.3f}' for v in noises)}]")

    out = {
        "lengthscales": [float(v) for v in lengthscales],
        "amplitudes": [float(v) for v in amplitudes],
        "noises": [float(v) for v in noises],
        "avg_mll": float(mll_per_pt),
        "n_snapshots": len(snapshots),
        "n_observations": int(n_obs),
        "split_seed": args.split_seed,
        "input_space": "km",
        "norm_params": params.to_dict(),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
