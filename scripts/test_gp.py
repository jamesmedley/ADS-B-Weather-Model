"""
test_gp.py — Held-out evaluation for the GP baseline, metric-for-metric
comparable to scripts/test.py for the ANP.

Uses the SAME day-grouped split, SAME WindSnapshotDataset, SAME
collate_fn_val context/target splitter and SAME eval_seed as scripts/test.py,
so the context/target assignment is identical. The GP is fit per-snapshot on
the context and evaluated on the held-out targets (exact predictive Gaussian,
no latent sampling).

NOTE: scripts/test.py reports a K-sample marginal NLL (averaged over latent
draws); the GP NLL is exact. Compare wind_speed MAE/RMSE, wind_dir MAE and
coverage primarily.

Usage:
    python scripts/test_gp.py --params checkpoint/gp_params.json \
        --cache data/npy_cache
"""

import argparse
import json
import math

import numpy as np
import torch as t
from torch.utils.data import DataLoader

from wind_map.gp import GaussianProcessRegressor, cache_x_to_gp_x
from wind_map.uncertainty import uv_to_speed_dir_std
from wind_map.preprocess import (
    WindSnapshotDataset, day_grouped_split, load_params, collate_fn_val,
)


@t.no_grad()
def evaluate(params_path, cache_dir, split="test", context_frac=0.5,
             batch_size=16, split_seed=42, eval_seed=0, verbose=True):
    """Return the same metric keys as scripts/test.py for the GP."""
    with open(params_path) as f:
        hp = json.load(f)
    reg = GaussianProcessRegressor(
        hp["lengthscales"], hp["amplitudes"], hp["noises"])
    params = load_params(cache_dir)

    train_ids, val_ids, test_ids = day_grouped_split(
        cache_dir, seed=split_seed)
    ids = {"train": train_ids, "val": val_ids, "test": test_ids}[split]
    ds = WindSnapshotDataset(cache_dir, snapshot_ids=ids)

    # Same seed/order as scripts/test.py -> identical context/target splits
    t.manual_seed(eval_seed)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda b: collate_fn_val(b, context_frac=context_frac))

    nll_sum, nll_count = 0.0, 0
    abs_speed_err_sum, sq_speed_err_sum = 0.0, 0.0
    dir_err_sum = 0.0
    cov68_dir = cov95_dir = cov68_spd = cov95_spd = 0.0
    n_snapshots = 0

    for (context_x, context_y, target_x, target_y,
         context_mask, target_mask) in loader:
        for i in range(context_x.shape[0]):
            nc = int(context_mask[i].sum().item())
            nt = int(target_mask[i].sum().item())
            if nc < 1 or nt < 1:
                continue

            x_c = cache_x_to_gp_x(context_x[i, :nc].numpy(), params)
            y_c = context_y[i, :nc].numpy().astype(np.float64)
            x_t = cache_x_to_gp_x(target_x[i, :nt].numpy(), params)
            y_t = target_y[i, :nt].numpy().astype(np.float64)

            reg.fit(x_c, y_c)
            mean, var_total, _ = reg.predict(x_t)

            # Predictive NLL (exact; no latent marginalisation)
            log_p = -0.5 * (((y_t - mean) ** 2) / var_total
                            + np.log(2.0 * math.pi * var_total)).sum(axis=-1)
            nll_sum += -log_p.sum()
            nll_count += nt

            # Decode u/v to physical wind
            u_pred = mean[:, 0] * params.u_std + params.u_mean
            v_pred = mean[:, 1] * params.v_std + params.v_mean
            pred_speed = np.sqrt(u_pred**2 + v_pred**2)
            pred_dir = np.degrees(np.arctan2(-u_pred, -v_pred)) % 360

            u_true = y_t[:, 0] * params.u_std + params.u_mean
            v_true = y_t[:, 1] * params.v_std + params.v_mean
            true_speed = np.sqrt(u_true**2 + v_true**2)
            true_dir = np.degrees(np.arctan2(-u_true, -v_true)) % 360

            speed_err = pred_speed - true_speed
            dir_diff = np.abs(pred_dir - true_dir) % 360
            dir_err = np.minimum(dir_diff, 360 - dir_diff)

            abs_speed_err_sum += np.abs(speed_err).sum()
            sq_speed_err_sum += (speed_err ** 2).sum()
            dir_err_sum += dir_err.sum()

            # Physical-space uncertainty via the shared delta method applied
            # to the u/v vector uncertainty (same path as scripts/test.py).
            su = np.sqrt(var_total[:, 0]) * params.u_std
            sv = np.sqrt(var_total[:, 1]) * params.v_std
            std_dir, std_speed = uv_to_speed_dir_std(u_pred, v_pred, su, sv)

            cov68_dir += (dir_err <= std_dir).sum()
            cov95_dir += (dir_err <= 2 * std_dir).sum()
            cov68_spd += (np.abs(speed_err) <= std_speed).sum()
            cov95_spd += (np.abs(speed_err) <= 2 * std_speed).sum()
            n_snapshots += 1

    metrics = {
        "predictive_nll": float(nll_sum / nll_count),
        "wind_speed_mae_kt": float(abs_speed_err_sum / nll_count),
        "wind_speed_rmse_kt": float(math.sqrt(sq_speed_err_sum / nll_count)),
        "wind_dir_mae_deg": float(dir_err_sum / nll_count),
        "coverage_68_dir": float(cov68_dir / nll_count),
        "coverage_95_dir": float(cov95_dir / nll_count),
        "coverage_68_speed": float(cov68_spd / nll_count),
        "coverage_95_speed": float(cov95_spd / nll_count),
        "coverage_68": float((cov68_dir + cov68_spd) / (2 * nll_count)),
        "coverage_95": float((cov95_dir + cov95_spd) / (2 * nll_count)),
        "n_snapshots": int(n_snapshots),
        "n_held_out_points": int(nll_count),
        "split": split,
        "params": params_path,
        "model": "gp",
    }

    if verbose:
        print(f"[{split}] predictive_nll={metrics['predictive_nll']:.4f}  "
              f"speed_mae={metrics['wind_speed_mae_kt']:.2f}kt  "
              f"speed_rmse={metrics['wind_speed_rmse_kt']:.2f}kt  "
              f"dir_mae={metrics['wind_dir_mae_deg']:.1f} deg  "
              f"coverage_68={metrics['coverage_68']:.2%}  "
              f"coverage_95={metrics['coverage_95']:.2%}  "
              f"(n={metrics['n_held_out_points']} pts / "
              f"{metrics['n_snapshots']} snapshots)")
        print("  NOTE: ANP test.py reports a K-sample marginal NLL; "
              "the GP NLL is exact. Compare MAE/RMSE/coverage primarily.")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Evaluate the GP baseline on held-out data.")
    p.add_argument("--params", required=True,
                   help="Path to gp_params.json from train_gp.py")
    p.add_argument("--cache", required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--context_frac", type=float, default=0.5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--eval_seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    result = evaluate(
        params_path=args.params, cache_dir=args.cache, split=args.split,
        context_frac=args.context_frac, batch_size=args.batch,
        split_seed=args.split_seed, eval_seed=args.eval_seed)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Metrics written to {args.out}")
