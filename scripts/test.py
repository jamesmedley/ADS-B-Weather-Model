"""
test.py — Held-out evaluation metrics for the Wind ANP.

Reports:
  1. Predictive NLL via K-sample marginal log-likelihood (primary metric)
  2. Physical error metrics: speed MAE/RMSE, direction MAE
  3. Calibration: 68/95% coverage

Metrics are computed on target points NOT in the context set (true held-out).

Usage:
    python test.py --checkpoint checkpoint/best_model.pth.tar \
                    --cache data/npy_cache
"""

import json
import math
import argparse
import numpy as np
import torch as t
from torch.utils.data import DataLoader

from wind_map.network import LatentModel
from wind_map.preprocess import (
    WindSnapshotDataset, day_grouped_split, MAX_WIND_KT
)


# --- Context / held-out-target collate ---

def eval_collate_fn(batch, context_frac=0.5):
    """
    Like preprocess.collate_fn, but target = complement of context (true
    held-out points). No augmentation. Zero-pads with bool masks.
    """
    context_xs, context_ys, target_xs, target_ys = [], [], [], []
    ctx_lens, tgt_lens = [], []

    for (x, y) in batch:
        n = x.size(0)
        if n < 2:
            continue

        n_ctx = max(1, min(int(n * context_frac), n - 1))
        perm = t.randperm(n)
        ctx_idx = perm[:n_ctx]
        held_idx = perm[n_ctx:]
        if held_idx.numel() == 0:
            continue

        context_xs.append(x[ctx_idx])
        context_ys.append(y[ctx_idx])
        target_xs.append(x[held_idx])
        target_ys.append(y[held_idx])

        ctx_lens.append(n_ctx)
        tgt_lens.append(held_idx.numel())

    if not context_xs:
        raise RuntimeError(
            "Empty batch — every item had too few"
            " observations to split."
        )

    B = len(context_xs)
    max_ctx = max(ctx_lens)
    max_tgt = max(tgt_lens)
    x_dim = context_xs[0].size(-1)
    y_dim = context_ys[0].size(-1)

    context_x = context_xs[0].new_zeros(B, max_ctx, x_dim)
    context_y = context_ys[0].new_zeros(B, max_ctx, y_dim)
    target_x = target_xs[0].new_zeros(B, max_tgt, x_dim)
    target_y = target_ys[0].new_zeros(B, max_tgt, y_dim)
    context_mask = t.zeros(B, max_ctx, dtype=t.bool)
    target_mask = t.zeros(B, max_tgt, dtype=t.bool)

    for i in range(B):
        nc, nt = ctx_lens[i], tgt_lens[i]
        context_x[i, :nc] = context_xs[i]
        context_y[i, :nc] = context_ys[i]
        target_x[i, :nt] = target_xs[i]
        target_y[i, :nt] = target_ys[i]
        context_mask[i, :nc] = True
        target_mask[i, :nt] = True

    return context_x, context_y, target_x, target_y, context_mask, target_mask


def _circular_abs_diff_deg(a, b):
    """Elementwise circular |a - b| in degrees, result in [0, 180]."""
    d = np.abs(a - b) % 360
    return np.minimum(d, 360 - d)


def load_model(checkpoint_path, device,
               num_hidden=None, num_layers=None,
               ffn_expansion=None, dropout=None):
    """Reconstruct LatentModel from checkpoint.
    Recent checkpoints carry hparams.
    """
    ckpt = t.load(
        checkpoint_path, map_location=device,
        weights_only=False)
    hp = ckpt.get('hparams', {})

    num_hidden = num_hidden if num_hidden is not None else hp.get('num_hidden')
    num_layers = (
        num_layers if num_layers is not None
        else hp.get('num_layers', 2)
    )
    ffn_expansion = (
        ffn_expansion if ffn_expansion is not None
        else hp.get('ffn_expansion', 2)
    )
    dropout = dropout if dropout is not None else hp.get('dropout', 0.0)

    if num_hidden is None:
        raise ValueError(
            "No 'hparams' in checkpoint"
            " — pass --hidden explicitly."
        )

    model = LatentModel(
        num_hidden, x_dim=3, y_dim=3,
        num_layers=num_layers,
        ffn_expansion=ffn_expansion, dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt


@t.no_grad()
def evaluate(checkpoint_path, cache_dir, split='test', num_hidden=None,
             num_layers=None, ffn_expansion=None, dropout=None,
             context_frac=0.5, n_samples=30, batch_size=16, split_seed=42,
             eval_seed=0, device=None, verbose=True):
    """
    Returns aggregate metrics over `split`: predictive_nll, speed MAE/RMSE,
    direction MAE, 68/95% coverage.
    """
    if device is None:
        device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    else:
        device = t.device(device)

    model, ckpt = load_model(checkpoint_path, device, num_hidden, num_layers,
                             ffn_expansion, dropout)

    train_ids, val_ids, test_ids = day_grouped_split(
        cache_dir, seed=split_seed
    )
    ids = {'train': train_ids, 'val': val_ids, 'test': test_ids}[split]
    ds = WindSnapshotDataset(cache_dir, snapshot_ids=ids)

    # Fixed seed keeps context/held-out splits
    # reproducible across evaluate() calls
    t.manual_seed(eval_seed)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=0,
        collate_fn=lambda b: eval_collate_fn(b, context_frac=context_frac))

    nll_sum, nll_count = 0.0, 0
    abs_speed_err_sum, sq_speed_err_sum = 0.0, 0.0
    dir_err_sum = 0.0
    cov68_sum, cov95_sum = 0.0, 0.0
    n_snapshots = 0

    for (context_x, context_y, target_x,
         target_y, context_mask, target_mask) in loader:
        context_x = context_x.to(device)
        context_y = context_y.to(device)
        target_x = target_x.to(device)
        target_y_dev = target_y.to(device)
        context_mask = context_mask.to(device)
        mask = target_mask.to(device)

        mu_samples, sigma_samples = [], []
        for _ in range(n_samples):
            mu, sigma, _, _ = model(context_x, context_y, target_x,
                                    target_y=None, context_mask=context_mask)
            mu_samples.append(mu)
            sigma_samples.append(sigma)

        mu_stack = t.stack(mu_samples, dim=0)
        sigma_stack = t.stack(sigma_samples, dim=0)

        # K-sample marginal log-likelihood
        dist = t.distributions.Normal(loc=mu_stack, scale=sigma_stack)
        log_p_k = dist.log_prob(target_y_dev.unsqueeze(0)).sum(dim=-1)
        log_p_marginal = t.logsumexp(log_p_k, dim=0) - math.log(n_samples)

        mask_f = mask.to(log_p_marginal.dtype)
        nll_sum += (-log_p_marginal * mask_f).sum().item()
        nll_count += mask_f.sum().item()

        # Point estimate + law-of-total-variance uncertainty
        mean_mu = mu_stack.mean(dim=0)
        mean_sigma2 = (sigma_stack ** 2).mean(dim=0)
        var_mu = mu_stack.var(dim=0)
        total_std = (mean_sigma2 + var_mu).sqrt()

        mean_mu_np = mean_mu.cpu().numpy()
        total_std_np = total_std.cpu().numpy()
        target_y_np = target_y.numpy()
        mask_np = mask.cpu().numpy().astype(bool)

        pred_dir = np.degrees(
            np.arctan2(
                mean_mu_np[..., 0], mean_mu_np[..., 1]
            )
        ) % 360
        pred_speed = mean_mu_np[..., 2] * MAX_WIND_KT
        true_dir = np.degrees(
            np.arctan2(
                target_y_np[..., 0],
                target_y_np[..., 1]
            )
        ) % 360
        true_speed = target_y_np[..., 2] * MAX_WIND_KT

        speed_err = pred_speed - true_speed
        dir_err = _circular_abs_diff_deg(pred_dir, true_dir)

        abs_speed_err_sum += np.abs(speed_err[mask_np]).sum()
        sq_speed_err_sum += (speed_err[mask_np] ** 2).sum()
        dir_err_sum += dir_err[mask_np].sum()

        # Raw-space coverage (elementwise on
        # sin/cos/speed_norm, averaged over 3 dims)
        abs_err = np.abs(mean_mu_np - target_y_np)
        within_1s = (abs_err <= total_std_np).mean(axis=-1)
        within_2s = (abs_err <= 2 * total_std_np).mean(axis=-1)
        cov68_sum += within_1s[mask_np].sum()
        cov95_sum += within_2s[mask_np].sum()

        n_snapshots += context_x.size(0)

    metrics = {
        'predictive_nll': float(nll_sum / nll_count),
        'wind_speed_mae_kt': float(abs_speed_err_sum / nll_count),
        'wind_speed_rmse_kt': float(math.sqrt(sq_speed_err_sum / nll_count)),
        'wind_dir_mae_deg': float(dir_err_sum / nll_count),
        'coverage_68': float(cov68_sum / nll_count),
        'coverage_95': float(cov95_sum / nll_count),
        'n_snapshots': int(n_snapshots),
        'n_held_out_points': int(nll_count),
        'split': split,
        'checkpoint': checkpoint_path,
        'checkpoint_epoch': ckpt.get('epoch'),
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

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a trained Wind ANP checkpoint on held-out data.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument(
        '--split',
        choices=['train', 'val', 'test'],
        default='test',
    )
    parser.add_argument('--hidden', type=int, default=None)
    parser.add_argument('--layers', type=int, default=None)
    parser.add_argument('--ffn-expansion', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument(
        '--context-frac', type=float, default=0.5,
        help=(
            'Fraction of each snapshot used as'
            ' context (default: 0.5)'
        ),
    )
    parser.add_argument('--samples', type=int, default=30,
                        help='Number of latent z samples (default: 30)')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--eval-seed', type=int, default=0)
    parser.add_argument(
        '--out', default=None,
        help='Optional JSON output path',
    )

    args = parser.parse_args()

    result = evaluate(
        checkpoint_path=args.checkpoint,
        cache_dir=args.cache,
        split=args.split,
        num_hidden=args.hidden,
        num_layers=args.layers,
        ffn_expansion=args.ffn_expansion,
        dropout=args.dropout,
        context_frac=args.context_frac,
        n_samples=args.samples,
        batch_size=args.batch,
        split_seed=args.split_seed,
        eval_seed=args.eval_seed,
    )

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Metrics written to {args.out}")
