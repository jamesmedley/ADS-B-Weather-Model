"""
plot_training.py — Diagnostic plots from the training CSV log.

Usage:
    python scripts/plot_training.py path/to/training_log.csv [--out plot.png]
"""

import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt


def read_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows, reader.fieldnames


def to_array(rows, key):
    return np.array([float(r[key]) for r in rows])


def plot_training(csv_path, out_path=None, dpi=150):
    rows, fields = read_csv(csv_path)

    epoch = to_array(rows, 'epoch')
    train_loss = to_array(rows, 'train_loss')
    val_loss = to_array(rows, 'val_loss')
    lr = to_array(rows, 'lr')
    val_nll = to_array(rows, 'val_nll')
    val_kl = to_array(rows, 'val_kl')
    speed_rmse = to_array(rows, 'speed_rmse_kt')
    speed_mae = to_array(rows, 'speed_mae_kt')
    dir_mae = to_array(rows, 'dir_mae_deg')
    cov_68 = to_array(rows, 'cov_68')
    cov_95 = to_array(rows, 'cov_95')
    active_pct = to_array(rows, 'active_pct')
    sigma_speed = to_array(rows, 'sigma_speed_kt')
    sigma_dir = to_array(rows, 'sigma_dir_deg')

    best_mask = np.array([int(r.get('best_so_far', '0')) for r in rows], dtype=bool)
    best_epoch = epoch[best_mask][0] if best_mask.any() else epoch[-1]
    best_idx = np.searchsorted(epoch, best_epoch)

    fig, axes = plt.subplots(4, 2, figsize=(16, 14))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    def _mark_best(ax, ydata=None):
        x = best_epoch
        if ydata is not None:
            y = ydata[best_idx]
            ax.plot(x, y, marker='*', color='red', ms=10, zorder=5)
        ax.axvline(x=x, color='red', linestyle=':', alpha=0.4, linewidth=1)

    # (0,0) — training & validation loss
    ax = axes[0, 0]
    ax.plot(epoch, train_loss, label='train_loss', color='#1f77b4')
    ax.plot(epoch, val_loss, label='val_loss', color='#ff7f0e')
    _mark_best(ax)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.set_title('Training & Validation Loss')

    # (0,1) — learning rate
    ax = axes[0, 1]
    ax.plot(epoch, lr, color='#2ca02c')
    _mark_best(ax)
    ax.set_ylabel('Learning Rate')
    ax.set_xlabel('Epoch')
    ax.set_title('Learning Rate Schedule')

    # (1,0) — speed RMSE & MAE
    ax = axes[1, 0]
    ax.plot(epoch, speed_rmse, label='RMSE', color='#1f77b4')
    ax.plot(epoch, speed_mae, label='MAE', color='#ff7f0e')
    _mark_best(ax)
    ax.set_ylabel('knots')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.set_title('Wind Speed Error')

    # (1,1) — direction MAE
    ax = axes[1, 1]
    ax.plot(epoch, dir_mae, color='#d62728')
    _mark_best(ax)
    ax.set_ylabel('degrees')
    ax.set_xlabel('Epoch')
    ax.set_title('Wind Direction MAE')

    # (2,0) — NLL & KL
    ax = axes[2, 0]
    ax.plot(epoch, val_nll, label='val_nll', color='#1f77b4')
    ax.plot(epoch, val_kl, label='val_kl', color='#ff7f0e')
    _mark_best(ax)
    ax.set_ylabel('nats')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.set_title('Validation NLL & KL')

    # (2,1) — active latent dimensions
    ax = axes[2, 1]
    ax.plot(epoch, active_pct, color='#9467bd')
    _mark_best(ax)
    ax.set_ylabel('%')
    ax.set_xlabel('Epoch')
    ax.set_title('Active Latent Dimensions')

    # (3,0) — coverage
    ax = axes[3, 0]
    ax.plot(epoch, cov_68, label='68% coverage', color='#1f77b4')
    ax.plot(epoch, cov_95, label='95% coverage', color='#ff7f0e')
    ax.axhline(y=0.68, color='#1f77b4', linestyle='--', alpha=0.3)
    ax.axhline(y=0.95, color='#ff7f0e', linestyle='--', alpha=0.3)
    _mark_best(ax)
    ax.set_ylabel('Coverage')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.set_title('Prediction Interval Coverage')
    ax.set_ylim(0, 1)

    # (3,1) — mean predicted sigma
    ax = axes[3, 1]
    ax.plot(epoch, sigma_speed, label='σ speed (kt)', color='#1f77b4')
    ax.plot(epoch, sigma_dir, label='σ dir (deg)', color='#ff7f0e')
    _mark_best(ax)
    ax.set_ylabel('Std')
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.set_title('Mean Predicted Uncertainty')

    fig.suptitle(
        f'Training Diagnostics — best epoch {int(best_epoch)}',
        fontsize=14, y=0.98)

    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {out_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot training diagnostics from CSV log')
    parser.add_argument('csv', help='Path to training_log.csv')
    parser.add_argument('--out', default=None,
                        help='Output image path (default: show interactive)')
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args()
    plot_training(args.csv, args.out, dpi=args.dpi)
