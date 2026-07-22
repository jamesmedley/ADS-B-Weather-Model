"""
train.py — CLI wrapper for training the Wind ANP.

Usage:
    python scripts/train.py --cache data/npy_cache --epochs 200 --hidden 128

For the train() function directly, see wind_map.train.
"""

import argparse

from wind_map.train import train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Wind ANP')
    parser.add_argument(
        '--cache', required=True,
        help='Path to .npy cache directory')
    parser.add_argument(
        '--epochs', type=int, default=1000)
    parser.add_argument(
        '--hidden', type=int, default=128)
    parser.add_argument(
        '--batch', type=int, default=256)
    parser.add_argument(
        '--workers', type=int, default=4)
    parser.add_argument(
        '--layers', type=int, default=4)
    parser.add_argument(
        '--dropout', type=float, default=0.2)
    parser.add_argument(
        '--lr', type=float, default=1.65e-03)
    parser.add_argument(
        '--warmup-steps', type=int, default=4000,
        help='LR warmup steps '
             '(ignored if --warmup-frac is set)')
    parser.add_argument(
        '--warmup-frac', type=float, default=None,
        help='Warmup as fraction of total run steps'
             ' (overrides --warmup-steps)')
    parser.add_argument(
        '--kl-warmup-steps', type=int, default=2000,
        help='KL weight linear warmup steps')
    parser.add_argument(
        '--free-bits', type=float, default=0.01,
        help='Free bits per latent dimension (nats)')
    parser.add_argument(
        '--checkpoint-dir',
        default='./checkpoint')
    parser.add_argument(
        '--patience', type=int, default=0,
        help='Early stop patience (default: 0, no early stopping)')
    parser.add_argument(
        '--init_checkpoint', default=None,
        help='Optional pretrained checkpoint '
             'to initialise from')
    parser.add_argument(
        '--no-amp', action='store_true',
        help='Disable automatic mixed precision')

    args = parser.parse_args()

    train(
        cache_dir=args.cache,
        num_hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch,
        num_workers=args.workers,
        num_layers=args.layers,
        dropout=args.dropout,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        warmup_frac=args.warmup_frac,
        kl_warmup_steps=args.kl_warmup_steps,
        free_bits=args.free_bits,
        checkpoint_dir=args.checkpoint_dir,
        init_checkpoint=args.init_checkpoint,
        patience=args.patience,
        use_amp=not args.no_amp,
    )
