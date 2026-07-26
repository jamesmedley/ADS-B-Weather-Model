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
        '--batch', type=int, default=512)
    parser.add_argument(
        '--workers', type=int, default=4)
    parser.add_argument(
        '--layers', type=int, default=4)
    parser.add_argument(
        '--decoder_layers', type=int, default=3,
        help='Number of hidden layers in the decoder MLP')
    parser.add_argument(
        '--dropout', type=float, default=0.2)
    parser.add_argument(
        '--lr', type=float, default=1.65e-03)
    parser.add_argument(
        '--warmup_steps', type=int, default=4000,
        help='LR warmup steps '
             '(ignored if --warmup_frac is set)')
    parser.add_argument(
        '--warmup_frac', type=float, default=None,
        help='Warmup as fraction of total run steps'
             ' (overrides --warmup_steps)')
    parser.add_argument(
        '--kl_warmup_steps', type=int, default=2000,
        help='KL weight linear warmup steps')
    parser.add_argument(
        '--free_bits', type=float, default=0.01,
        help='Free bits per latent dimension (nats)')
    parser.add_argument(
        '--checkpoint_dir',
        default='./checkpoint')
    parser.add_argument(
        '--patience', type=int, default=0,
        help='Early stop patience (default: 0, no early stopping)')
    parser.add_argument(
        '--init_checkpoint', default=None,
        help='Optional pretrained checkpoint '
             'to initialise from')
    parser.add_argument(
        '--no_amp', action='store_true',
        help='Disable automatic mixed precision')
    parser.add_argument(
        '--weight_decay', type=float, default=1e-5,
        help='Weight decay for AdamW')
    parser.add_argument(
        '--no_dist_bias', action='store_true',
        help='Disable distance bias in cross-attention')
    args = parser.parse_args()

    train(
        cache_dir=args.cache,
        num_hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch,
        num_workers=args.workers,
        layers=args.layers,
        num_decoder_layers=args.decoder_layers,
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
        weight_decay=args.weight_decay,
        use_dist_bias=not args.no_dist_bias,
    )
