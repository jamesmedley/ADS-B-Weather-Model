"""
train.py — CLI wrapper for training the Wind ANP.

Usage:
    python scripts/train.py --cache data/npy_cache --epochs 200 --hidden 128

For the train() function directly, see wind_map.train.
"""

import argparse

from wind_map.logging import run_script
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
        '--num_latents', type=int, default=64,
        help='Number of latent dimensions')
    parser.add_argument(
        '--batch', type=int, default=512)
    parser.add_argument(
        '--workers', type=int, default=4)
    parser.add_argument(
        '--latent_layers', type=int, default=4)
    parser.add_argument(
        '--deterministic_layers', type=int, default=2)
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
    parser.add_argument(
        '--coupled_rotation', action='store_true',
        help='Rigidly rotate positions and wind vectors together by a '
             'random per-snapshot angle (augmentation)')
    parser.add_argument(
        '--deterministic_only', action='store_true',
        help='Disable the latent pathway (num_latents=0)')
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Master RNG seed for reproducible runs')
    parser.add_argument(
        '--log_file', default=None,
        help='Path to the universal log file '
             '(default: run.log in the project root)')
    args = parser.parse_args()

    if args.deterministic_only:
        args.num_latents = 0
        args.free_bits = 0.0
        args.kl_warmup_steps = 0
        args.latent_layers = 0

    run_script(lambda: train(
        cache_dir=args.cache,
        num_hidden=args.hidden,
        num_latents=args.num_latents,
        epochs=args.epochs,
        batch_size=args.batch,
        num_workers=args.workers,
        latent_layers=args.latent_layers,
        deterministic_layers=args.deterministic_layers,
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
        use_coupled_rotation=args.coupled_rotation,
        seed=args.seed,
        log_file=args.log_file,
    ))
