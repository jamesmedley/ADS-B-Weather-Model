"""
hp_search.py — Bayesian hyperparameter
search using scikit-optimize.

Usage:
    python hp_search.py --cache data/npy_cache
"""

import argparse
import itertools
import logging
import math

from skopt import gp_minimize, dump
from skopt.space import Real, Integer, Categorical

from wind_map.train import train

# Search space
space = [
    Real(1e-4, 1e-2, "log-uniform"),        # learning rate
    Categorical([16, 32, 64, 128, 256]),    # hidden
    Integer(256, 1024),                     # batch size
    Integer(1, 4),                          # latent layers
    Integer(1, 4),                          # deterministic layers
    Categorical([16, 32, 64, 128, 256]),    # latent dim
]


def make_objective(cache_dir, search_epochs,
                   num_workers, split_seed,
                   warmup_frac, dropout, patience,
                   use_amp):
    run_counter = itertools.count(1)

    def objective(params):
        n = next(run_counter)
        print(f"Search Run {n}")
        (lr, hidden, batch, latent_layers,
         deterministic_layers, num_latents) = params
        lr = float(lr)
        hidden = int(hidden)
        batch = int(batch)
        latent_layers = int(latent_layers)
        deterministic_layers = int(deterministic_layers)
        num_latents = int(num_latents)

        try:
            train_result = train(
                cache_dir=cache_dir,
                num_hidden=hidden,
                num_latents=num_latents,
                epochs=search_epochs,
                batch_size=batch,
                num_workers=num_workers,
                latent_layers=latent_layers,
                deterministic_layers=deterministic_layers,
                dropout=dropout,
                lr=lr,
                warmup_frac=warmup_frac,
                save_checkpoint=False,
                run_test_eval=False,
                split_seed=split_seed,
                verbose=True,
                patience=patience,
                use_amp=use_amp,
            )
        except Exception:
            logging.exception(
                "Search trial %d crashed during training.",
                n)
            print(
                f"  Trial {n} crashed; returning penalty.")
            return 1e9

        composite = train_result['best_composite']
        speed = train_result['best_speed_mae']
        direction = train_result['best_dir_mae']

        if not (math.isfinite(composite)
                and math.isfinite(speed)
                and math.isfinite(direction)):
            print(
                f"  Trial {n} produced non-finite "
                f"metrics; returning penalty.")
            return 1e9

        print(f"  composite={composite:.2f}  "
              f"speed_mae={speed:.2f}kt  dir_mae={direction:.1f} deg  "
              f"(lr={lr:.2e} hidden={hidden} batch={batch} "
              f"latent_layers={latent_layers} "
              f"det_layers={deterministic_layers} "
              f"num_latents={num_latents} dropout={dropout:.3f})")
        return composite

    return objective


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description=(
            'Bayesian hyperparameter search'
            ' for the Wind ANP'
            ' (scikit-optimize).'))
    p.add_argument('--cache', required=True)
    p.add_argument('--n_calls', type=int, default=400)
    p.add_argument('--search_epochs', type=int, default=250)
    p.add_argument('--warmup_frac', type=float, default=0.05)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--split_seed', type=int, default=42)
    p.add_argument('--random_state', type=int, default=42)
    p.add_argument('--out', default='hp_optim_results.pkl')
    p.add_argument('--no_amp', action='store_true',
                   help='Disable automatic mixed precision')

    args = p.parse_args()

    objective = make_objective(
        args.cache, args.search_epochs, args.workers,
        args.split_seed, args.warmup_frac, args.dropout,
        args.patience,
        use_amp=not args.no_amp)

    result = gp_minimize(
        objective, space,
        n_calls=args.n_calls,
        random_state=args.random_state)
    dump(result, args.out, store_objective=False)

    print("\nBest Hyperparameters:")
    names = [
        'Learning Rate', 'Hidden', 'Batch', 'Latent Layers',
        'Det Layers', 'Latent Dim',
    ]
    for name, value in zip(names, result.x):
        print(f"{name}: {value}")
    print(f"Best composite (speed_mae + dir_mae/4): {result.fun:.2f}")
    print(f"\nResults saved to {args.out} - run plot_hp_results.py for plots.")
