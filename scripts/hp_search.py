"""
hp_search.py — Bayesian hyperparameter
search using scikit-optimize.

Usage:
    python hp_search.py --cache data/npy_cache
"""

import argparse

from skopt import gp_minimize, dump
from skopt.space import Real, Integer, Categorical

from wind_map.train import train

# Search space — hidden must be multiples of 8 (num_heads)
space = [
    Real(1e-4, 1e-2, "log-uniform"),   # learning rate
    Categorical([64, 128, 256]),        # hidden
    Integer(128, 384),                  # batch size
    Integer(2, 4),                      # layers
    Integer(1, 4),                      # ffn expansion
]

n = 0


def make_objective(cache_dir, search_epochs,
                   num_workers, split_seed,
                   warmup_frac, dropout, patience):
    def objective(params):
        global n
        n += 1
        print(f"Search Run {n}")
        lr, hidden, batch, layers, ffn_expansion = params
        lr = float(lr)
        hidden = int(hidden)
        batch = int(batch)
        layers = int(layers)
        ffn_expansion = int(ffn_expansion)

        train_result = train(
            cache_dir=cache_dir,
            num_hidden=hidden,
            epochs=search_epochs,
            batch_size=batch,
            num_workers=num_workers,
            num_layers=layers,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
            lr=lr,
            warmup_frac=warmup_frac,
            save_checkpoint=False,
            run_test_eval=False,
            split_seed=split_seed,
            verbose=True,
            patience=patience,
        )

        val_loss = train_result['best_val_loss']
        print(f"  val_loss={val_loss:.4f}  "
              f"(lr={lr:.2e} hidden={hidden} batch={batch} layers={layers} "
              f"dropout={dropout:.3f} ffn_expansion={ffn_expansion})")
        return val_loss

    return objective


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description=(
            'Bayesian hyperparameter search'
            ' for the Wind ANP'
            ' (scikit-optimize).'))
    p.add_argument('--cache', required=True)
    p.add_argument('--n-calls', type=int, default=200)
    p.add_argument('--search-epochs', type=int, default=2000)
    p.add_argument('--warmup-frac', type=float, default=0.05)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--random-state', type=int, default=42)
    p.add_argument('--out', default='hp_optim_results.pkl')

    args = p.parse_args()

    objective = make_objective(args.cache, args.search_epochs, args.workers,
                               args.split_seed, args.warmup_frac, args.dropout,
                               args.patience)

    result = gp_minimize(
        objective, space,
        n_calls=args.n_calls,
        random_state=args.random_state)
    dump(result, args.out, store_objective=False)

    print("\nBest Hyperparameters:")
    names = [
        'Learning Rate', 'Hidden', 'Batch',
        'Layers', 'Dropout', 'FFN Expansion']
    for name, value in zip(names, result.x):
        print(f"{name}: {value}")
    print(f"Best val_loss: {result.fun:.4f}")
    print(f"\nResults saved to {args.out} — run plot_hp_results.py for plots.")
