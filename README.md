# ADS-B Wind Map

An Attentive Neural Process (ANP) that predicts wind vectors from ADS-B aircraft observations. Given a set of aircraft-derived wind measurements (direction and speed) at known positions, the model learns to interpolate wind fields across a region and quantify its own uncertainty.

![Predicted wind field](outputs/imgs/wind_field.png)

## How it works

A Raspberry Pi running [dump1090](https://github.com/antirez/dump1090) collects ADS-B position reports from nearby aircraft. Each aircraft's reported ground track and airspeed, compared against its expected behaviour, yields a single wind observation (direction, speed) at a lat/lon/altitude. Over time, thousands of these observations accumulate into "snapshots" of the local wind field.

## Architecture

The model is an [Attentive Neural Process](https://arxiv.org/abs/1901.05761) — a conditional generative model that maps a set of observed context points to a predictive distribution over query locations.

**Encoders.** Two parallel encoding paths process the context set `{(x_i, y_i)}`:

- **Latent encoder** — aggregates context into a global stochastic variable `z ~ N(mu, sigma)` via self-attention over context pairs followed by mean-pooling. This captures overall wind regime uncertainty (epistemic).
- **Deterministic encoder** — produces a per-target representation `r*` via self-attention followed by multi-head cross-attention from target queries to context keys. This captures local spatial structure conditioned on nearby observations.

**Decoder.** Concatenates `r*`, `z`, and the target position `x*`, then passes them through an MLP to output per-point mean and standard deviation of the predictive wind distribution.

Both encoders use pre-norm Transformer blocks (scaled dot-product self-attention + position-wise FFN) rather than the simple MLPs in the original ANP paper.

**Training.** Optimised via the negative ELBO: reconstruction log-likelihood under the predictive distribution minus KL divergence between the posterior (from target observations) and prior (from context only). Direction is encoded as `(sin, cos)` to respect circularity.

## Requirements

- Python 3.10+
- PyTorch 2.x
- NumPy
- tqdm
- Matplotlib
- Pillow
- scipy (for animated visualisations)
- contextily (for basemap tiles)
- scikit-optimize (for hyperparameter search)

## Usage

### Convert database to training cache

```bash
python scripts/convert_db.py --db path/to/adsb.db --out data/npy_cache
```

### Train

```bash
python scripts/train.py --cache data/npy_cache --epochs 200 --hidden 256
```

Checkpoints are saved to `./checkpoint/best_model.pth.tar` whenever validation loss improves. Supports early stopping (`--patience 50`), LR warmup (`--warmup-steps 4000` or `--warmup-frac 0.05`), and pretrained initialisation (`--init_checkpoint path/to/ckpt`).

### Evaluate on test set

```bash
python scripts/test.py --checkpoint checkpoint/best_model.pth.tar --cache data/npy_cache
```

Reports predictive NLL, speed MAE/RMSE, direction MAE, and 68/95% coverage. Use `--split val` to evaluate on the validation set, `--context-frac 0.5` to control context/target ratio.

### Hyperparameter search

```bash
python scripts/hp_search.py --cache data/npy_cache --n-calls 60 --search-epochs 200
```

Bayesian optimisation over learning rate, hidden size, batch size, layers, and FFN expansion. Results saved to `hp_optim_results.pkl`.

### Plot hyperparameter search results

```bash
python scripts/plot_hp_results.py
```

Generates convergence, objective, and evaluations plots from `hp_optim_results.pkl`.

### Visualise wind field

```bash
python scripts/visualise_wind.py --checkpoint checkpoint/best_model.pth.tar --cache data/npy_cache --alt_ft 35000
```

Plots predicted wind as a quiver map with uncertainty heatmap. Use `--context_json path/to/context.json` instead of `--cache` to predict from custom observations.

### Animate wind flow

```bash
python scripts/animate_wind.py --checkpoint checkpoint/best_model.pth.tar --cache data/npy_cache --alt_ft 35000 --output outputs/imgs/wind_flow.gif
```

Particle advection animation. Controls: `--n_particles 1000`, `--n_frames 500`, `--fps 30`, `--dt_seconds 45`, `--trail_len 8`.

### Visualise uncertainty components

```bash
python scripts/visualise_uncertainty.py --checkpoint checkpoint/best_model.pth.tar --cache data/npy_cache --alt_ft 35000
```

Side-by-side aleatoric vs epistemic maps. Add `--average` to average over all snapshots in the split.

### Analyse wind variability

```bash
python scripts/analyse_variability.py --cache data/npy_cache --alt_ft 36000
```

Raw-data diagnostics: leave-one-out wind variability by latitude and lat/altitude density.
