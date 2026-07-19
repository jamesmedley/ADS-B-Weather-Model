# ML Review: ADS-B Wind Map (Attentive Neural Process)

> **Date:** 2026-07-18
> **Reviewer:** opencode (big-pickle)

---

## Overall Impression

This is a genuinely creative and well-engineered project. Using aircraft as flying anemometers is a novel sensing paradigm, and the ANP is a strong architectural choice — it naturally handles variable-size context sets, provides calibrated uncertainty, and interpolates to arbitrary query locations. The code quality is high: modular, clean, well-documented, with proper day-grouped splits, memory-mapped data loading, and principled uncertainty decomposition.

That said, there are several concrete issues, inconsistencies, and high-impact experiments worth discussing.

---

## 1. Bugs & Inconsistencies

### b) Wind speed exceeds normalization constant

`MAX_WIND_KT = 200` but the data contains observations up to 248 kt (`wind_speed_kt` max). This means normalized speed values exceed 1.0 (up to 1.24). The model can learn this, but it's inconsistent with the documented design intent of `[0, 1]` range. Either increase `MAX_WIND_KT` to 250 or clip in `convert_db.py`.

### e) Coverage metric computed in raw normalized space

`test.py:230-232` computes 68/95% coverage by averaging over 3 raw dimensions (sin, cos, speed_norm). This is problematic:

- sin/cos are correlated components of a single angle; treating them as independent dims is misleading
- A point can be within 1σ on sin but outside 1σ on cos
- Speed is normalized to [0, ~1.24], so 1σ in normalized space has different physical meaning than 1σ in direction

Coverage should be computed separately for wind direction (in degrees) and wind speed (in knots), or use a proper multivariate coverage metric.

---

## 2. Model Architecture Observations

### a) Latent encoder sigma is tightly bounded [0.1, 1.0]

`module.py:258`: `sigma = 0.1 + 0.9 * sigmoid(log_sigma)` means the latent z can never have variance below 0.1² or above 1.0². This is quite restrictive. When you have many context observations covering a region, the posterior should collapse to very low uncertainty (sigma → 0). The floor of 0.1 is fine, but the ceiling of 1.0 may prevent the model from expressing high uncertainty when context is very sparse. Consider widening to `[0.1, 2.0]` or `[0.05, 2.0]`.

### b) No positional encoding for the input coordinates

The raw (lat, lon, alt) coordinates go directly into the MLP. For spatial interpolation tasks, **Fourier feature mapping** (NeRF-style) can dramatically improve the model's ability to learn high-frequency spatial patterns. This is a well-known technique that's particularly relevant here since wind fields have spatial structure at multiple scales (synoptic-scale gradients, local orographic effects, etc.).

### c) Decoder has no dropout

The `Decoder` class in `module.py:263-277` has no dropout, unlike the encoder's `TransformerBlock`. The best HP search found `dropout=0.19`, but this only applies to the encoder. Adding dropout to the decoder MLP could help regularize, especially given the small dataset.

### d) Single latent variable z may be too compressed

The latent encoder outputs a single `z` vector of size `num_hidden=128` to represent the entire wind regime. For snapshots with 2-16 aircraft spread over 70km and 1000-47000ft altitude, this single vector must encode all the global context. Consider:

- Hierarchical latent variables (e.g., per-altitude-band z)
- Multiple latent slots with attention-based routing

---

## 3. Missing Features / Inputs

This is where I think the biggest gains are, beyond just "more data":

### a) No temporal features (HIGH IMPACT)

The model has no concept of time. Wind patterns are strongly correlated with:

- Time of day (diurnal boundary layer cycle, thermal effects)
- Season (prevailing wind direction shifts)
- Recent weather (pressure systems moving through)

Adding temporal features as additional input dimensions could provide significant signal. At minimum:

- `sin(2π * hour/24)`, `cos(2π * hour/24)` — time of day
- `sin(2π * day_of_year/365)`, `cos(2π * day_of_year/365)` — season

This changes `x_dim` from 3 to 7, but the architecture handles variable `x_dim` already.

### b) Temperature data is available but unused

The SQLite DB contains `oat_c` (outside air temperature) and `tat_c` (total air temperature) for every observation. Temperature is physically related to wind (thermal wind, pressure gradients) and could improve predictions. Even just using OAT as an additional output or auxiliary feature would be valuable.

### c) No weather context (pressure, fronts)

The Dockerfile installs `xarray`, `cfgrib`, `eccodes` and `note.txt` mentions a planned `pretrain.py --grib` script. Pre-training on ERA5 reanalysis data or GRIB forecast data would be transformative — it would give the model knowledge of synoptic-scale wind patterns that it could then fine-tune with sparse ADS-B observations. This is probably the single highest-impact improvement you could make.

### d) Vertical rate / rate of climb not used

`baro_rate_fpm` and `geom_rate_fpm` are in the DB. An aircraft's vertical rate affects the wind estimation process and could serve as a useful auxiliary feature or quality signal.

---

## 4. Data Limitations & How to Address Them

You're right that data is a significant limitation, but I'd frame it more precisely as three distinct issues:

### a) Volume: 38 days, 60K observations

This is small for a deep learning model. The ANP architecture helps (meta-learning, attention), but more data would still help. Mitigations:

- **Collect more data** (trivially, just keep the receiver running)
- **Pre-train on ERA5/GRIB data** (synthetic "context → wind field" pairs from reanalysis)
- **Synthetic data augmentation**: Generate fake wind fields (e.g., smooth random Gaussian wind fields), sample synthetic "observations" from them, and train the ANP to reconstruct the field. This teaches the model spatial interpolation before it sees real data.

### b) Spatial bias: clustered along flight paths

Aircraft don't sample uniformly — they follow airways and approach/departure corridors. The model must interpolate to areas with no flight coverage. This is actually a strength of the ANP (it gives high epistemic uncertainty there), but more uniform coverage would improve training. Consider:

- Including data from nearby receivers (network of dump1090 stations)
- Using MLAT or multilateration data for lower-altitude coverage

### c) Temporal sparsity: no seasonal variation

38 days doesn't capture seasonal wind patterns. Wind climatology in southern England is very different in winter vs summer. This is arguably the most important data limitation for real-world deployment.

---

## 5. Training Improvements

### d) More systematic data augmentation

The current augmentation (random wind direction rotation) is clever and correct, but limited. Consider:

- **Spatial jitter**: add small Gaussian noise to lat/lon (simulates GPS uncertainty and slightly different observation positions)
- **Time masking**: randomly drop the temporal position of snapshots (forces model to rely on spatial context)
- **Observation dropout**: during training, randomly zero-out the wind values for some context points (simulates noisy/missing data)

### e) KL annealing (beta-VAE style)

The ELBO loss includes KL divergence without any weighting. For VAE-style models, starting with KL weight = 0 and annealing to 1 over the first N epochs ("KL warmup") often stabilizes training and prevents posterior collapse.

---

## 6. Evaluation Improvements

### a) CRPS (Continuous Ranked Probability Score)

NLL is a proper scoring rule but can be dominated by tail behavior. CRPS is more robust and directly measures the quality of the probabilistic forecast. `properscoring` or `crps` packages in Python make this easy.

### b) Reliability / calibration diagrams

Rather than just reporting "coverage 68% = X%", plot calibration curves showing the fraction of observations falling within each percentile interval. This reveals whether the model is over/under-confident at different uncertainty levels.

### c) Stratified evaluation

Break down metrics by:

- Altitude band (surface layer < 3000ft, cruise > 25000ft, in-between)
- Number of context observations (sparse vs dense)
- Time of day
- Distance from nearest observation

This reveals where the model fails and where more data would help most.

### d) Spatial correlation evaluation

Wind fields are spatially correlated. A good wind map should preserve this structure. Compute the variogram of predicted vs actual wind fields to check spatial consistency.

---

## 7. High-Value Experiments to Try

In approximate priority order:

1. **Add temporal features** (hour-of-day, day-of-year as sin/cos). This is low-effort, high-impact.

2. **Pre-train on ERA5 reanalysis data.** Download ERA5 hourly wind data for your region, create synthetic "sparse observation → dense field" training pairs. This addresses both volume and temporal coverage simultaneously.

3. **Fourier feature mapping for spatial inputs.** Map (lat, lon, alt) through random Fourier features before the MLP. This is a 1-line change that often dramatically improves spatial models.

4. **Synthetic data pre-training.** Generate smooth random wind fields, sample context points, train the ANP to reconstruct. Teaches spatial interpolation without needing real data.

5. **Increase z dimensionality** or use **multiple latent variables** to better capture global wind regime.

6. **Add wind speed to the data augmentation** (multiply by random factor in [0.8, 1.2] to simulate different weather regimes).

7. **Ablation study: deterministic-only vs latent-only vs full ANP.** This tells you whether the latent variable or the deterministic cross-attention is doing the heavy lifting.

8. **Evaluate the aleatoric vs epistemic decomposition.** Does epistemic uncertainty actually decrease with more context observations? Does it increase in areas with no flight coverage? Visual inspection of your uncertainty maps would be informative.

---

## 8. Summary Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| Problem formulation | Excellent | Novel, creative, well-motivated |
| Architecture choice | Very good | ANP is well-suited; minor tweaks could help |
| Code quality | Excellent | Clean, modular, well-documented |
| Data pipeline | Very good | Memory-mapped, proper splits, augmentation |
| Training setup | Good | Solid foundations; AMP, EMA, better schedules would help |
| Evaluation | Adequate | Coverage metric is flawed; needs CRPS, calibration plots |
| Feature engineering | Needs work | No temporal features; temperature unused; no Fourier features |
| Data volume | Primary bottleneck | 38 days is too few for seasonal variation |
| Missing: pre-training | High priority | GRIB/ERA5 pre-training would be transformative |

**Your instinct is correct: data is the primary limitation.** But specifically, it's *temporal* coverage (no seasons) and *lack of external weather context* (no ERA5/GRIB pre-training) that hurt most. Simply adding temporal features and pre-training on reanalysis data could yield larger improvements than doubling the ADS-B collection time.
