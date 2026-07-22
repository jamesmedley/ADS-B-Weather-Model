"""
diagnose.py - Comprehensive model health diagnostics for Wind ANP.

Focuses on diagnosing why uncertainty heatmaps may appear grainy
by checking representation variance, attention patterns, latent
utilisation, and spatial smoothness of predictions.

Usage:
    python diagnose.py
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.stats import pearsonr

from wind_map.network import LatentModel
from wind_map.preprocess import (
    WindSnapshotDataset, day_grouped_split, collate_fn_val, _worker_init,
    CENTRE_LAT, CENTRE_LON, LAT_RANGE_DEG, LON_RANGE_DEG, WIND_SPEED_STD, WIND_SPEED_MEAN,
)
from torch.utils.data import DataLoader
from wind_map.utils import circular_mean, circular_std, make_grid, wind_to_uv
from wind_map.infer import observations_to_tensors, queries_to_tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

ckpt = torch.load('checkpoint/best_model.pth.tar', map_location=device, weights_only=False)
hp = ckpt['hparams']

model = LatentModel(
    hp['num_hidden'],
    num_layers=hp.get('num_layers', 4),
    free_bits=hp.get('free_bits', 0.001),
)
model.load_state_dict(ckpt['model'])
model.eval().to(device)
print(f"Loaded checkpoint: epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss'):.4f}")
print(f"Free bits: {model.free_bits}, Latent dim: {model.latent_encoder.mean_layer.out_features}")
print(f"Layers: {hp.get('num_layers')}, Hidden: {hp['num_hidden']}, Dropout: {hp.get('dropout', 0.0)}")

# ---- Load a single batch ----
_, val_ids, _ = day_grouped_split('data/npy_cache')
val_ds = WindSnapshotDataset('data/npy_cache', snapshot_ids=val_ids[:8])
loader = DataLoader(val_ds, batch_size=8, shuffle=False,
                    collate_fn=collate_fn_val, worker_init_fn=_worker_init)
context_x, context_y, target_x, target_y, ctx_mask, tgt_mask = next(iter(loader))
context_x = context_x.to(device)
context_y = context_y.to(device)
target_x = target_x.to(device)
target_y = target_y.to(device)
ctx_mask = ctx_mask.to(device)
tgt_mask = tgt_mask.to(device)

B, N_ctx, N_tgt = context_x.size(0), context_x.size(1), target_x.size(1)
n_valid = tgt_mask.sum(dim=1)
print(f"\nBatch: B={B}, max_ctx={N_ctx}, max_tgt={N_tgt}")
print(f"Valid targets per sample: min={n_valid.min().item()}, "
      f"mean={n_valid.float().mean().item():.1f}, max={n_valid.max().item()}")

mask_np = tgt_mask.cpu().numpy()

# =========================================================================
# 1. DETERMINISTIC REPRESENTATION ANALYSIS
# =========================================================================
print("\n" + "=" * 60)
print("1. DETERMINISTIC REPRESENTATION")
print("=" * 60)
with torch.no_grad():
    det = model.deterministic_encoder(context_x, context_y, target_x, ctx_mask)
    det_per_point_var = det.var(dim=1)
    det_mean_var = det_per_point_var.mean().item()
    det_max_var = det_per_point_var.max().item()
    det_min_var = det_per_point_var.min().item()
    det_zero_dims = (det_per_point_var < 1e-8).float().mean().item() * 100

    det_per_dim_var = det.var(dim=1).mean(dim=0)
    top_dims = det_per_dim_var.sort(descending=True)

print(f"Variance across target points per sample:")
print(f"  Mean:  {det_mean_var:.8f}")
print(f"  Range: [{det_min_var:.8f}, {det_max_var:.8f}]")
print(f"  Dims near-zero: {det_zero_dims:.1f}%")
print(f"  Top-5 dim variances: {top_dims.values[:5].cpu().tolist()}")
print(f"  Bottom-5 dim variances: {top_dims.values[-5:].cpu().tolist()}")
print(f"  Det rep norm per sample: mean={det.norm(dim=-1).mean().item():.4f}, "
      f"std={det.norm(dim=-1).std().item():.4f}")

if det_mean_var < 1e-4:
    print("  !! WARNING: Deterministic rep variance is VERY LOW -")
    print("    the model may be ignoring spatial structure from context.")
    print("    Uncertainty will be driven by position + latent z only.")

# =========================================================================
# 2. LATENT SPACE ANALYSIS (KL & posterior/prior overlap)
# =========================================================================
print("\n" + "=" * 60)
print("2. LATENT SPACE (KL & GLOBAL LATENT)")
print("=" * 60)
with torch.no_grad():
    prior = model.latent_encoder(context_x, context_y, mask=ctx_mask)
    posterior = model.latent_encoder(target_x, target_y, mask=tgt_mask)
    kl_raw = torch.distributions.kl_divergence(posterior, prior)
    kl_clamped = kl_raw.clone().clamp(min=model.free_bits)

print(f"Raw KL per dim:   min={kl_raw.min().item():.6f}  "
      f"mean={kl_raw.mean().item():.6f}  max={kl_raw.max().item():.6f}")
print(f"Clamped KL/sample:  min={kl_clamped.sum(-1).min().item():.4f}  "
      f"mean={kl_clamped.sum(-1).mean().item():.4f}  "
      f"max={kl_clamped.sum(-1).max().item():.4f}")
print(f"Expected floor: {model.free_bits} x {kl_clamped.size(-1)} = "
      f"{model.free_bits * kl_clamped.size(-1):.4f}")
print(f"Fraction dims below free_bits: "
      f"{(kl_raw < model.free_bits).float().mean().item()*100:.1f}%")
print(f"\nPrior mu    range: [{prior.loc.min().item():.3f}, {prior.loc.max().item():.3f}]")
print(f"Posterior mu range: [{posterior.loc.min().item():.3f}, {posterior.loc.max().item():.3f}]")
print(f"Mean |mu_post - mu_prior|: {(posterior.loc - prior.loc).abs().mean().item():.4f}")
print(f"Prior sigma    range: [{prior.scale.min().item():.4f}, {prior.scale.max().item():.4f}]")

latent_kl_total = kl_clamped.sum(-1).mean().item()
if latent_kl_total < 0.1:
    print("  !! WARNING: KL near zero - latent code not being used.")
    print("    The model degenerates to a deterministic NN.")
elif latent_kl_total > 10.0:
    print("  !! WARNING: KL very high - latent code may be dominating.")
    print("    The model might be ignoring context.")

# =========================================================================
# 3. EPISTEMIC vs ALEATORIC UNCERTAINTY BREAKDOWN
# =========================================================================
print("\n" + "=" * 60)
print("3. UNCERTAINTY BREAKDOWN")
print("=" * 60)
N_SAMPLES = 200
mu_samples, sigma_samples = [], []
with torch.no_grad():
    for _ in range(N_SAMPLES):
        z = prior.rsample()
        z_bc = z.unsqueeze(1).expand(-1, N_tgt, -1)
        det_rep = model.deterministic_encoder(
            context_x, context_y, target_x, ctx_mask)
        rep = torch.cat([det_rep, z_bc], dim=-1)
        _, mu, sigma = model.decoder(rep, target_x)
        mu_samples.append(mu.cpu().numpy())
        sigma_samples.append(sigma.cpu().numpy())

mu_stack = np.stack(mu_samples, axis=0)
sigma_stack = np.stack(sigma_samples, axis=0)

# Epistemic: std of mu across samples
epistemic = mu_stack.std(axis=0)
# Aleatoric: mean of sigma across samples
aleatoric = sigma_stack.mean(axis=0)
# Total predictive uncertainty from y ~ mu + sigma*noise
# Var(y) ~ Var(mu) + E[sigma^2]
aleatoric_var = (sigma_stack ** 2).mean(axis=0)
total_pred_std = np.sqrt(epistemic ** 2 + aleatoric_var)

for i, name in enumerate(['wind_dir_sin', 'wind_dir_cos', 'wind_speed_norm']):
    eps = epistemic[..., i][mask_np]
    ale = aleatoric[..., i][mask_np]
    tot = total_pred_std[..., i][mask_np]
    print(f"\n[{name:16s}]")
    print(f"  Epistemic std:  mean={eps.mean():.4f}, range=[{eps.min():.4f}, {eps.max():.4f}]")
    print(f"  Aleatoric std:  mean={ale.mean():.4f}, range=[{ale.min():.4f}, {ale.max():.4f}]")
    print(f"  Total pred std: mean={tot.mean():.4f}")
    print(f"  Epistemic/total ratio: {eps.mean() / max(tot.mean(), 1e-9):.2f}")
    eps_ratio = eps.mean() / max(tot.mean(), 1e-9)
    if eps_ratio < 0.1:
        print(f"  ** aleatoric dominated - sigma noise swamps signal")

# =========================================================================
# 4. ATTENTION PATTERN ANALYSIS
# =========================================================================
print("\n" + "=" * 60)
print("4. ATTENTION PATTERNS")
print("=" * 60)
print(f"  (analysis uses first batch sample - max_ctx={N_ctx})")
with torch.no_grad():
    # Cross-attention from deterministic encoder
    enc_input = torch.cat([context_x, context_y], dim=-1)
    hidden = model.deterministic_encoder.input_projection(enc_input)
    for attn in model.deterministic_encoder.self_attentions:
        hidden, _ = attn(hidden, hidden, hidden, mask=ctx_mask)

    B_sample, N_tgt_sample = target_x.shape[:2]
    dummy_y = torch.zeros(B_sample, N_tgt_sample, model.y_dim,
                          dtype=target_x.dtype, device=target_x.device)
    tgt_input = torch.cat([target_x, dummy_y], dim=-1)
    query = model.deterministic_encoder.input_projection(tgt_input)

    # Extract attention from the cross-attention layer
    for layer_idx, attn_mod in enumerate(model.deterministic_encoder.cross_attentions):
        with torch.no_grad():
            residual = query
            B_a, seq_q_a, _ = query.shape
            seq_k_a = hidden.size(1)
            H_a = attn_mod.h
            hs_a = attn_mod.num_hidden_per_attn

            k = attn_mod.key(hidden).view(B_a, seq_k_a, H_a, hs_a).transpose(1, 2)
            v = attn_mod.value(hidden).view(B_a, seq_k_a, H_a, hs_a).transpose(1, 2)
            q = attn_mod.query(query).view(B_a, seq_q_a, H_a, hs_a).transpose(1, 2)

            scale = hs_a ** 0.5
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
            if ctx_mask is not None:
                attn_scores = attn_scores.masked_fill(
                    ~ctx_mask[:, None, None, :], float('-inf'))
            attn_weights = torch.softmax(attn_scores, dim=-1)

            # Per-head entropy
            for h in range(H_a):
                w = attn_weights[0, h]
                valid = ctx_mask[0].float()
                w_masked = w * valid.unsqueeze(0)
                w_masked = w_masked / (w_masked.sum(dim=-1, keepdim=True) + 1e-9)
                max_ent = np.log(valid.sum().item()) if valid.sum() > 1 else 1.0
                ent = (-w_masked * torch.log(w_masked + 1e-8)).sum(dim=-1)
                kept = ctx_mask[0].sum().item()
                print(f"  Cross-attention layer {layer_idx}, head {h}: "
                      f"mean entropy={ent.mean().item():.3f}/{max_ent:.3f} "
                      f"({ent.mean().item()/max_ent*100:.0f}%) "
                      f"[{int(kept)} context pts]")

            # Measure how peaked attention is
            max_w = attn_weights.max(dim=-1).values
            mean_max_w = max_w.mean().item()
            print(f"  Mean max attention weight: {mean_max_w:.4f} "
                  f"(1/{int(1/max(mean_max_w, 1e-9))} of context)")
        if mean_max_w > 0.5:
            print(f"  ** Very peaked - each target attends to <2 context points")
            print(f"    This can cause grainy uncertainty as small position")
            print(f"    changes switch which context point is attended to.")

# =========================================================================
# 5. UNCERTAINTY ON A REGULAR GRID (like the actual wind_field plot)
# =========================================================================
print("\n" + "=" * 60)
print("5. GRID-BASED UNCERTAINTY ANALYSIS")
print("=" * 60)
n_lat, n_lon = 25, 25
alt_ft = 35000.0

from wind_map.utils import load_snapshot
obs, snapshot_time = load_snapshot('data/npy_cache', val_ids[0])

queries, lat_grid, lon_grid, _, _ = make_grid(alt_ft, n_lat, n_lon)
context_x_s, context_y_s = observations_to_tensors(obs, device)
target_x_s = queries_to_tensor(queries, device)

N_GRID_SAMPLES = 200
grid_mu_samples, grid_sigma_samples = [], []
with torch.no_grad():
    prior_s = model.latent_encoder(context_x_s, context_y_s)
    for _ in range(N_GRID_SAMPLES):
        z_s = prior_s.rsample()
        z_bc_s = z_s.unsqueeze(1).expand(-1, n_lat * n_lon, -1)
        det_s = model.deterministic_encoder(
            context_x_s, context_y_s, target_x_s)
        rep_s = torch.cat([det_s, z_bc_s], dim=-1)
        _, mu_s, sigma_s = model.decoder(rep_s, target_x_s)
        grid_mu_samples.append(mu_s.squeeze(0).cpu().numpy())
        grid_sigma_samples.append(sigma_s.squeeze(0).cpu().numpy())

grid_mu_stack = np.stack(grid_mu_samples, axis=0)
grid_sigma_stack = np.stack(grid_sigma_samples, axis=0)

grid_epistemic = grid_mu_stack.std(axis=0)
grid_aleatoric = grid_sigma_stack.mean(axis=0)
grid_aleatoric_var = (grid_sigma_stack ** 2).mean(axis=0)
grid_total_std = np.sqrt(grid_epistemic ** 2 + grid_aleatoric_var)

# Compute spatial autocorrelation of the uncertainty field
for name, field in [('epistemic_speed', grid_epistemic[:, 2]),
                    ('aleatoric_speed', grid_aleatoric[:, 2]),
                    ('total_speed', grid_total_std[:, 2])]:
    f_grid = field.reshape(n_lat, n_lon)
    # Moran's I-like: correlation between neighboring cells
    neighbor_corrs = []
    for shift in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        shifted = np.roll(np.roll(f_grid, shift[0], axis=0), shift[1], axis=1)
        mask = np.ones_like(f_grid, dtype=bool)
        if shift[0] > 0:
            mask[:shift[0]] = False
        elif shift[0] < 0:
            mask[shift[0]:] = False
        if shift[1] > 0:
            mask[:, :shift[1]] = False
        elif shift[1] < 0:
            mask[:, shift[1]:] = False
        if mask.sum() > 10:
            r, _ = pearsonr(f_grid[mask].ravel(), shifted[mask].ravel())
            neighbor_corrs.append(r)
    mean_ac = np.mean(neighbor_corrs)
    print(f"  {name:25s}: mean neighbor autocorrelation = {mean_ac:.4f}")
    if mean_ac < 0.3:
        print(f"    !! Very low spatial autocorrelation - field is noisy/grainy")
    elif mean_ac < 0.6:
        print(f"    ~~ Moderate - some structure but may appear grainy")
    else:
        print(f"    OK Smooth spatial structure")
total_speed_ac = mean_ac

# =========================================================================
# 6. SPATIAL SMOOTHNESS OF PREDICTIONS (pairwise from grid)
# =========================================================================
print("\n" + "=" * 60)
print("6. SPATIAL SMOOTHNESS OF PREDICTIONS")
print("=" * 60)

grid_mu_mean = grid_mu_stack.mean(axis=0)
grid_sigma_mean = grid_sigma_stack.mean(axis=0)
grid_pos = target_x_s.squeeze(0).cpu().numpy()

n_grid_pts = grid_pos.shape[0]
n_check = min(n_grid_pts, 500)
idxs = np.random.choice(n_grid_pts, size=n_check, replace=False)

dists, mu_diffs, sigma_diffs = [], [], []
for i in range(len(idxs)):
    for j in range(i+1, len(idxs)):
        d = np.linalg.norm(grid_pos[idxs[i], :2] - grid_pos[idxs[j], :2])
        if d < 0.15:
            dists.append(d)
            mu_diffs.append(np.linalg.norm(grid_mu_mean[idxs[i]] - grid_mu_mean[idxs[j]]))
            sigma_diffs.append(np.linalg.norm(grid_sigma_mean[idxs[i]] - grid_sigma_mean[idxs[j]]))

if len(dists) >= 20:
    dists = np.array(dists)
    mu_diffs = np.array(mu_diffs)
    sigma_diffs = np.array(sigma_diffs)
    corr_mu, _ = pearsonr(dists, mu_diffs)
    corr_sigma, _ = pearsonr(dists, sigma_diffs)
    print(f"  Mu smoothness: corr(dist, ||delta_mu||) = {corr_mu:.3f} "
          f"(positive = nearby points have similar mu)")
    print(f"  Sigma smoothness: corr(dist, ||delta_sigma||) = {corr_sigma:.3f} "
          f"(positive = nearby points have similar sigma)")
    if corr_mu < 0.2:
        print(f"  ** Low mu smoothness - adjacent points may have very different predictions")
    if corr_sigma < 0.2:
        print(f"  ** Low sigma smoothness - aleatoric noise varies rapidly in space")
else:
    print(f"  Not enough nearby point pairs ({len(dists)}) for smoothness test. Skipping.")

# =========================================================================
# 7. SIGMA DISTRIBUTION ANALYSIS
# =========================================================================
print("\n" + "=" * 60)
print("7. LEARNED NOISE (SIGMA) DISTRIBUTION")
print("=" * 60)
all_sigmas = grid_sigma_stack  # [n_samples, N, 3]
for i, name in enumerate(['wind_dir_sin', 'wind_dir_cos', 'wind_speed_norm']):
    s = all_sigmas[..., i].ravel()
    print(f"  [{name:16s}]  mean={s.mean():.4f}  median={np.median(s):.4f}  "
          f"min={s.min():.4f}  max={s.max():.4f}  "
          f"p5={np.percentile(s,5):.4f}  p95={np.percentile(s,95):.4f}")

# Check how sigma varies with z (across samples) at each grid point
sigma_cv = grid_sigma_stack.std(axis=0) / (grid_sigma_stack.mean(axis=0) + 1e-9)
for i, name in enumerate(['wind_dir_sin', 'wind_dir_cos', 'wind_speed_norm']):
    cv = sigma_cv[..., i]
    print(f"  Sigma CV [{name:16s}]: mean={cv.mean():.4f} - "
          f"how much sigma varies with latent z")

# =========================================================================
# 8. PREDICTION ERROR vs UNCERTAINTY (calibration)
# =========================================================================
print("\n" + "=" * 60)
print("8. ERROR-vs-UNCERTAINTY (on training batch)")
print("=" * 60)
with torch.no_grad():
    _, _, _, loss = model(context_x, context_y, target_x, target_y,
                          context_mask=ctx_mask, target_mask=tgt_mask)
    print(f"  Validation loss on batch: {loss.item():.4f}")

    # Get per-point predictions and errors
    z_cal = prior.rsample()
    z_bc_cal = z_cal.unsqueeze(1).expand(-1, N_tgt, -1)
    det_cal = model.deterministic_encoder(context_x, context_y, target_x, ctx_mask)
    rep_cal = torch.cat([det_cal, z_bc_cal], dim=-1)
    _, mu_cal, sigma_cal = model.decoder(rep_cal, target_x)

    mu_np_cal = mu_cal.cpu().numpy()
    sigma_np_cal = sigma_cal.cpu().numpy()
    target_np = target_y.cpu().numpy()
    error_np = np.abs(mu_np_cal - target_np)

    for i, name in enumerate(['wind_dir_sin', 'wind_dir_cos', 'wind_speed_norm']):
        err = error_np[..., i][mask_np]
        sig = sigma_np_cal[..., i][mask_np]
        # Standardised error (should be ~1 if sigma is well-calibrated)
        std_err = err / (sig + 1e-9)
        print(f"  [{name:16s}]  mean|error|={err.mean():.4f}  "
              f"mean(sigma)={sig.mean():.4f}  "
              f"mean(|error|/sigma)={std_err.mean():.4f}  "
              f"(ideally ~0.8 for Laplace, ~0.64 for Normal)")

# =========================================================================
# SUMMARY
# =========================================================================
print("\n" + "=" * 60)
print("DIAGNOSTIC SUMMARY")
print("=" * 60)
flags = []
if det_mean_var < 1e-4:
    flags.append("LOW_DET_VAR: deterministic rep is near-constant across targets")
if latent_kl_total < 0.1:
    flags.append("LOW_KL: latent code under-utilised")
if latent_kl_total > 10.0:
    flags.append("HIGH_KL: latent code may be dominating")
if eps_ratio < 0.2:
    flags.append("ALEATORIC_DOMINATED: aleatoric noise swamps epistemic signal (epistemic < 20%)")
if 'mean_max_w' in dir() and mean_max_w > 0.5:
    flags.append("SHARP_ATTENTION: each target attends to <2 context points")
if 'corr_sigma' in dir() and corr_sigma < 0.2:
    flags.append("ROUGH_SIGMA: aleatoric noise varies rapidly in space")
if total_speed_ac < 0.3:
    flags.append("GRAINY_UNCERTAINTY: low spatial autocorrelation in uncertainty")
elif total_speed_ac < 0.6:
    flags.append("MODERATE_UNCERTAINTY_SMOOTHNESS: spatial structure present but weak")

if not flags:
    print("  No major issues detected.")
else:
    print("  Issues found:")
    for f in flags:
        print(f"    ** {f}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
