"""
diagnose.py — Comprehensive model diagnostics for Wind ANP.
Run after training: python diagnose.py
"""
import torch
import numpy as np
from wind_map.network import LatentModel
from wind_map.preprocess import (
    WindSnapshotDataset, day_grouped_split, collate_fn_val, _worker_init
)
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

ckpt = torch.load('checkpoint/best_model.pth.tar', map_location=device, weights_only=False)
hp = ckpt['hparams']

model = LatentModel(
    hp['num_hidden'],
    num_layers=hp.get('num_layers', 4),
    free_bits=hp.get('free_bits', 0.01),
)
model.load_state_dict(ckpt['model'])
model.eval().to(device)
print(f"Loaded checkpoint: epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss'):.4f}")
print(f"Free bits: {model.free_bits}, Latent dim: {model.latent_encoder.mean_layer.out_features}")

# ---- Build a batch ----
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

# =========================================================================
# 1. KL - raw vs clamped
# =========================================================================
print("\n" + "=" * 60)
print("1. KL DIVERGENCE")
print("=" * 60)
with torch.no_grad():
    prior = model.latent_encoder(context_x, context_y, mask=ctx_mask)
    posterior = model.latent_encoder(target_x, target_y, mask=tgt_mask)
    kl_raw = torch.distributions.kl_divergence(posterior, prior)
    kl_clamped = kl_raw.clone().clamp(min=model.free_bits)

print(f"Raw KL per dim:   min={kl_raw.min().item():.6f}  "
      f"mean={kl_raw.mean().item():.6f}  max={kl_raw.max().item():.6f}")
print(f"Clamped per sample:  min={kl_clamped.sum(-1).min().item():.4f}  "
      f"mean={kl_clamped.sum(-1).mean().item():.4f}  "
      f"max={kl_clamped.sum(-1).max().item():.4f}")
print(f"Expected floor: {model.free_bits} x {kl_clamped.size(-1)} = "
      f"{model.free_bits * kl_clamped.size(-1):.4f}")
print(f"\nPrior μ    range: [{prior.loc.min().item():.3f}, {prior.loc.max().item():.3f}]")
print(f"Posterior μ range: [{posterior.loc.min().item():.3f}, {posterior.loc.max().item():.3f}]")
print(f"Mean |μ_post - μ_prior|: {(posterior.loc - prior.loc).abs().mean().item():.4f}")

# =========================================================================
# 2. Epistemic uncertainty
# =========================================================================
print("\n" + "=" * 60)
print("2. EPISTEMIC UNCERTAINTY")
print("=" * 60)
N_SAMPLES = 100
mu_samples = []
with torch.no_grad():
    for _ in range(N_SAMPLES):
        z = prior.rsample()
        z_bc = z.unsqueeze(1).expand(-1, N_tgt, -1)
        det = model.deterministic_encoder(context_x, context_y, target_x, ctx_mask)
        rep = torch.cat([det, z_bc], dim=-1)
        _, mu, _ = model.decoder(rep, target_x)
        mu_samples.append(mu.cpu().numpy())

mu_stack = np.stack(mu_samples, axis=0)  # [N, B, N_tgt, 3]
epistemic = mu_stack.std(axis=0)
mask_np = tgt_mask.cpu().numpy()
for i, name in enumerate(['wind_dir_sin', 'wind_dir_cos', 'wind_speed_norm']):
    eps_masked = epistemic[..., i][mask_np]
    print(f"Epistemic std [{name:16s}]:  mean={eps_masked.mean():.4f}  "
          f"range=[{eps_masked.min():.4f}, {eps_masked.max():.4f}]")

# =========================================================================
# 3. Deterministic rep variance (KEY METRIC)
# =========================================================================
print("\n" + "=" * 60)
print("3. DETERMINISTIC REP VARIANCE")
print("=" * 60)
with torch.no_grad():
    det = model.deterministic_encoder(context_x, context_y, target_x, ctx_mask)
    per_sample_var = det.var(dim=1)
    mean_var = per_sample_var.mean().item()
    print(f"det_rep variance across targets: mean={mean_var:.8f}  "
          f"max={per_sample_var.max().item():.8f}")
    print(f"Non-zero var dims: {(per_sample_var > 1e-8).float().mean().item()*100:.1f}%")

    # Full decoder output variance
    z = prior.rsample()
    z_bc = z.unsqueeze(1).expand(-1, N_tgt, -1)
    rep_full = torch.cat([det, z_bc], dim=-1)
    _, mu, _ = model.decoder(rep_full, target_x)
    mu_var = mu.var(dim=1).mean().item()
    print(f"Decoder μ variance across targets: {mu_var:.8f}")

# =========================================================================
# 4. Attention weights
# =========================================================================
print("\n" + "=" * 60)
print("4. ATTENTION WEIGHTS")
print("=" * 60)
with torch.no_grad():
    attn_mod = model.deterministic_encoder._attention
    k = attn_mod.mlp(context_x)   # [B, N_ctx, 128]
    q = attn_mod.mlp(target_x)    # [B, N_tgt, 128]
    H = attn_mod.multihead._num_heads
    hs = attn_mod.multihead.head_size

    q_mh = attn_mod.multihead.wq(q)
    k_mh = attn_mod.multihead.wk(k)
    q_v = q_mh[:1].view(1, N_tgt, H, hs).transpose(1, 2)
    k_v = k_mh[:1].view(1, N_ctx, H, hs).transpose(1, 2)
    scores = torch.softmax(q_v @ k_v.transpose(-2, -1) / (hs ** 0.5), dim=-1)

    for h in range(H):
        s = scores[0, h]
        max_ent = np.log(N_ctx) if N_ctx > 1 else 1.0
        ent = (-s * torch.log(s + 1e-8)).sum(dim=-1)
        print(f"Head {h}: entropy={ent.mean().item():.4f}/{max_ent:.4f} "
              f"({ent.mean().item()/max_ent*100:.1f}%)")

    # Diff between first and last target attention
    s_flat = scores.reshape(H, N_tgt, N_ctx)
    diff_0last = (s_flat[:, 0] - s_flat[:, -1]).norm(dim=-1).mean().item()
    print(f"Attention L2 diff: target[0] vs target[{N_tgt-1}]: {diff_0last:.6f}")

# =========================================================================
# 5. ROOT-CAUSE TEST: Use hidden as keys/queries (bypass MLP)
# =========================================================================
print("\n" + "=" * 60)
print("5. ROOT-CAUSE TEST: HIDDEN as KEYS/QUERIES")
print("=" * 60)
with torch.no_grad():
    # Current: det_rep from raw-coord keys
    det_orig = model.deterministic_encoder(context_x, context_y, target_x, ctx_mask)
    var_orig = det_orig.var(dim=1).mean().item()
    print(f"[CURRENT: raw-coord keys]  det_rep variance: {var_orig:.8f}")

    # Compute context hidden (already done by encoder)
    enc_in = torch.cat([context_x, context_y], dim=-1)
    hidden = model.deterministic_encoder.encoder_mlp(enc_in)
    hidden = model.deterministic_encoder.self_attn(hidden, mask=ctx_mask)

    # Compute target hidden (encode target_x with zeros for y, NO self-attn)
    dummy_y = torch.zeros(B, N_tgt, model.y_dim, device=target_x.device)
    tgt_enc_in = torch.cat([target_x, dummy_y], dim=-1)
    tgt_hidden = model.deterministic_encoder.encoder_mlp(tgt_enc_in)

    # Bypass Attention wrapper MLP: use multihead directly
    mh = model.deterministic_encoder._attention.multihead
    det_new = mh(q=tgt_hidden, k=hidden, v=hidden, mask=ctx_mask)
    var_new = det_new.var(dim=1).mean().item()
    print(f"[HIDDEN→Q/K]  det_rep variance: {var_new:.8f}")
    print(f"  Improvement factor: {var_new/var_orig:.1f}x" if var_orig > 0 else f"  From ~0 to {var_new:.8f}")

# =========================================================================
# 6. Decoder target_projection
# =========================================================================
print("\n" + "=" * 60)
print("6. DECODER TARGET PROJECTION")
print("=" * 60)
with torch.no_grad():
    proj = model.decoder.target_projection(target_x)
    proj_var = proj.var(dim=1).mean().item()
    print(f"Projected target_x variance: {proj_var:.6f}")
    print(f"Weight norm: {model.decoder.target_projection.weight.norm().item():.4f}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)