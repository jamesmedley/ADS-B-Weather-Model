"""
wind_map.train — Training loop for the Wind ANP.
"""

import os
import csv
import copy
import math
from contextlib import nullcontext

import numpy as np
import torch as t
from torch.optim.lr_scheduler import (
    SequentialLR, LinearLR, CosineAnnealingWarmRestarts,
)
from tqdm import tqdm

from wind_map.network import LatentModel
from wind_map.preprocess import (
    WindSnapshotDataset, day_grouped_split,
    collate_fn, collate_fn_val, _worker_init,
    load_params,
)
from torch.utils.data import DataLoader


class EMA:
    """Exponential moving average using a single flat buffer.

    All shadow parameters are stored in one contiguous tensor so
    the update does one ``mul_`` + one ``add_`` per step instead of
    one per parameter tensor (avoids ~100 CUDA kernel launches).
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self._params = list(model.parameters())
        self._sizes = [p.numel() for p in self._params]
        self._offsets = []
        off = 0
        for s in self._sizes:
            self._offsets.append(off)
            off += s
        self._total = off
        self.shadow = t.cat(
            [p.data.reshape(-1) for p in self._params]
        ).clone().detach()
        self._backup = t.empty_like(self.shadow)
        self._device = self.shadow.device

    @t.no_grad()
    def update(self, model):
        # Build flat view of current params (one cat, one kernel)
        flat = t.cat(
            [p.data.reshape(-1) for p in model.parameters()]
        )
        # Single mul + add on the entire buffer
        self.shadow.mul_(self.decay).add_(
            flat, alpha=1 - self.decay
        )

    def _to_shadow(self):
        """Overwrite model parameters with the shadow copy."""
        for p, off, sz in zip(
                self._params, self._offsets, self._sizes):
            p.data.copy_(self.shadow[off:off + sz].reshape(p.shape))

    def _from_model(self):
        """Copy current model parameters into the shadow buffer."""
        for p, off, sz in zip(
                self._params, self._offsets, self._sizes):
            self.shadow[off:off + sz].copy_(
                p.data.reshape(-1))

    def apply_shadow(self, model):
        """Backup real weights, then load the shadow weights."""
        flat = t.cat(
            [p.data.reshape(-1) for p in self._params])
        self._backup.copy_(flat)
        self._to_shadow()

    def restore(self, model):
        """Restore the real weights from the backup."""
        for p, off, sz in zip(
                self._params, self._offsets, self._sizes):
            p.data.copy_(
                self._backup[off:off + sz].reshape(p.shape))


def _save_checkpoint(path, ckpt):
    """Write checkpoint to *path* via an atomic tmp file."""
    tmp = path + '.tmp'
    t.save(ckpt, tmp)
    os.replace(tmp, path)


def train(cache_dir, num_hidden=128, num_latents=None, epochs=200,
          batch_size=16, num_workers=4,
          latent_layers=4, deterministic_layers=4, dropout=0.0,
          init_checkpoint=None,
          split_seed=42, lr=1e-3, warmup_steps=4000,
          warmup_frac=None, kl_warmup_steps=2000,
          free_bits=0.001, checkpoint_dir='./checkpoint',
          save_checkpoint=True,
          run_test_eval=True, verbose=True, patience=50,
          ema_decay=0.999,
          use_amp=True,
          weight_decay=1e-5,
          use_dist_bias=True,
          num_decoder_layers=3):
    """
    Train the Wind ANP.

    save_checkpoint=False keeps the best model in memory
    only (for search trials).
    run_test_eval=False skips the held-out test eval
    (test set should only be touched once, on the final
    chosen config).

    use_amp: use automatic mixed precision (1.5-3x speedup).

    Returns dict with: best_val_loss, best_epoch,
    checkpoint_path, test_loss.
    """
    device = t.device(
        'cuda' if t.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Training on {device}")

    # --- Data ---
    train_ids, val_ids, test_ids = day_grouped_split(
        cache_dir, train_frac=0.8, test_frac=0.1,
        val_frac=0.1, seed=split_seed)

    train_ds = WindSnapshotDataset(
        cache_dir, snapshot_ids=train_ids)
    val_ds = WindSnapshotDataset(
        cache_dir, snapshot_ids=val_ids)
    test_ds = WindSnapshotDataset(
        cache_dir, snapshot_ids=test_ids)

    use_persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
        persistent_workers=use_persistent_workers,
        worker_init_fn=_worker_init,
        pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn_val, num_workers=num_workers,
        persistent_workers=use_persistent_workers,
        worker_init_fn=_worker_init,
        pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn_val, num_workers=num_workers,
        persistent_workers=use_persistent_workers,
        worker_init_fn=_worker_init,
        pin_memory=(device.type == 'cuda'))

    # --- Normalisation params (saved in checkpoint for inference) ---
    norm_params = load_params(cache_dir)

    # --- Model ---
    model = LatentModel(
        num_hidden, num_latents=num_latents, x_dim=3, y_dim=3,
        latent_layers=latent_layers,
        deterministic_layers=deterministic_layers,
        dropout=dropout,
        free_bits=free_bits,
        use_dist_bias=use_dist_bias,
        num_decoder_layers=num_decoder_layers).to(device)

    if init_checkpoint is not None:
        if verbose:
            print(
                f"Loading pretrained weights from "
                f"{init_checkpoint} ...")
        pre_ckpt = t.load(
            init_checkpoint, map_location=device,
            weights_only=False)
        model.load_state_dict(pre_ckpt['model'])
        if verbose:
            ep = pre_ckpt['epoch']
            vl = pre_ckpt['val_loss']
            print(
                f"  Loaded (epoch {ep}, "
                f"val_loss={vl:.4f})")

    optim = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # AMP setup
    if use_amp and device.type == 'cuda':
        cc = t.cuda.get_device_capability()
        if cc >= (8, 0):  # Ampere+: native bf16 tensor cores
            amp_dtype = t.bfloat16
            scaler = None
        else:              # Turing/Volta: fp16 tensor cores + GradScaler
            amp_dtype = t.float16
            scaler = t.amp.GradScaler()
        if verbose:
            print(f"  AMP enabled (dtype={amp_dtype}, sm_{cc[0]}{cc[1]})")
    else:
        amp_dtype = None
        scaler = None
        if use_amp and verbose:
            print("  AMP requested but CUDA unavailable.")

    steps_per_epoch = len(train_loader)
    if warmup_frac is not None:
        warmup_steps = max(
            1, int(warmup_frac * epochs * steps_per_epoch))

    # --- LR schedule: warmup + cosine annealing ---
    warmup_sched = LinearLR(
        optim, start_factor=1e-3, total_iters=warmup_steps)
    cosine_T0 = steps_per_epoch * 20
    cosine_sched = CosineAnnealingWarmRestarts(
        optim, T_0=cosine_T0, T_mult=2, eta_min=lr * 0.01)
    scheduler = SequentialLR(
        optim, [warmup_sched, cosine_sched],
        milestones=[warmup_steps])

    if verbose:
        total_steps = epochs * steps_per_epoch
        pct = 100 * warmup_steps / max(total_steps, 1)
        print(
            f"  LR: warmup {warmup_steps} steps "
            f"({pct:.1f}%), then cosine restarts "
            f"(T_0={cosine_T0})")

    # --- EMA ---
    ema = EMA(model, decay=ema_decay)

    global_step = 0
    best_val_loss = float('inf')
    best_epoch = None
    best_state_dict = None
    epochs_since_improvement = 0
    log_interval = 50
    best_ckpt_path = (
        os.path.join(
            checkpoint_dir, 'best_model.pth.tar')
        if save_checkpoint else None
    )

    if save_checkpoint:
        os.makedirs(checkpoint_dir, exist_ok=True)
        csv_path = os.path.join(checkpoint_dir, 'training_log.csv')
        csv_columns = [
            'epoch', 'lr',
            'train_loss', 'val_loss',
            'val_nll', 'val_kl',
            'speed_rmse_kt', 'speed_mae_kt', 'dir_mae_deg',
            'cov_68', 'cov_95',
            'active_pct', 'kl_max', 'kl_min',
            'sigma_speed_kt', 'sigma_dir_deg',
            'best_so_far',
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_columns)
    else:
        csv_path = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        running_loss = t.zeros((), device=device)
        running_kl = t.zeros((), device=device)
        desc = f"Epoch {epoch}/{epochs} [train]"
        pbar = tqdm(
            train_loader, desc=desc,
            disable=not verbose)

        non_blocking = (device.type == 'cuda')

        for batch in pbar:
            (context_x, context_y, target_x, target_y,
             context_mask, target_mask) = [
                x.to(device, non_blocking=non_blocking) for x in batch
            ]
            global_step += 1

            kl_weight = min(1.0, global_step / max(kl_warmup_steps, 1))

            with (t.cuda.amp.autocast(dtype=amp_dtype)
                  if amp_dtype is not None else nullcontext()):
                mu, sigma, kl, kl_per_dim, loss = model(
                    context_x, context_y,
                    target_x, target_y,
                    context_mask=context_mask,
                    target_mask=target_mask,
                    kl_weight=kl_weight)

            optim.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            t.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)
            if scaler is not None:
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
            scheduler.step()
            ema.update(model)

            train_loss_sum += loss.item()
            running_loss += loss.detach()
            if kl is not None:
                running_kl += kl.mean().detach()
            if (verbose
                    and global_step % log_interval == 0):
                rl = (running_loss / log_interval
                      ).item()
                kl_val = (
                    (running_kl / log_interval).item()
                    if kl is not None else "N/A")
                pbar.set_postfix(
                    loss=f"{rl:.4f}",
                    kl=f"{kl_val:.4f}"
                    if kl is not None else "N/A")
                running_loss.zero_()
                running_kl.zero_()

        avg_train = train_loss_sum / len(train_loader)

        # Validate (using EMA weights)
        ema.apply_shadow(model)
        model.eval()

        val_loss_sum = 0.0
        val_nll_sum = 0.0
        val_kl_sum = 0.0
        n_points = 0
        n_batches = 0

        speed_abs_err_sum = 0.0
        speed_sq_err_sum = 0.0
        dir_abs_err_sum = 0.0

        cov_68_sum = 0.0
        cov_95_sum = 0.0

        sigma_speed_sum = 0.0
        sigma_dir_sum = 0.0

        kl_dim_sum = None
        num_latents = 0

        with t.no_grad():
            for batch in val_loader:
                (context_x, context_y,
                 target_x, target_y,
                 context_mask, target_mask) = [
                    x.to(device, non_blocking=non_blocking) for x in batch
                ]
                with (t.cuda.amp.autocast(dtype=amp_dtype)
                      if amp_dtype is not None else nullcontext()):
                    mu, sigma, kl, kl_per_dim, loss = model(
                        context_x, context_y,
                        target_x, target_y,
                        context_mask=context_mask,
                        target_mask=target_mask)

                val_loss_sum += loss.item()

                mask_f = target_mask.to(mu.dtype)

                # Per-target-point KL (clamped)
                kl_valid = (kl * mask_f).sum() / mask_f.sum().clamp(min=1.0)
                val_kl_sum += kl_valid.item()

                # Per-dimension KL (unclamped) for active dim tracking
                if kl_per_dim is not None:
                    num_latents = kl_per_dim.size(-1)
                    batch_avg_dim = kl_per_dim.mean(dim=0)  # [latent_dim]
                    if kl_dim_sum is None:
                        kl_dim_sum = batch_avg_dim.cpu()
                    else:
                        kl_dim_sum += batch_avg_dim.cpu()
                    n_batches += 1

                # Reconstruction NLL
                dist = t.distributions.Normal(mu, sigma)
                log_p = dist.log_prob(target_y).sum(dim=-1)
                n_valid = mask_f.sum()
                val_nll_sum += (-log_p * mask_f).sum().item()
                n_points += n_valid.item()

                # Physical-space metrics (on CPU)
                mu_np = mu.cpu().numpy()
                sigma_np = sigma.cpu().numpy()
                target_y_np = target_y.cpu().numpy()
                mask_np = target_mask.cpu().numpy().astype(bool)

                sin_pred, cos_pred, spd_norm_pred = (
                    mu_np[..., 0], mu_np[..., 1], mu_np[..., 2])
                sin_true, cos_true, spd_norm_true = (
                    target_y_np[..., 0], target_y_np[..., 1], target_y_np[..., 2])

                pred_dir = np.degrees(
                    np.arctan2(sin_pred, cos_pred)) % 360
                true_dir = np.degrees(
                    np.arctan2(sin_true, cos_true)) % 360

                pred_log_spd = (spd_norm_pred * norm_params.log_speed_std
                                + norm_params.log_speed_mean)
                pred_speed = np.exp(pred_log_spd) - 1
                true_log_spd = (spd_norm_true * norm_params.log_speed_std
                                + norm_params.log_speed_mean)
                true_speed = np.exp(true_log_spd) - 1

                speed_err = pred_speed - true_speed
                dir_diff = np.abs(pred_dir - true_dir) % 360
                dir_err = np.minimum(dir_diff, 360 - dir_diff)

                m = mask_np
                speed_abs_err_sum += np.abs(speed_err[m]).sum()
                speed_sq_err_sum += (speed_err[m] ** 2).sum()
                dir_abs_err_sum += dir_err[m].sum()

                # Coverage: 1σ / 2σ intervals via delta-method approximation
                R2 = sin_pred ** 2 + cos_pred ** 2 + 1e-9
                var_dir_rad = (
                    ((cos_pred / R2) * sigma_np[..., 0]) ** 2
                    + ((sin_pred / R2) * sigma_np[..., 1]) ** 2)
                std_dir_deg = np.degrees(np.sqrt(var_dir_rad))

                std_speed_kt = (
                    sigma_np[..., 2] * norm_params.log_speed_std
                    * np.exp(pred_log_spd))

                cov_68_sum += (np.abs(speed_err[m]) <= std_speed_kt[m]).sum()
                cov_68_sum += (dir_err[m] <= std_dir_deg[m]).sum()
                cov_95_sum += (np.abs(speed_err[m]) <= 2 * std_speed_kt[m]).sum()
                cov_95_sum += (dir_err[m] <= 2 * std_dir_deg[m]).sum()

                # Mean predicted sigma in physical units
                sigma_speed_sum += std_speed_kt[m].sum()
                sigma_dir_sum += std_dir_deg[m].sum()

        ema.restore(model)

        avg_val = val_loss_sum / len(val_loader)
        avg_nll = val_nll_sum / n_points
        avg_kl = val_kl_sum / n_batches if n_batches > 0 else 0.0
        speed_rmse = math.sqrt(speed_sq_err_sum / n_points)
        speed_mae = speed_abs_err_sum / n_points
        dir_mae = dir_abs_err_sum / n_points
        cov_68 = cov_68_sum / (2 * n_points)
        cov_95 = cov_95_sum / (2 * n_points)
        mean_sigma_speed = sigma_speed_sum / n_points
        mean_sigma_dir = sigma_dir_sum / n_points

        active_pct = 0.0
        kl_max_dim = 0.0
        kl_min_dim = 0.0
        if kl_dim_sum is not None and n_batches > 0:
            avg_kl_per_dim = kl_dim_sum / n_batches
            active_mask = avg_kl_per_dim > free_bits
            active_pct = (
                active_mask.sum().item() / num_latents * 100)
            kl_max_dim = avg_kl_per_dim.max().item()
            kl_min_dim = avg_kl_per_dim.min().item()

        if verbose:
            print(
                f"  -> train_loss={avg_train:.4f}"
                f"  val_loss={avg_val:.4f}")
            print(
                f"  -> val_nll={avg_nll:.4f}"
                f"  val_kl={avg_kl:.4f}"
                f"  speed_rmse={speed_rmse:.2f}kt"
                f"  speed_mae={speed_mae:.2f}kt"
                f"  dir_mae={dir_mae:.1f}°"
                f"  cov_68={cov_68:.1%}"
                f"  cov_95={cov_95:.1%}"
                f"  active={active_pct:.0f}%"
                f"  σ_s={mean_sigma_speed:.2f}kt"
                f"  σ_d={mean_sigma_dir:.1f}°")

        # Write CSV row
        if csv_path is not None:
            current_lr = optim.param_groups[0]['lr']
            is_best = avg_val < best_val_loss
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, f'{current_lr:.6f}',
                    f'{avg_train:.6f}', f'{avg_val:.6f}',
                    f'{avg_nll:.6f}', f'{avg_kl:.6f}',
                    f'{speed_rmse:.4f}', f'{speed_mae:.4f}', f'{dir_mae:.4f}',
                    f'{cov_68:.6f}', f'{cov_95:.6f}',
                    f'{active_pct:.2f}', f'{kl_max_dim:.6f}', f'{kl_min_dim:.6f}',
                    f'{mean_sigma_speed:.4f}', f'{mean_sigma_dir:.4f}',
                    '1' if is_best else '0',
                ])

        # Checkpoint best epoch
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            epochs_since_improvement = 0

            if save_checkpoint:
                # Build checkpoint on main thread (GPU work),
                # then write to disk in a background thread so
                # the GPU is not idle during the disk I/O.
                ema.apply_shadow(model)
                ckpt = {
                    'epoch': epoch,
                    'model': {
                        k: v.cpu()
                        for k, v in model.state_dict().items()
                    },
                    'optimizer': {
                        k: v.cpu() if isinstance(v, t.Tensor) else v
                        for k, v in optim.state_dict().items()
                    },
                    'scheduler': scheduler.state_dict(),
                    'val_loss': avg_val,
                    'norm_params': norm_params.to_dict(),
                    'hparams': {
                        'num_hidden': num_hidden,
                        'num_latents': num_latents if num_latents is not None else num_hidden,
                        'latent_layers': latent_layers,
                        'deterministic_layers': deterministic_layers,
                        'dropout': dropout,
                        'lr': lr,
                        'batch_size': batch_size,
                        'ema_decay': ema_decay,
                        'kl_warmup_steps': kl_warmup_steps,
                        'free_bits': free_bits,
                        'weight_decay': weight_decay,
                        'use_dist_bias': use_dist_bias,
                        'num_decoder_layers': num_decoder_layers,
                    },
                }
                ema.restore(model)
                _save_checkpoint(best_ckpt_path, ckpt)
                if verbose:
                    bvl = best_val_loss
                    print(
                        f"  New best "
                        f"val_loss={bvl:.4f} saved.")
            else:
                ema.apply_shadow(model)
                best_state_dict = (
                    copy.deepcopy(model.state_dict()))
                ema.restore(model)
        else:
            epochs_since_improvement += 1
            if (patience is not None
                    and patience > 0
                    and epochs_since_improvement
                    >= patience):
                if verbose:
                    esi = epochs_since_improvement
                    print(
                        f"  No improvement for "
                        f"{esi} epochs "
                        f"(patience={patience}). "
                        f"Early stop at "
                        f"epoch {epoch}.")
                break

    if verbose:
        print("Training complete.")

    # Final test evaluation
    avg_test = None
    if run_test_eval:
        if verbose:
            print(
                "Evaluating best checkpoint on "
                "held-out test set...")
        if save_checkpoint:
            best_ckpt = t.load(
                best_ckpt_path, map_location=device,
                weights_only=False)
            model.load_state_dict(
                best_ckpt['model'])
        elif best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        model.eval()
        test_loss_sum = 0.0
        with t.no_grad():
            for batch in test_loader:
                (context_x, context_y,
                 target_x, target_y,
                 context_mask, target_mask) = [
                    x.to(device, non_blocking=non_blocking) for x in batch
                ]
                with (t.cuda.amp.autocast(dtype=amp_dtype)
                      if amp_dtype is not None else nullcontext()):
                    _, _, _, _, loss = model(
                        context_x, context_y,
                        target_x, target_y,
                        context_mask=context_mask,
                        target_mask=target_mask)
                test_loss_sum += loss.item()
        avg_test = (
            test_loss_sum / len(test_loader))
        if verbose:
            bvl = best_val_loss
            print(
                f"  -> test_loss={avg_test:.4f}"
                f"  (best val_loss={bvl:.4f},"
                f" epoch {best_epoch})")

    return {
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'checkpoint_path': best_ckpt_path,
        'test_loss': avg_test,
    }
