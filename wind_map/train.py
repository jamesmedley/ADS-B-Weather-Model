"""
wind_map.train — Training loop for the Wind ANP.
"""

import os
import copy

import torch as t
from torch.optim.lr_scheduler import (
    SequentialLR, LinearLR, CosineAnnealingWarmRestarts,
)
from tqdm import tqdm

from wind_map.network import LatentModel
from wind_map.preprocess import (
    WindSnapshotDataset, day_grouped_split,
    collate_fn, collate_fn_val,
)
from torch.utils.data import DataLoader


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }

    @t.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(
                param.data, alpha=1 - self.decay
            )

    def apply_shadow(self, model):
        self.backup = {
            name: param.clone()
            for name, param in model.named_parameters()
        }
        for name, param in model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            param.data.copy_(self.backup[name])
        del self.backup


def train(cache_dir, num_hidden=128, epochs=200,
          batch_size=16, num_workers=4,
          num_layers=2, ffn_expansion=2, dropout=0.0,
          init_checkpoint=None,
          split_seed=42, lr=1e-3, warmup_steps=4000,
          warmup_frac=None,
          checkpoint_dir='./checkpoint',
          save_checkpoint=True,
          run_test_eval=True, verbose=True, patience=50,
          ema_decay=0.999):
    """
    Train the Wind ANP.

    save_checkpoint=False keeps the best model in memory
    only (for search trials).
    run_test_eval=False skips the held-out test eval
    (test set should only be touched once, on the final
    chosen config).

    Returns dict with: best_val_loss, best_epoch,
    checkpoint_path, test_loss.
    """
    device = t.device(
        'cuda' if t.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    if verbose:
        amp_msg = " + AMP" if use_amp else ""
        print(f"Training on {device}{amp_msg}")

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
        persistent_workers=use_persistent_workers)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn_val, num_workers=num_workers,
        persistent_workers=use_persistent_workers)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn_val, num_workers=num_workers,
        persistent_workers=use_persistent_workers)

    # --- Model ---
    model = LatentModel(
        num_hidden, x_dim=3, y_dim=3,
        num_layers=num_layers,
        ffn_expansion=ffn_expansion,
        dropout=dropout).to(device)

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

    optim = t.optim.Adam(model.parameters(), lr=lr)

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

    # --- AMP scaler ---
    scaler = t.amp.GradScaler(device.type)

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

        for batch in pbar:
            (context_x, context_y, target_x, target_y,
             context_mask, target_mask) = batch
            global_step += 1

            context_x = context_x.to(device)
            context_y = context_y.to(device)
            target_x = target_x.to(device)
            target_y = target_y.to(device)
            context_mask = context_mask.to(device)
            target_mask = target_mask.to(device)

            with t.amp.autocast(device.type, enabled=use_amp):
                mu, sigma, kl, loss = model(
                    context_x, context_y,
                    target_x, target_y,
                    context_mask=context_mask,
                    target_mask=target_mask)

            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            t.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()
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
        with t.no_grad():
            for batch in val_loader:
                (context_x, context_y,
                 target_x, target_y,
                 context_mask, target_mask) = batch
                context_x = context_x.to(device)
                context_y = context_y.to(device)
                target_x = target_x.to(device)
                target_y = target_y.to(device)
                context_mask = (
                    context_mask.to(device))
                target_mask = (
                    target_mask.to(device))
                with t.amp.autocast(
                        device.type, enabled=use_amp):
                    _, _, _, loss = model(
                        context_x, context_y,
                        target_x, target_y,
                        context_mask=context_mask,
                        target_mask=target_mask)
                val_loss_sum += loss.item()
        ema.restore(model)

        avg_val = val_loss_sum / len(val_loader)
        if verbose:
            print(
                f"  -> train_loss={avg_train:.4f}"
                f"  val_loss={avg_val:.4f}")

        # Checkpoint best epoch
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch
            epochs_since_improvement = 0

            if save_checkpoint:
                # Save with EMA weights
                ema.apply_shadow(model)
                ckpt = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'val_loss': avg_val,
                    'hparams': {
                        'num_hidden': num_hidden,
                        'num_layers': num_layers,
                        'ffn_expansion': ffn_expansion,
                        'dropout': dropout,
                        'lr': lr,
                        'batch_size': batch_size,
                        'ema_decay': ema_decay,
                    },
                }
                t.save(ckpt, best_ckpt_path)
                ema.restore(model)
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
                 context_mask, target_mask) = batch
                context_x = context_x.to(device)
                context_y = context_y.to(device)
                target_x = target_x.to(device)
                target_y = target_y.to(device)
                context_mask = (
                    context_mask.to(device))
                target_mask = (
                    target_mask.to(device))
                with t.amp.autocast(
                        device.type, enabled=use_amp):
                    _, _, _, loss = model(
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
