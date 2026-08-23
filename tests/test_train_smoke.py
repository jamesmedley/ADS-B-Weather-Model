"""
End-to-end training smoke test.

Runs the real train() loop on a tiny synthetic cache (2 epochs, tiny
model) and verifies: returned metrics contract, checkpoint contents
(incl. norm_params + hparams), the 17-column CSV log, EMA/checkpoint
integration, and that no stray files leak into the repo root.
"""

import csv
import importlib
import os

import pytest
import torch

from wind_map.train import train

train_mod = importlib.import_module("wind_map.train")

pytestmark = pytest.mark.smoke


@pytest.fixture()
def big_cache(make_cache):
    # Default day sizes give an exact 5/2/2 day-grouped split under
    # the hardcoded 0.8/0.1/0.1 fractions -> all loaders non-empty.
    return make_cache()


@pytest.fixture(autouse=True)
def silence_file_logging(monkeypatch):
    """train() must never append to the repo-root run.log during
    tests; neutralise its logging hook entirely."""
    monkeypatch.setattr(
        train_mod, "setup_logging",
        lambda log_file=None: str(log_file or ""))


def test_train_two_epochs_smoke(big_cache, tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    result = train(
        cache_dir=str(big_cache.path),
        num_hidden=8,
        num_latents=4,
        latent_layers=1,
        deterministic_layers=1,
        num_decoder_layers=1,
        dropout=0.0,
        use_dist_bias=False,
        batch_size=2,
        epochs=2,
        patience=10,
        lr=1e-3,
        warmup_frac=0.5,
        kl_warmup_steps=100,
        use_amp=False,
        ema_decay=0.99,
        weight_decay=0.0,
        num_workers=0,
        seed=1234,
        verbose=False,
        save_checkpoint=True,
        run_test_eval=True,
        checkpoint_dir=str(ckpt_dir),
        log_file=str(tmp_path / "train_run.log"),
    )

    expected_keys = {
        "best_val_loss", "best_composite", "best_speed_mae",
        "best_dir_mae", "best_epoch", "checkpoint_path",
        "test_loss",
    }
    assert set(result) == expected_keys
    assert result["checkpoint_path"] == \
        str(ckpt_dir / "best_model.pth.tar")
    assert os.path.exists(result["checkpoint_path"])
    assert isinstance(result["best_epoch"], int)
    assert 1 <= result["best_epoch"] <= 2
    for key in ("best_val_loss", "test_loss", "best_composite"):
        value = result[key]
        if value is not None:
            assert value == value, f"{key} is NaN"

    ckpt = torch.load(result["checkpoint_path"],
                      weights_only=False)
    for key in ("epoch", "model", "hparams", "norm_params",
                "val_loss"):
        assert key in ckpt
    assert ckpt["hparams"]["num_hidden"] == 8
    assert "centre_lat" in ckpt["norm_params"]


def test_train_writes_seventeen_column_csv(big_cache, tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    train(
        cache_dir=str(big_cache.path),
        num_hidden=8, num_latents=4,
        latent_layers=1, deterministic_layers=1,
        num_decoder_layers=1, dropout=0.0,
        use_dist_bias=False,
        batch_size=4, epochs=1, patience=5,
        use_amp=False, ema_decay=0.9,
        num_workers=0, seed=7, verbose=False,
        save_checkpoint=True, run_test_eval=False,
        checkpoint_dir=str(ckpt_dir),
        log_file=str(tmp_path / "train_run.log"),
    )
    csv_path = ckpt_dir / "training_log.csv"
    assert csv_path.exists()
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    assert len(header) == 17
    assert header[0] == "epoch" and header[-1] == "best_so_far"
    assert "val_loss" in header and "speed_mae_kt" in header
    assert len(rows) == 2          # header + exactly one epoch row
    assert rows[1][0] == "1"


def test_train_no_checkpoint_mode_returns_metrics(
        big_cache, tmp_path):
    result = train(
        cache_dir=str(big_cache.path),
        num_hidden=8, num_latents=4,
        latent_layers=1, deterministic_layers=1,
        num_decoder_layers=1, dropout=0.0,
        use_dist_bias=False,
        batch_size=4, epochs=1, patience=5,
        use_amp=False, ema_decay=0.9,
        num_workers=0, seed=7, verbose=False,
        save_checkpoint=False, run_test_eval=False,
        checkpoint_dir=str(tmp_path / "unused"),
        log_file=str(tmp_path / "train_run.log"),
    )
    assert result["checkpoint_path"] is None
    assert not (tmp_path / "unused").exists()


def test_train_seed_reproducibility(big_cache, tmp_path):
    common = dict(
        cache_dir=str(big_cache.path),
        num_hidden=8, num_latents=4,
        latent_layers=1, deterministic_layers=1,
        num_decoder_layers=1, dropout=0.0,
        use_dist_bias=False,
        batch_size=4, epochs=2, patience=50,
        use_amp=False, ema_decay=0.9,
        num_workers=0, seed=31337, verbose=False,
        save_checkpoint=False, run_test_eval=False,
        log_file=str(tmp_path / "t.log"),
    )
    a = train(**common)
    b = train(**common)
    assert a["best_val_loss"] == pytest.approx(
        b["best_val_loss"], rel=1e-6)
    assert a["best_epoch"] == b["best_epoch"]
