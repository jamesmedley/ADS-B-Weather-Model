"""
WindSnapshotDataset and dataloader factory tests.
"""

import numpy as np
import pytest
import torch

from wind_map.preprocess import WindSnapshotDataset, make_dataloader

pytestmark = pytest.mark.unit


def test_len_matches_snapshot_count(cache_dir):
    ds = WindSnapshotDataset(cache_dir)
    n_ids = len(np.load(cache_dir + "/snapshot_ids.npy"))
    assert len(ds) == n_ids


def test_getitem_shapes_and_dtypes(cache_dir):
    ds = WindSnapshotDataset(cache_dir)
    x, y = ds[0]
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert x.shape[1] == 3 and y.shape[1] == 2
    assert x.shape[0] == y.shape[0]
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()


def test_getitem_row_count_matches_raw(cache):
    ds = WindSnapshotDataset(str(cache.path))
    for i, sid in enumerate(ds.snapshot_ids):
        x, _ = ds[i]
        assert x.shape[0] == len(cache.raw[sid])


def test_subset_ids_preserve_order_and_content(cache):
    ids = cache.ids
    reordered = [ids[2], ids[0]]
    ds = WindSnapshotDataset(str(cache.path), snapshot_ids=reordered)
    assert ds.snapshot_ids == reordered
    full = WindSnapshotDataset(str(cache.path))
    by_id = {}
    for i, sid in enumerate(full.snapshot_ids):
        by_id[sid] = full[i]
    for i, sid in enumerate(reordered):
        x, y = ds[i]
        assert torch.equal(x, by_id[sid][0])
        assert torch.equal(y, by_id[sid][1])


def test_unknown_snapshot_id_raises(cache_dir):
    with pytest.raises(KeyError):
        WindSnapshotDataset(cache_dir, snapshot_ids=[10 ** 9])


def test_snapshot_times_loaded(cache):
    ds = WindSnapshotDataset(str(cache.path))
    for sid, expected in zip(ds.snapshot_ids, ds.snapshot_times):
        assert expected == cache.times[sid]


def test_make_dataloader_yields_padded_batches(cache_dir):
    loader = make_dataloader(
        cache_dir, batch_size=4, shuffle=False, num_workers=0)
    batches = list(loader)
    assert len(batches) > 0
    (cx, cy, tx, ty, cmask, tmask) = batches[0]
    B = cx.shape[0]
    assert cx.shape == (B, cx.shape[1], 3)
    assert cy.shape == (B, cx.shape[1], 2)
    assert tx.shape[2] == 3 and ty.shape[2] == 2
    assert cmask.dtype == torch.bool and tmask.dtype == torch.bool
    assert cmask.sum() > 0 and tmask.sum() > 0


def test_make_dataloader_default_workers_ok(cache_dir):
    loader = make_dataloader(
        cache_dir, batch_size=8, shuffle=False, num_workers=0)
    (cx, cy, tx, ty, cmask, tmask) = next(iter(loader))
    assert cx.size(0) <= 8
