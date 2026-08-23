"""
Collate / batching tests.

Two collate semantics are safety-critical and pinned here:
  collate_fn     -> context is a SUBSET of target (NP training objective)
  collate_fn_val -> context and target are DISJOINT (held-out evaluation)

The coupled-rotation augmentation must preserve the physics it claims
to: pairwise distances and wind-vector relative geometry are invariant.
"""

import pytest
import torch

from wind_map.preprocess import (
    _rotate_windfield, collate_fn, collate_fn_val, pad_batch,
)

pytestmark = pytest.mark.unit


def fake_item(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, generator=g)
    y = torch.randn(n, 3, generator=g)
    return x, y


def row_in(row, rows, tol=1e-6):
    return bool((torch.abs(rows - row).max(dim=1).values < tol).any())


# ---------------------------------------------------------------- pad_batch

def test_pad_batch_masks_match_lengths():
    xs = [torch.ones(3, 3), torch.full((5, 3), 2.0)]
    ys = [torch.ones(3, 3), torch.full((5, 3), 2.0)]
    out = pad_batch(xs, ys, xs, ys, [3, 5], [3, 5])
    cx, cy, tx, ty, cmask, tmask = out
    assert cx.shape == (2, 5, 3)
    assert cmask.dtype == torch.bool
    assert cmask[0].tolist() == [True, True, True, False, False]
    assert tmask[1].all()
    assert torch.equal(cx[0, :3], xs[0])
    assert torch.equal(cx[1], xs[1])
    assert torch.count_nonzero(cx[0, 3:]) == 0


# ------------------------------------------------------------- train collate

def test_collate_context_is_subset_of_target(train_batch):
    (cx, cy, tx, ty, cmask, tmask) = train_batch
    B = cx.shape[0]
    for i in range(B):
        nc = int(cmask[i].sum())
        nt = int(tmask[i].sum())
        assert 1 <= nc < nt
        for j in range(nc):
            assert row_in(cx[i, j], tx[i, :nt])
            assert row_in(cy[i, j], ty[i, :nt])


def test_collate_skips_snapshots_with_single_observation():
    good = fake_item(6, seed=0)
    tiny = (torch.zeros(1, 3), torch.zeros(1, 3))
    torch.manual_seed(0)
    (cx, _, tx, _, cmask, tmask) = collate_fn([good, tiny])
    assert cx.shape[0] == 1
    assert int(tmask[0].sum()) == 6
    assert int(cmask[0].sum()) >= 1


def test_collate_all_invalid_batch_raises():
    a = (torch.zeros(1, 3), torch.zeros(1, 3))
    b = (torch.zeros(1, 3), torch.zeros(1, 3))
    with pytest.raises(RuntimeError):
        collate_fn([a, b])


def test_rotation_preserves_pairwise_distances():
    torch.manual_seed(0)
    angle = torch.tensor(1.234)
    x, y = fake_item(7, seed=1)
    xr, yr = _rotate_windfield(x, y, angle.cos(), angle.sin())

    def pdist(m):
        d = torch.cdist(m[:, :2], m[:, :2])
        return d

    assert torch.allclose(pdist(x), pdist(xr), atol=1e-5)
    assert torch.allclose(x[..., 2], xr[..., 2], atol=1e-6)


def test_rotation_preserves_wind_vector_gram_matrix():
    torch.manual_seed(0)
    angle = torch.tensor(-0.57)
    _, y = fake_item(6, seed=2)
    _, yr = _rotate_windfield(y.clone(), y, angle.cos(), angle.sin())
    gram_before = y[..., :2] @ y[..., :2].T
    gram_after = yr[..., :2] @ yr[..., :2].T
    assert torch.allclose(gram_before, gram_after, atol=1e-5)
    norms_before = (y[..., :2] ** 2).sum(-1)
    norms_after = (yr[..., :2] ** 2).sum(-1)
    assert torch.allclose(norms_before, norms_after, atol=1e-5)


def test_collate_rotation_option_changes_targets_consistently(cache_dir):
    from wind_map.preprocess import WindSnapshotDataset
    ds = WindSnapshotDataset(cache_dir)
    items = [ds[i] for i in range(3)]
    torch.manual_seed(0)
    plain = collate_fn(items, use_coupled_rotation=False)
    torch.manual_seed(0)
    rot = collate_fn(items, use_coupled_rotation=True)
    # Same snapshots present as targets; distances preserved although
    # coordinates differ (rotation applied).
    (_, _, t0, _, _, m0) = plain
    (_, _, t1, _, _, m1) = rot
    assert t0.shape == t1.shape
    for i in range(t0.shape[0]):
        nt = int(m0[i].sum())
        d0 = torch.cdist(t0[i, :nt, :2], t0[i, :nt, :2])
        d1 = torch.cdist(t1[i, :nt, :2], t1[i, :nt, :2])
        assert not torch.allclose(t0[i, :nt], t1[i, :nt], atol=1e-4)
        assert torch.allclose(d0, d1, atol=1e-4)


# ---------------------------------------------------------- val collate

def test_collate_val_context_target_disjoint(val_batch):
    (cx, cy, tx, ty, cmask, tmask) = val_batch
    B = cx.shape[0]
    for i in range(B):
        nc = int(cmask[i].sum())
        nt = int(tmask[i].sum())
        ctx_rows = {tuple(r.tolist()) for r in cx[i, :nc]}
        held_rows = {tuple(r.tolist()) for r in tx[i, :nt]}
        assert len(ctx_rows & held_rows) == 0


def test_collate_val_sizes_sum_to_snapshot_size(cache_dir):
    from wind_map.preprocess import WindSnapshotDataset
    ds = WindSnapshotDataset(cache_dir)
    g = torch.Generator().manual_seed(3)
    parts = {}
    for i in range(len(ds)):
        n = ds[i][0].shape[0]
        perm = torch.randperm(n, generator=g)
        parts[i] = (perm[:n // 2], perm[n // 2:])
    items = [(i, *ds[i]) for i in range(len(ds))]
    (cx, _, tx, _, cmask, tmask) = collate_fn_val(
        items, partitions=parts)
    for i in range(cx.shape[0]):
        nc = int(cmask[i].sum())
        nt = int(tmask[i].sum())
        orig_n = ds[int(items[i][0])][0].shape[0]
        assert nc >= 1 and nt >= 1
        assert nc + nt == orig_n


def test_collate_val_frozen_partitions_reused(cache_dir):
    from wind_map.preprocess import WindSnapshotDataset
    ds = WindSnapshotDataset(cache_dir)
    g = torch.Generator().manual_seed(9)
    parts = {}
    for i in range(len(ds)):
        n = ds[i][0].shape[0]
        perm = torch.randperm(n, generator=g)
        parts[i] = (perm[:n // 2], perm[n // 2:])
    items = [(i, *ds[i]) for i in range(len(ds))]
    a = collate_fn_val(items, partitions=parts)
    b = collate_fn_val(items, partitions=parts)
    for xa, xb in zip(a, b):
        assert torch.equal(xa, xb)


def test_collate_val_skips_empty_partition_side():
    x, y = fake_item(5, seed=4)
    empty = torch.zeros(0, dtype=torch.long)
    parts = {0: (empty, torch.arange(5))}
    with pytest.raises(RuntimeError):
        collate_fn_val([(0, x, y)], partitions=parts)


def test_pad_rows_are_zeroed_and_masked(cache_dir):
    from wind_map.preprocess import WindSnapshotDataset
    ds = WindSnapshotDataset(cache_dir)
    lens = [ds[i][0].shape[0] for i in range(len(ds))]
    if len(set(lens)) < 2:
        pytest.skip("synthetic cache has uniform lengths")
    torch.manual_seed(1)
    batch = collate_fn([ds[i] for i in range(len(ds))])
    (cx, cy, tx, ty, cmask, tmask) = batch
    for i in range(cx.shape[0]):
        nc = int(cmask[i].sum())
        nt = int(tmask[i].sum())
        assert torch.count_nonzero(cx[i, nc:]) == 0
        assert torch.count_nonzero(tx[i, nt:]) == 0
