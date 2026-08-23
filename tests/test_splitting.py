"""
Day-grouped train/val/test splitting tests.

The splitter is the project's primary defence against temporal data
leakage: every snapshot from a calendar day must land in exactly one
split, and a fixed seed must reproduce the same split bit-for-bit
(documented RNG-consumption contract in preprocess.py).
"""

import numpy as np
import pytest

from wind_map.preprocess import day_grouped_split, split_snapshot_ids
from wind_map.utils import _parse_datetime

pytestmark = pytest.mark.unit


def make_ids_and_times(n_days=8, per_day=3):
    ids, times = [], []
    sid = 1
    for day in range(n_days):
        for _ in range(per_day):
            ids.append(sid)
            times.append(f"2024-06-{day + 1:02d}T12:00:00")
            sid += 1
    return ids, times


def test_fractions_must_sum_to_one():
    ids, times = make_ids_and_times()
    with pytest.raises(ValueError):
        split_snapshot_ids(ids, times, 0.8, 0.1, 0.05)


def test_mismatched_lengths_raise():
    ids, times = make_ids_and_times()
    with pytest.raises(RuntimeError):
        split_snapshot_ids(ids, times[:-1])


def test_empty_input_raises():
    with pytest.raises(RuntimeError):
        split_snapshot_ids([], [])


@pytest.mark.parametrize("n_days", [1, 2])
def test_needs_at_least_three_days(n_days):
    ids, times = make_ids_and_times(n_days=n_days)
    with pytest.raises(RuntimeError):
        split_snapshot_ids(ids, times)


def test_splits_are_disjoint_and_cover_everything():
    ids, times = make_ids_and_times(n_days=10, per_day=4)
    tr, va, te = split_snapshot_ids(
        ids, times, verbose=False, seed=42)
    tr_s, va_s, te_s = set(tr), set(va), set(te)
    assert not (tr_s & va_s), "train/val share snapshots"
    assert not (tr_s & te_s), "train/test share snapshots"
    assert not (va_s & te_s), "val/test share snapshots"
    assert tr_s | va_s | te_s == set(ids)
    assert len(tr) + len(va) + len(te) == len(ids)


def test_no_calendar_day_spans_two_splits():
    """Core leakage guard: a day's weather must never appear in both
    training and held-out splits."""
    ids, times = make_ids_and_times(n_days=9, per_day=3)
    day_of = {sid: _parse_datetime(t).date()
              for sid, t in zip(ids, times)}
    tr, va, te = split_snapshot_ids(ids, times, verbose=False)
    tr_days = {day_of[s] for s in tr}
    va_days = {day_of[s] for s in va}
    te_days = {day_of[s] for s in te}
    assert not (tr_days & va_days)
    assert not (tr_days & te_days)
    assert not (va_days & te_days)


def test_same_seed_reproduces_split_exactly():
    ids, times = make_ids_and_times(n_days=8, per_day=5)
    a = split_snapshot_ids(ids, times, verbose=False, seed=42)
    b = split_snapshot_ids(ids, times, verbose=False, seed=42)
    assert a == b


def test_other_seeds_produce_valid_reproducible_splits():
    """Seed plumbs into the DP shuffle; whatever days it selects,
    every seed must give a valid, bit-for-bit reproducible split."""
    ids, times = make_ids_and_times(n_days=8, per_day=5)
    for seed in (0, 7, 42, 43):
        a = split_snapshot_ids(ids, times, verbose=False,
                               seed=seed)
        b = split_snapshot_ids(ids, times, verbose=False,
                               seed=seed)
        assert a == b
        tr_s = set(a[0])
        assert 0 < len(tr_s) < len(ids)


def test_split_sizes_approximate_targets():
    n_per = [10, 12, 11, 14, 13, 15, 16, 18, 12, 9]
    ids, times = [], []
    sid = 1
    for day, size in enumerate(n_per):
        for _ in range(size):
            ids.append(sid)
            times.append(f"2024-07-{day + 1:02d}T06:00:00")
            sid += 1
    n = len(ids)
    tr, va, te = split_snapshot_ids(ids, times, verbose=False)
    assert abs(len(tr) / n - 0.8) < 0.15
    assert abs(len(te) / n - 0.1) < 0.15
    assert abs(len(va) / n - 0.1) < 0.15
    assert len(tr) > 0 and len(va) > 0 and len(te) > 0


def test_parse_datetime_iso_and_epoch():
    dt = _parse_datetime("2024-06-01T09:00:00")
    assert (dt.year, dt.month, dt.day, dt.hour) == (2024, 6, 1, 9)
    dt2 = _parse_datetime(str(1717222800.0))
    assert isinstance(dt2.year, int)
    with pytest.raises(ValueError):
        _parse_datetime("not-a-date")


def test_day_grouped_split_reads_cache(cache_dir):
    tr, va, te = day_grouped_split(cache_dir)
    all_ids = np.load(
        cache_dir + "/snapshot_ids.npy").tolist()
    assert set(tr) | set(va) | set(te) == set(all_ids)
    assert not (set(tr) & set(va))
    assert not (set(tr) & set(te))


def test_day_grouped_split_missing_cache_raises(tmp_path):
    with pytest.raises(RuntimeError):
        day_grouped_split(str(tmp_path / "nope"))
