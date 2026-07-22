"""
preprocess.py — Data loading and normalisation for the Wind ANP.

Reads from the flat .npy cache built by convert_db.py.

Coordinate normalisation:
  lat/lon : centred and scaled to ~[-1, 1] over the service area
  altitude: divided by 50,000 ft -> [0, 1]
  wind_dir: encoded as (sin, cos) to handle circularity
  wind_speed: z-score normalised (mean 0, std 1) over dataset

Network input x  -> [lat_norm, lon_norm, alt_norm]          (dim=3)
Network output y -> [wind_dir_sin, wind_dir_cos, speed_norm] (dim=3)
"""

import os
import math
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

# --- Normalisation constants (tune to your area) ---

CENTRE_LAT = 51.071066
CENTRE_LON = -1.042441

MAX_ALT_FT = 50_000.0
WIND_SPEED_MEAN = 34.95125809377175  # computed by convert_db.py
WIND_SPEED_STD = 21.849448514436578  # computed by convert_db.py
MIN_AIRCRAFT = 2

# Half-width of the service area in degrees (~70km radius)
LAT_RANGE_DEG = 0.63
LON_RANGE_DEG = 1.00


# --- Feature engineering ---

def normalise_coords(lat, lon, alt_ft):
    """Normalise spatial coords to approximately [-1, 1]."""
    lat_n = (lat - CENTRE_LAT) / LAT_RANGE_DEG
    lon_n = (lon - CENTRE_LON) / LON_RANGE_DEG
    alt_n = alt_ft / MAX_ALT_FT
    return lat_n, lon_n, alt_n


def encode_wind(wind_dir_deg, wind_speed_kt):
    """Encode wind direction circularly and normalise speed.

    Returns (sin, cos, speed_norm).
    """
    rad = math.radians(wind_dir_deg)
    return (
        math.sin(rad), math.cos(rad),
        (wind_speed_kt - WIND_SPEED_MEAN) / WIND_SPEED_STD
    )


def decode_wind(sin_val, cos_val, speed_norm):
    """Inverse of encode_wind.

    Returns (wind_dir_deg [0,360), wind_speed_kt).
    """
    deg = math.degrees(math.atan2(sin_val, cos_val)) % 360
    speed = speed_norm * WIND_SPEED_STD + WIND_SPEED_MEAN
    return deg, speed


# --- Day-grouped train/val/test split ---

def _parse_datetime(time_str):
    """Parse a snapshot_time string to a datetime.

    Handles ISO strings and epoch seconds. Returns UTC datetime.
    """
    s = str(time_str).strip()

    try:
        epoch = float(s)
        return datetime.utcfromtimestamp(epoch)
    except ValueError:
        pass

    iso = s.replace('T', ' ')
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        pass

    try:
        return datetime.strptime(iso[:10], '%Y-%m-%d')
    except ValueError as e:
        raise ValueError(f"Could not parse snapshot_time {time_str!r}") from e


def _parse_snapshot_date(time_str):
    """Parse a snapshot_time string to a calendar date.

    Handles ISO strings and epoch seconds.
    """
    return _parse_datetime(time_str).date()


def _closest_day_subset(day_sizes, target, rng):
    """DP subset-sum: find days whose snapshot count is closest to target."""
    order = list(range(len(day_sizes)))
    rng.shuffle(order)

    dp = {0: []}
    for idx in order:
        _, size = day_sizes[idx]
        for s in list(dp.keys()):
            new_s = s + size
            if new_s not in dp:
                dp[new_s] = dp[s] + [idx]

    best_sum = min(dp.keys(), key=lambda s: abs(s - target))
    return dp[best_sum]


def day_grouped_split(cache_dir, train_frac=0.8, test_frac=0.1, val_frac=0.1,
                      seed=42):
    """
    Split snapshots into train/val/test by calendar day (every snapshot from a
    given day goes to the same split). Uses DP subset-sum to hit the target
    fractions as closely as possible.
    """
    total_frac = train_frac + test_frac + val_frac
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {total_frac}")

    ids_path = os.path.join(cache_dir, 'snapshot_ids.npy')
    times_path = os.path.join(cache_dir, 'snapshot_times.npy')
    if not os.path.exists(ids_path):
        raise RuntimeError(
            f"No npy cache at '{cache_dir}'. "
            "Run convert_db.py first.")
    if not os.path.exists(times_path):
        raise RuntimeError(
            f"No snapshot_times.npy in '{cache_dir}'. "
            "Re-run convert_db.py.")

    ids = np.load(ids_path).tolist()
    times = np.load(times_path).tolist()
    n = len(ids)
    if n == 0:
        raise RuntimeError("No snapshots in cache.")
    if len(times) != n:
        raise RuntimeError(
            "snapshot_ids.npy and snapshot_times.npy "
            "have mismatched lengths.")

    ids_by_day = {}
    for sid, t in zip(ids, times):
        day = _parse_snapshot_date(t)
        ids_by_day.setdefault(day, []).append(sid)

    day_keys = sorted(ids_by_day.keys())
    day_sizes = [(day, len(ids_by_day[day])) for day in day_keys]

    if len(day_sizes) < 3:
        raise RuntimeError(
            f"Only {len(day_sizes)} distinct day(s) "
            "\u2014 need at least 3.")

    rng = np.random.default_rng(seed)

    # Pick days for train
    train_target = round(n * train_frac)
    train_idx = set(_closest_day_subset(day_sizes, train_target, rng))
    train_ids = [sid for i in train_idx for sid in ids_by_day[day_sizes[i][0]]]

    # Split remaining days between test and val
    remaining = [ds for i, ds in enumerate(day_sizes) if i not in train_idx]
    if len(remaining) < 2:
        raise RuntimeError("Fewer than 2 days left for val/test.")

    test_target = round(n * test_frac)
    test_idx = set(_closest_day_subset(remaining, test_target, rng))
    test_ids = [sid for i in test_idx for sid in ids_by_day[remaining[i][0]]]
    val_ids = [sid for i, ds in enumerate(remaining) if i not in test_idx
               for sid in ids_by_day[ds[0]]]

    if not train_ids:
        raise RuntimeError("Training set is empty.")
    if not val_ids:
        raise RuntimeError("Validation set is empty.")
    if not test_ids:
        raise RuntimeError("Test set is empty.")

    n_train_days = len(train_idx)
    n_test_days = len(test_idx)
    n_val_days = len(remaining) - n_test_days
    print(
        f"Day-grouped split ({len(day_sizes)} days total): "
        f"{len(train_ids)} train "
        f"({len(train_ids) / n:.1%}, {n_train_days} days)  |  "
        f"{len(val_ids)} val "
        f"({len(val_ids) / n:.1%}, {n_val_days} days)  |  "
        f"{len(test_ids)} test "
        f"({len(test_ids) / n:.1%}, {n_test_days} days)  |  "
        f"total {n} snapshots"
    )
    return train_ids, val_ids, test_ids


# --- Dataset ---

class WindSnapshotDataset(Dataset):
    """
    Loads snapshots from the .npy cache. Each item is (x, y) where:
        x [N, 3] — (lat_norm, lon_norm, alt_norm)
        y [N, 3] — (wind_dir_sin, wind_dir_cos, wind_speed_norm)
    Memory-mapped reads, no per-item parsing.
    """

    def __init__(self, cache_dir, snapshot_ids=None):
        self.cache_dir = cache_dir
        # Lazy-loaded so each DataLoader worker opens its own mmaps
        self._x = None
        self._y = None

        all_ids = np.load(os.path.join(cache_dir, 'snapshot_ids.npy'))
        offsets = np.load(os.path.join(cache_dir, 'offsets.npy'))
        id_to_range = {
            int(sid): (int(offsets[i]), int(offsets[i + 1]))
            for i, sid in enumerate(all_ids)
        }

        times_path = os.path.join(cache_dir, 'snapshot_times.npy')
        if os.path.exists(times_path):
            all_times = np.load(times_path)
            id_to_time = {
                int(sid): str(t)
                for sid, t in zip(all_ids, all_times)
            }
        else:
            id_to_time = {}

        if snapshot_ids is None:
            snapshot_ids = all_ids.tolist()
        self.snapshot_ids = list(snapshot_ids)

        try:
            self._ranges = [id_to_range[int(sid)] for sid in self.snapshot_ids]
        except KeyError as e:
            raise KeyError(f"snapshot_id {e} not found in cache.")

        self.snapshot_times = [
            id_to_time.get(int(sid))
            for sid in self.snapshot_ids
        ]

    def _ensure_loaded(self):
        if self._x is None:
            self._x = np.load(
                os.path.join(self.cache_dir, 'x.npy'),
                mmap_mode='r')
            self._y = np.load(
                os.path.join(self.cache_dir, 'y.npy'),
                mmap_mode='r')

    def __len__(self):
        return len(self.snapshot_ids)

    def __getitem__(self, idx):
        self._ensure_loaded()
        start, end = self._ranges[idx]
        x = torch.from_numpy(
            self._x[start:end].copy().astype(np.float32))
        y = torch.from_numpy(
            self._y[start:end].copy().astype(np.float32))
        return x, y


# --- Collate ---

def _rotate_wind(y, cos_a, sin_a):
    """Rotate the (sin, cos) wind-direction components.

    cos_a and sin_a can be scalars or tensors broadcastable
    with y.
    """
    y_rot = y.clone()
    s, c = y[..., 0], y[..., 1]
    y_rot[..., 0] = s * cos_a + c * sin_a
    y_rot[..., 1] = c * cos_a - s * sin_a
    return y_rot


def pad_batch(
    context_xs, context_ys, target_xs, target_ys, ctx_lens, tgt_lens,
):
    """Zero-pad variable-length context/target lists to a single batch tensor
    with boolean masks.

    Returns (context_x, context_y, target_x, target_y,
             context_mask, target_mask).
    """
    B = len(context_xs)
    max_ctx = max(ctx_lens)
    max_tgt = max(tgt_lens)
    x_dim = context_xs[0].size(-1)
    y_dim = context_ys[0].size(-1)

    context_x = context_xs[0].new_zeros(B, max_ctx, x_dim)
    context_y = context_ys[0].new_zeros(B, max_ctx, y_dim)
    target_x = target_xs[0].new_zeros(B, max_tgt, x_dim)
    target_y = target_ys[0].new_zeros(B, max_tgt, y_dim)
    context_mask = torch.zeros(B, max_ctx, dtype=torch.bool)
    target_mask = torch.zeros(B, max_tgt, dtype=torch.bool)

    for i in range(B):
        nc, nt = ctx_lens[i], tgt_lens[i]
        context_x[i, :nc] = context_xs[i]
        context_y[i, :nc] = context_ys[i]
        target_x[i, :nt] = target_xs[i]
        target_y[i, :nt] = target_ys[i]
        context_mask[i, :nc] = True
        target_mask[i, :nt] = True

    return context_x, context_y, target_x, target_y, context_mask, target_mask


def collate_fn(batch, augment=True):
    """
    Split each snapshot into context / target, with optional wind-direction
    augmentation. C is a random subset, T is the full snapshot (C ⊂ T).
    Zero-pads to batch max with bool masks.
    """
    B_raw = len(batch)
    if augment:
        angles = torch.rand(B_raw) * (2 * math.pi)
        cos_a = angles.cos()
        sin_a = angles.sin()
    else:
        cos_a = sin_a = None

    context_xs, context_ys, target_xs, target_ys = [], [], [], []
    ctx_lens, tgt_lens = [], []

    for i, (x, y) in enumerate(batch):
        n = x.size(0)
        if n < 2:
            continue

        if augment:
            y = _rotate_wind(y, cos_a[i], sin_a[i])

        n_ctx = int(n * np.random.uniform(0.25, 0.75))
        n_ctx = max(1, min(n_ctx, n - 1))

        perm = torch.randperm(n)
        ctx_idx = perm[:n_ctx]

        context_xs.append(x[ctx_idx])
        context_ys.append(y[ctx_idx])
        target_xs.append(x)
        target_ys.append(y)

        ctx_lens.append(n_ctx)
        tgt_lens.append(n)

    if not context_xs:
        raise RuntimeError(
            "Empty batch — every item had < 2 valid wind observations.")

    return pad_batch(
        context_xs, context_ys, target_xs, target_ys, ctx_lens, tgt_lens)


def _worker_init(worker_id):
    """Reseed numpy RNG per DataLoader worker."""
    import torch
    np.random.seed((torch.initial_seed() + worker_id) % 2**32)


def make_dataloader(cache_dir, batch_size=16, shuffle=True, num_workers=4):
    dataset = WindSnapshotDataset(cache_dir)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=num_workers,
        worker_init_fn=_worker_init,
        persistent_workers=num_workers > 0)


def collate_fn_val(batch):
    """collate_fn with augment=False for validation."""
    return collate_fn(batch, augment=False)
