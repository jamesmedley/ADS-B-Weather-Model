"""
preprocess.py — Data loading and normalisation for the Wind ANP.

Reads from the flat .npy cache built by convert_db.py.

Normalisation (data-driven, stored in meta.json):
  lat/lon : converted to local km coordinates, centred at dataset
            median, scaled to ~[-1, 1] by isotropic range_km
  altitude: log(1 + ft) / log(1 + max_alt_ft)  ->  [0, ~1]
  wind    : encoded as meteorological (u, v) components, z-scored

Network input x  -> [lat_norm, lon_norm, alt_norm]  (dim=3)
Network output y -> [wind_u_norm, wind_v_norm]       (dim=2)
"""

import os
import math
import json
import dataclasses
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import Dataset, DataLoader


# --- Data-driven normalisation parameters ---

KM_PER_DEG_LAT = 111.0  # nearly constant at all latitudes


@dataclasses.dataclass
class NormParams:
    """Data-driven normalisation parameters, computed by convert_db.py."""
    centre_lat: float
    centre_lon: float
    range_km: float          # isotropic half-width, km (99th %ile)
    max_alt_ft: float        # 99th percentile altitude
    u_mean: float            # mean of meteorological u component (kt)
    u_std: float             # std  of meteorological u component (kt)
    v_mean: float            # mean of meteorological v component (kt)
    v_std: float             # std  of meteorological v component (kt)

    @property
    def km_per_deg_lon(self) -> float:
        return 111.0 * math.cos(math.radians(self.centre_lat))

    @classmethod
    def from_meta(cls, meta_path: str) -> "NormParams":
        with open(meta_path) as f:
            meta = json.load(f)
        n = meta["normalisation"]
        if "u_mean" in n:
            return cls(
                centre_lat=n["centre_lat"],
                centre_lon=n["centre_lon"],
                range_km=n["range_km"],
                max_alt_ft=n["max_alt_ft"],
                u_mean=n["u_mean"],
                u_std=n["u_std"],
                v_mean=n["v_mean"],
                v_std=n["v_std"],
            )
        # Backward compat: old (sin, cos, speed) cache
        ws_mean = n.get("wind_speed_kt_mean", 35.0)
        ws_std = n.get("wind_speed_kt_std", 22.0)
        rms = math.sqrt(ws_mean**2 + ws_std**2)
        uv_std = rms / math.sqrt(2)
        return cls(
            centre_lat=n["centre_lat"],
            centre_lon=n["centre_lon"],
            range_km=n["range_km"],
            max_alt_ft=n["max_alt_ft"],
            u_mean=0.0, u_std=uv_std,
            v_mean=0.0, v_std=uv_std,
        )

    def to_dict(self) -> dict:
        return {
            "centre_lat": self.centre_lat,
            "centre_lon": self.centre_lon,
            "range_km": self.range_km,
            "max_alt_ft": self.max_alt_ft,
            "u_mean": self.u_mean,
            "u_std": self.u_std,
            "v_mean": self.v_mean,
            "v_std": self.v_std,
        }


def load_params(cache_dir: str) -> NormParams:
    """Load NormParams from npy_cache/meta.json."""
    return NormParams.from_meta(os.path.join(cache_dir, "meta.json"))


# --- Legacy hardcoded defaults (used only when meta.json is unavailable) ---

CENTRE_LAT = 51.071066
CENTRE_LON = -1.042441
MAX_ALT_FT = 50_000.0
WIND_SPEED_MEAN = 34.95125809377175
WIND_SPEED_STD = 21.849448514436578
MIN_AIRCRAFT = 2
LAT_RANGE_DEG = 0.63
LON_RANGE_DEG = 1.00

_RMS_SPEED = math.sqrt(WIND_SPEED_MEAN**2 + WIND_SPEED_STD**2)
_UV_STD = _RMS_SPEED / math.sqrt(2)

LEGACY_PARAMS = NormParams(
    centre_lat=CENTRE_LAT,
    centre_lon=CENTRE_LON,
    range_km=math.sqrt(
        (LAT_RANGE_DEG * KM_PER_DEG_LAT) ** 2 +
        (LON_RANGE_DEG * 111.0 * math.cos(math.radians(CENTRE_LON))) ** 2
    ),
    max_alt_ft=MAX_ALT_FT,
    u_mean=0.0, u_std=_UV_STD,
    v_mean=0.0, v_std=_UV_STD,
)


# --- Feature engineering ---

def normalise_coords(lat, lon, alt_ft, params: NormParams):
    """Normalise spatial coords to isotropic ~[-1, 1] in km.

    Converts lat/lon to local km coordinates centred at params.centre,
    then divides by the isotropic range_km. Altitude uses log scaling
    to expand the boundary-layer range.
    """
    lat_km = (lat - params.centre_lat) * KM_PER_DEG_LAT
    lon_km = (lon - params.centre_lon) * params.km_per_deg_lon
    lat_n = lat_km / params.range_km
    lon_n = lon_km / params.range_km
    alt_n = math.log(1 + alt_ft) / math.log(1 + params.max_alt_ft)
    return lat_n, lon_n, alt_n


def encode_wind(wind_dir_deg, wind_speed_kt, params: NormParams):
    """Encode wind as z-scored meteorological (u, v) components.

    u = -speed * sin(dir), v = -speed * cos(dir)  (toward-vector),
    then z-scored with training-set statistics.

    Returns (u_norm, v_norm).
    """
    rad = math.radians(wind_dir_deg)
    u = -wind_speed_kt * math.sin(rad)
    v = -wind_speed_kt * math.cos(rad)
    u_norm = (u - params.u_mean) / params.u_std
    v_norm = (v - params.v_mean) / params.v_std
    return u_norm, v_norm


def decode_wind(u_norm, v_norm, params: NormParams):
    """Inverse of encode_wind.

    Returns (wind_dir_deg [0,360), wind_speed_kt).
    """
    u = u_norm * params.u_std + params.u_mean
    v = v_norm * params.v_std + params.v_mean
    speed = math.sqrt(u**2 + v**2)
    if speed < 1e-12:
        return 0.0, 0.0
    direction = math.degrees(math.atan2(-u, -v)) % 360
    return direction, speed


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


def split_snapshot_ids(ids, times, train_frac=0.8, test_frac=0.1,
                       val_frac=0.1, seed=42, verbose=True):
    """Split snapshot IDs into train/val/test by calendar day.

    Every snapshot from a given calendar day goes to the same split; a DP
    subset-sum over day sizes hits the target fractions as closely as
    possible. Operates on plain ID/time lists so callers can use it before
    the .npy cache exists (e.g. convert_db.py fitting normalisation on
    train days only).

    The RNG consumption order (shuffle orders, subset-sum picks) must not
    change: existing seed-42 splits reproduce bit-for-bit.
    """
    total_frac = train_frac + test_frac + val_frac
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {total_frac}")

    n = len(ids)
    if len(times) != n:
        raise RuntimeError(
            "Snapshot IDs and times have mismatched lengths.")
    if n == 0:
        raise RuntimeError("No snapshots to split.")

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

    if verbose:
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


def day_grouped_split(cache_dir, train_frac=0.8, test_frac=0.1, val_frac=0.1,
                      seed=42):
    """Split cached snapshots into train/val/test by calendar day.

    Thin wrapper around split_snapshot_ids that reads the ID/time arrays
    from the .npy cache written by convert_db.py.
    """
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

    return split_snapshot_ids(
        ids, times, train_frac=train_frac, test_frac=test_frac,
        val_frac=val_frac, seed=seed, verbose=True)


# --- Dataset ---

class WindSnapshotDataset(Dataset):
    """
    Loads snapshots from the .npy cache. Each item is (x, y) where:
        x [N, 3] — (lat_norm, lon_norm, alt_norm)
        y [N, 2] — (wind_u_norm, wind_v_norm)
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
            self._x[start:end].copy())
        y = torch.from_numpy(
            self._y[start:end].copy())
        return x, y


# --- Collate ---

def _rotate_windfield(x, y, cos_a, sin_a, params=None):
    """Rigidly rotate positions and wind vectors together by angle ``a``.

    Rotates the (lat, lon) position columns counter-clockwise about the
    origin and applies the same rotation to the (u, v) wind components.

    When ``params`` is given the rotation is performed in physical (u,v)
    space then re-normalised.  Without ``params`` the rotation is applied
    directly to the first two y columns (correct only when those columns
    already represent an un-normalised 2-D vector).

    Pairwise distances and the angle between position offsets and wind
    vectors are preserved.  Altitude (x[..., 2]) is invariant.

    Modifies x and y **in place** (the caller should not reuse them).
    """
    cos_a = float(cos_a)
    sin_a = float(sin_a)

    lat, lon = x[..., 0], x[..., 1]
    new_lat = lat * cos_a + lon * sin_a
    new_lon = -lat * sin_a + lon * cos_a
    x[..., 0] = new_lat
    x[..., 1] = new_lon

    if params is not None:
        c, s = cos_a, sin_a
        u_s, u_m = params.u_std, params.u_mean
        v_s, v_m = params.v_std, params.v_mean
        a00 = c
        a01 = -s * v_s / u_s
        a10 = s * u_s / v_s
        a11 = c
        b0 = u_m * (c - 1.0) / u_s - v_m * s / u_s
        b1 = u_m * s / v_s + v_m * (c - 1.0) / v_s
        y0, y1 = y[..., 0], y[..., 1]
        new_y0 = y0 * a00 + y1 * a01 + b0
        new_y1 = y0 * a10 + y1 * a11 + b1
        y[..., 0] = new_y0
        y[..., 1] = new_y1
    else:
        u, v = y[..., 0], y[..., 1]
        new_u = u * cos_a - v * sin_a
        new_v = u * sin_a + v * cos_a
        y[..., 0] = new_u
        y[..., 1] = new_v

    return x, y


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


def _batch_rotate_positions(xs, cos_a, sin_a):
    """Rotate first two columns of batched position tensor in-place.

    xs : (B, max_n, 3),  cos_a, sin_a : (B,)
    """
    lat = xs[..., 0].clone()
    lon = xs[..., 1].clone()
    xs[..., 0] = lat * cos_a[:, None] + lon * sin_a[:, None]
    xs[..., 1] = -lat * sin_a[:, None] + lon * cos_a[:, None]


def _batch_rotate_wind(ys, cos_a, sin_a, params=None):
    """Rotate first two columns of batched wind tensor in-place.

    ys : (B, max_n, 2),  cos_a, sin_a : (B,)
    """
    if params is not None:
        u_s, u_m = params.u_std, params.u_mean
        v_s, v_m = params.v_std, params.v_mean
        c = cos_a[:, None]
        s = sin_a[:, None]
        y0 = ys[..., 0]
        y1 = ys[..., 1]
        new_y0 = (y0 * c + y1 * (-s * v_s / u_s)
                  + u_m * (c - 1.0) / u_s
                  - v_m * s / u_s)
        new_y1 = (y0 * (s * u_s / v_s) + y1 * c
                  + u_m * s / v_s
                  + v_m * (c - 1.0) / v_s)
        ys[..., 0] = new_y0
        ys[..., 1] = new_y1
    else:
        u = ys[..., 0].clone()
        v = ys[..., 1].clone()
        ys[..., 0] = u * cos_a[:, None] - v * sin_a[:, None]
        ys[..., 1] = u * sin_a[:, None] + v * cos_a[:, None]


def collate_fn(batch, augment=True, use_coupled_rotation=False,
               params=None):
    """
    Split each snapshot into context / target, with optional coupled
    rotation augmentation. C is a random subset, T is the full snapshot
    (C ⊂ T). Zero-pads to batch max with bool masks.

    Augmentation toggle:
      use_coupled_rotation : rigidly rotate positions and wind vectors
                             together by a random per-snapshot angle
                             (the only supported augmentation).
      params : NormParams, required when use_coupled_rotation=True
               so rotation is applied in physical (u,v) space.
    """
    context_xs, context_ys, target_xs, target_ys = [], [], [], []
    ctx_lens, tgt_lens = [], []

    for i, (x, y) in enumerate(batch):
        n = x.size(0)
        if n < 2:
            continue

        n_ctx = int(n * float(torch.empty(1).uniform_(0.25, 0.75)))
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

    cx, cy, tx, ty, cm, tm = pad_batch(
        context_xs, context_ys, target_xs, target_ys, ctx_lens, tgt_lens)

    if augment and use_coupled_rotation:
        B = cx.shape[0]
        angles = torch.rand(B) * (2 * math.pi)
        cos_a = angles.cos()
        sin_a = angles.sin()
        _batch_rotate_positions(cx, cos_a, sin_a)
        _batch_rotate_positions(tx, cos_a, sin_a)
        _batch_rotate_wind(cy, cos_a, sin_a, params=params)
        _batch_rotate_wind(ty, cos_a, sin_a, params=params)

    return cx, cy, tx, ty, cm, tm


def _worker_init(worker_id):
    """Reseed numpy and torch RNG per DataLoader worker."""
    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dataloader(cache_dir, batch_size=16, shuffle=True, num_workers=4):
    dataset = WindSnapshotDataset(cache_dir)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=collate_fn, num_workers=num_workers,
        worker_init_fn=_worker_init,
        persistent_workers=num_workers > 0)


def collate_fn_val(batch, context_frac=0.5, partitions=None):
    """Hold-out validation collate: target = complement of context (no leak).

    Unlike collate_fn (where context is a subset of target for the NP training
    objective), this splits each snapshot into disjoint context/target sets so
    validation loss measures generalisation to truly unseen points.

    partitions: optional {dataset_index: (ctx_idx, held_idx)} mapping. When
    given, batch items must be (index, x, y) and the frozen per-snapshot
    index split is reused every epoch, so validation metrics are comparable
    across epochs instead of being re-randomised by fresh randperm draws.
    When None, a fresh random split is drawn per pass (used by eval scripts,
    where the pass happens once under a fixed seed).
    """
    context_xs, context_ys, target_xs, target_ys = [], [], [], []
    ctx_lens, tgt_lens = [], []

    if partitions is not None:
        for (idx, x, y) in batch:
            ctx_idx, held_idx = partitions[idx]
            if ctx_idx.numel() == 0 or held_idx.numel() == 0:
                continue

            context_xs.append(x[ctx_idx])
            context_ys.append(y[ctx_idx])
            target_xs.append(x[held_idx])
            target_ys.append(y[held_idx])

            ctx_lens.append(ctx_idx.numel())
            tgt_lens.append(held_idx.numel())
    else:
        for (x, y) in batch:
            n = x.size(0)
            if n < 2:
                continue

            n_ctx = max(1, min(int(n * context_frac), n - 1))
            perm = torch.randperm(n)
            ctx_idx = perm[:n_ctx]
            held_idx = perm[n_ctx:]
            if held_idx.numel() == 0:
                continue

            context_xs.append(x[ctx_idx])
            context_ys.append(y[ctx_idx])
            target_xs.append(x[held_idx])
            target_ys.append(y[held_idx])

            ctx_lens.append(n_ctx)
            tgt_lens.append(held_idx.numel())

    if not context_xs:
        raise RuntimeError(
            "Empty batch — every item had too few observations to split.")

    return pad_batch(
        context_xs, context_ys, target_xs, target_ys, ctx_lens, tgt_lens)
