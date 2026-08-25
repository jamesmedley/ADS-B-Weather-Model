"""
gp.py — Exact multi-output Gaussian Process baseline for wind prediction.

A comparison baseline for the Attentive Neural Process (ANP) in this repo.
Entry points: scripts/train_gp.py, scripts/test_gp.py, scripts/
visualise_wind_gp.py (see scripts/ for usage).

Data representation
-------------------
The GP does NOT need the neural-network normalisation. Inputs are converted to
raw physical kilometres:

    x = [lat_km, lon_km, alt_km]

where lat/lon are projected onto a local tangent plane in km (centred at the
dataset centre, with a cos(lat) longitude correction) and altitude is linear
km. A stationary ARD kernel is translation invariant and absorbs any
per-dimension scale into its learned lengthscales, so these are directly
interpretable (e.g. "winds decorrelate over ~250 km horizontally and ~6 km
vertically").

Outputs are kept IDENTICAL to the ANP:

    y = [wind_u_norm, wind_v_norm]

Meteorological u/v components (toward-vector), z-scored with training-set
statistics. This is a Euclidean representation with no unit-circle
constraint, making the GP likelihood (Gaussian) a natural fit.

Model
-----
Three independent scalar GPs (one per output dimension) sharing a single ARD
Matérn-5/2 spatial kernel (shared lengthscales), each with its own output
amplitude and homoscedastic observation noise — mirroring the ANP's shared
representation + per-dimension predictive Gaussian. The prior mean is zero:
all outputs are centred / normalised in the cache.

Snapshots have at most ~16 context points, so the GP is exact (full Cholesky).
A per-snapshot fit just stores the context and factorises K + noise² I.

Hyperparameters (lengthscales, amplitudes, noises) are fit GLOBALLY by
maximising the sum of per-snapshot marginal log-likelihoods over a subsample
of TRAINING snapshots (the GP analogue of the ANP's shared weights), then held
fixed at predict time.

Uncertainty decomposition
-------------------------
For output j at query x* given context X:

    var_f(x*) = k(x*,x*) - k*^T (K + sigma_j^2 I)^-1 k*      (epistemic)
    var_y(x*) = var_f(x*) + sigma_j^2                         (aleatoric)

predict() reports both components converted to physical units via the same
delta-method used by scripts/test.py.
"""

import json
import math

import numpy as np
import torch as t

from wind_map.preprocess import (
    KM_PER_DEG_LAT, NormParams, encode_wind,
)

FT_PER_KM = 0.0003048


def norm_params_from_dict(d):
    """Build a NormParams from a dict, accepting either key naming convention.

    Supports both the new u/v normalisation and legacy (sin, cos, speed)
    conventions for backward compatibility with old checkpoints.
    """
    def _get(*keys):
        for k in keys:
            if k in d:
                return d[k]
        raise KeyError(f"missing key (tried {list(keys)}) in {sorted(d)}")

    if "u_mean" in d:
        return NormParams(
            centre_lat=_get("centre_lat"),
            centre_lon=_get("centre_lon"),
            range_km=_get("range_km"),
            max_alt_ft=_get("max_alt_ft"),
            u_mean=_get("u_mean"),
            u_std=_get("u_std"),
            v_mean=_get("v_mean"),
            v_std=_get("v_std"),
        )

    # Backward compat: old (sin, cos, speed) checkpoints
    ws_mean = _get("wind_speed_kt_mean", "wind_speed_mean_kt")
    ws_std = _get("wind_speed_kt_std", "wind_speed_std_kt")
    rms = math.sqrt(ws_mean**2 + ws_std**2)
    uv_std = rms / math.sqrt(2)
    return NormParams(
        centre_lat=_get("centre_lat"),
        centre_lon=_get("centre_lon"),
        range_km=_get("range_km"),
        max_alt_ft=_get("max_alt_ft"),
        u_mean=0.0, u_std=uv_std,
        v_mean=0.0, v_std=uv_std,
    )


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------

def _obs_to_local_km(lat, lon, alt_ft, params, altitude_km=True):
    """Project a lat/lon/alt observation to local km."""
    lat_km = (lat - params.centre_lat) * KM_PER_DEG_LAT
    lon_km = (lon - params.centre_lon) * params.km_per_deg_lon
    alt = alt_ft * (FT_PER_KM if altitude_km else 1.0)
    return lat_km, lon_km, alt


def observations_to_gp_x(observations, params, altitude_km=True):
    """List of observation dicts -> [N, 3] local-km input array."""
    xs = np.empty((len(observations), 3), dtype=np.float64)
    for i, obs in enumerate(observations):
        xs[i] = _obs_to_local_km(
            obs["lat"], obs["lon"], obs["alt_ft"], params,
            altitude_km=altitude_km)
    return xs


def cache_x_to_gp_x(x, params, altitude_km=True):
    """Cache-normalised x [N, 3] -> local-km input array.

    Inverse of the normalisation in wind_map.preprocess.normalise_coords:
        lat_km = lat_n * range_km
        lon_km = lon_n * range_km
        alt_km = (exp(alt_n * log(1 + max_alt_ft)) - 1) * FT_PER_KM
    """
    x = np.asarray(x, dtype=np.float64)
    alt_ft = np.exp(x[:, 2] * math.log(1.0 + params.max_alt_ft)) - 1.0
    alt = alt_ft * (FT_PER_KM if altitude_km else 1.0)
    return np.column_stack([
        x[:, 0] * params.range_km,
        x[:, 1] * params.range_km,
        alt,
    ])


# ---------------------------------------------------------------------------
# Kernel & exact GP
# ---------------------------------------------------------------------------

def _ard_squared_dist(x1, x2, inv_l2):
    """x1 [n1, D], x2 [n2, D], inv_l2 [D] -> ARD squared distance [n1, n2]."""
    diff = x1[:, None, :] - x2[None, :, :]
    return np.sum(diff * diff * inv_l2, axis=-1)


def _matern52_corr(d2):
    """Matérn-5/2 correlation function of squared distance d2."""
    r = np.sqrt(np.maximum(d2, 0.0))
    return ((1.0 + math.sqrt(5.0) * r + (5.0 / 3.0) * d2)
            * np.exp(-math.sqrt(5.0) * r))


class GaussianProcessRegressor:
    """Exact multi-output GP with a shared ARD Matérn-5/2 kernel.

    Two independent outputs (u, v wind components) share the spatial kernel
    (lengthscales) but each has its own amplitude and observation noise.
    Zero prior mean.
    """

    def __init__(self, lengthscales, amplitudes, noises, jitter=1e-6):
        self.lengthscales = np.asarray(lengthscales, dtype=np.float64)
        self.amplitudes = np.asarray(amplitudes, dtype=np.float64)
        self.noises = np.asarray(noises, dtype=np.float64)
        self.inv_l2 = 1.0 / self.lengthscales ** 2
        self.jitter = jitter
        self._fitted = False

    def fit(self, X, y):
        """Fit on context (X [m, 3], y [m, 2]). Cost is O(2 * m^3)."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        m = X.shape[0]
        d2 = _ard_squared_dist(X, X, self.inv_l2)
        k0 = _matern52_corr(d2)
        self._L = []
        for j in range(2):
            K = self.amplitudes[j] ** 2 * k0
            K = K + (self.noises[j] ** 2 + self.jitter) * np.eye(m)
            self._L.append(np.linalg.cholesky(K))
        self.X = X
        self.y = y
        self._fitted = True
        return self

    def predict(self, X_star):
        """Return (mean, total_var, latent_var), each [N, 2]."""
        if not self._fitted:
            raise RuntimeError(
                "GaussianProcessRegressor.fit() must be called first.")
        X_star = np.asarray(X_star, dtype=np.float64)
        n = X_star.shape[0]
        d2 = _ard_squared_dist(X_star, self.X, self.inv_l2)
        k0 = _matern52_corr(d2)
        mean = np.empty((n, 2))
        var_total = np.empty((n, 2))
        var_latent = np.empty((n, 2))
        for j in range(2):
            L = self._L[j]
            # K^{-1} = (L L^T)^{-1} = L^{-T} L^{-1}  ->  solve L, then L.T
            z = np.linalg.solve(L, self.y[:, j])
            w = np.linalg.solve(L.T, z)
            ks = self.amplitudes[j] ** 2 * k0
            mean[:, j] = ks @ w
            v = np.linalg.solve(L.T, np.linalg.solve(L, ks.T))
            var_latent[:, j] = np.maximum(
                self.amplitudes[j] ** 2 - np.sum(ks * v.T, axis=1), 0.0)
            var_total[:, j] = var_latent[:, j] + self.noises[j] ** 2
        return mean, var_total, var_latent


# ---------------------------------------------------------------------------
# Global hyperparameter fitting (torch, batched over snapshots)
# ---------------------------------------------------------------------------

def _pack_snapshots(snapshots, max_m):
    """snapshots: list of (x [m, 3], y [m, 2]) float64 arrays
    -> padded tensors."""
    S = len(snapshots)
    X = t.zeros(S, max_m, 3, dtype=t.float64)
    Y = t.zeros(S, max_m, 2, dtype=t.float64)
    mask = t.zeros(S, max_m, dtype=t.bool)
    for i, (x, y) in enumerate(snapshots):
        m = x.shape[0]
        X[i, :m] = t.as_tensor(x, dtype=t.float64)
        Y[i, :m] = t.as_tensor(y, dtype=t.float64)
        mask[i, :m] = True
    return X, Y, mask


def _snapshot_ll(X, Y, mask, log_lengthscales, log_amplitudes, log_noises,
                 jitter=1e-6):
    """Per-snapshot log-likelihood [S] (nats), summed over the 2 outputs.

    Padded rows are neutralised (identity block, zero y) so they contribute
    only parameter-independent constants.
    """
    S, M, _ = X.shape
    inv_l2 = t.exp(-2.0 * log_lengthscales)
    diff = X[:, :, None, :] - X[:, None, :, :]
    d2 = (diff * diff * inv_l2).sum(dim=-1)
    # Floor d2 (diagonal is exactly 0) so autograd never hits 0 * inf through
    # sqrt: the Matérn kernel's analytic gradient at r=0 is 0, but autograd
    # would compute d(r)/d(d2)=inf there and produce NaN.
    r = d2.clamp(min=1e-12).sqrt()
    k0 = ((1.0 + math.sqrt(5.0) * r + (5.0 / 3.0) * d2)
          * t.exp(-math.sqrt(5.0) * r))

    pair_mask = (mask[:, :, None] & mask[:, None, :]).to(t.float64)
    eye = t.eye(M, dtype=t.float64)
    noise2 = t.exp(2.0 * log_noises)
    amp2 = t.exp(2.0 * log_amplitudes)

    ll = t.zeros(S, dtype=t.float64)
    for j in range(2):
        K = amp2[j] * k0
        K = K * pair_mask + eye * (1.0 - pair_mask)
        K = K + noise2[j] * t.diag_embed(mask.to(t.float64))
        K = K + jitter * eye
        L = t.linalg.cholesky(K)
        logdet = 2.0 * L.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        Yj = Y[..., j].unsqueeze(-1)          # [S, M, 1]
        # K^{-1} = L^{-T} L^{-1};  quad = Yj^T K^{-1} Yj = ||L^{-1} Yj||^2
        tmp = t.linalg.solve_triangular(L, Yj, upper=False)
        quad = (tmp * tmp).sum(dim=-1).sum(dim=-1)   # [S]
        ll = ll - 0.5 * (quad + logdet + M * math.log(2.0 * math.pi))
    return ll


def _marginal_log_likelihood(X, Y, mask, log_lengthscales, log_amplitudes,
                             log_noises, jitter=1e-6):
    return _snapshot_ll(X, Y, mask, log_lengthscales, log_amplitudes,
                        log_noises, jitter=jitter).sum()


def fit_hyperparameters(snapshots, n_steps=800, lr=0.05, seed=0,
                        init_lengthscales=(100.0, 100.0, 3.0),
                        init_amplitudes=(0.6, 0.6),
                        init_noises=(0.15, 0.15),
                        jitter=1e-4, noise_floor=1e-2, verbose=True):
    """Global MLE of kernel hyperparameters over training snapshots.

    Returns (lengthscales, amplitudes, noises, mll_per_point) where
    mll_per_point is the corrected average marginal log-likelihood in nats.
    """
    t.manual_seed(seed)
    max_m = max(x.shape[0] for x, _ in snapshots)
    X, Y, mask = _pack_snapshots(snapshots, max_m)

    log_ls = t.nn.Parameter(t.tensor([math.log(v) for v in init_lengthscales]))
    log_amp = t.nn.Parameter(t.tensor([math.log(v) for v in init_amplitudes]))
    log_noi = t.nn.Parameter(t.tensor([math.log(v) for v in init_noises]))
    opt = t.optim.Adam([log_ls, log_amp, log_noi], lr=lr)
    log_noise_floor = math.log(noise_floor)

    best = None
    for step in range(1, n_steps + 1):
        opt.zero_grad()
        ll = _marginal_log_likelihood(X, Y, mask, log_ls, log_amp, log_noi,
                                      jitter=jitter)
        (-ll).backward()
        opt.step()
        # Keep the likelihood regularised by a noise floor so Adam cannot
        # collapse noise to ~0 (which makes near-duplicate points singular).
        with t.no_grad():
            log_noi.clamp_(min=log_noise_floor)
        if best is None or ll.item() > best[0]:
            best = (ll.item(), log_ls.detach().clone(),
                    log_amp.detach().clone(), log_noi.detach().clone())
        if verbose and step % 100 == 0:
            n_pts = int(mask.sum().item())
            print(f"  step {step:4d}/{n_steps}  "
                  f"mll={ll.item() / n_pts:+.4f} nats/point")

    _, log_ls, log_amp, log_noi = best

    with t.no_grad():
        ll_s = _snapshot_ll(X, Y, mask, log_ls, log_amp, log_noi,
                            jitter=jitter)
    counts = mask.sum(dim=-1).to(t.float64)
    # Remove the per-output (M - m_i) constants from padding:
    #   + 0.5 (M-m_i) log(2*pi)  (from the M*log(2*pi) term)
    #   + 0.5 (M-m_i) log(1+jitter)  (from the padded identity block)
    ll_true = ll_s + 1.5 * (max_m - counts) * (
        math.log(2.0 * math.pi) + math.log1p(jitter))
    mll_per_point = float((ll_true.sum() / counts.sum()).item())

    lengthscales = log_ls.exp().numpy()
    amplitudes = log_amp.exp().numpy()
    noises = log_noi.exp().numpy()
    return lengthscales, amplitudes, noises, mll_per_point


# ---------------------------------------------------------------------------
# Prediction API (drop-in analogue of wind_map.infer.WindPredictor)
# ---------------------------------------------------------------------------

class GaussianProcessPredictor:
    """GP baseline predictor with the same interface as WindPredictor.

    predict(context_observations, query_points, n_samples=...) returns the
    same dict keys as wind_map.infer.WindPredictor.predict. n_samples is
    accepted for API compatibility; the GP prediction is deterministic.
    """

    def __init__(self, params_path, device=None, params: NormParams = None):
        self.device = device
        with open(params_path) as f:
            hp = json.load(f)
        self.lengthscales = np.asarray(hp["lengthscales"], dtype=np.float64)
        self.amplitudes = np.asarray(
            hp.get("amplitudes", hp.get("output_scales")), dtype=np.float64)
        self.noises = np.asarray(hp["noises"], dtype=np.float64)
        self.input_space = hp.get("input_space", "km")
        if params is not None:
            self.params = params
        else:
            self.params = norm_params_from_dict(hp.get("norm_params", {}))
        print(f"Loaded GP params from {params_path} "
              f"(input_space={self.input_space})")

    def _altitude_km(self):
        return self.input_space != "raw_km_ft"

    def _encode_obs(self, observations):
        xs = observations_to_gp_x(
            observations, self.params, altitude_km=self._altitude_km())
        ys = np.empty((len(observations), 2), dtype=np.float64)
        for i, obs in enumerate(observations):
            u_n, v_n = encode_wind(
                obs["wind_dir"], obs["wind_speed"], self.params)
            ys[i] = (u_n, v_n)
        return xs, ys

    def _encode_queries(self, queries):
        return observations_to_gp_x(
            queries, self.params, altitude_km=self._altitude_km())

    def predict(self, context_observations, query_points, n_samples=50):
        x_c, y_c = self._encode_obs(context_observations)
        x_q = self._encode_queries(query_points)

        reg = GaussianProcessRegressor(
            self.lengthscales, self.amplitudes, self.noises)
        reg.fit(x_c, y_c)
        mean, var_total, var_latent = reg.predict(x_q)
        noise2 = self.noises ** 2

        # Decode u/v mean to physical wind
        u_m = mean[:, 0] * self.params.u_std + self.params.u_mean
        v_m = mean[:, 1] * self.params.v_std + self.params.v_mean
        speed = np.sqrt(u_m**2 + v_m**2)
        dirs = np.degrees(np.arctan2(-u_m, -v_m)) % 360

        # Delta-method uncertainty propagation
        speed_safe = np.maximum(speed, 1e-6)
        sigma_u = np.sqrt(var_total[:, 0]) * self.params.u_std
        sigma_v = np.sqrt(var_total[:, 1]) * self.params.v_std

        def _spd_std(su, sv):
            return np.sqrt((u_m / speed_safe)**2 * su**2
                           + (v_m / speed_safe)**2 * sv**2)

        def _dir_std(su, sv):
            return np.degrees(np.sqrt(
                (v_m / speed_safe**2)**2 * su**2
                + (u_m / speed_safe**2)**2 * sv**2))

        n = mean.shape[0]
        ale_u = noise2[0] * np.ones(n) * self.params.u_std**2
        ale_v = noise2[1] * np.ones(n) * self.params.v_std**2

        return {
            "wind_dir_deg": dirs,
            "wind_speed_kt": speed,
            "wind_dir_std": _dir_std(sigma_u, sigma_v),
            "wind_speed_std": _spd_std(sigma_u, sigma_v),
            "epistemic_dir_std": _dir_std(
                np.sqrt(var_latent[:, 0]) * self.params.u_std,
                np.sqrt(var_latent[:, 1]) * self.params.v_std),
            "epistemic_speed_std": _spd_std(
                np.sqrt(var_latent[:, 0]) * self.params.u_std,
                np.sqrt(var_latent[:, 1]) * self.params.v_std),
            "aleatoric_dir_std": _dir_std(np.sqrt(ale_u), np.sqrt(ale_v)),
            "aleatoric_speed_std": _spd_std(np.sqrt(ale_u), np.sqrt(ale_v)),
        }
