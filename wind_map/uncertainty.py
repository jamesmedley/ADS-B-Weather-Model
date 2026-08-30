"""
uncertainty.py — Single source of truth for deriving uncertainty from the
model's 2D wind-vector output.

The model's predictive output is a 2D wind vector (u, v), represented per
component as a Gaussian in normalised u/v space: ``mu = [u_n, v_n]`` with
independent ``sigma = [sigma_u_n, sigma_v_n]``. All uncertainty is therefore
computed in the Euclidean u/v representation first, and speed / direction
uncertainty is *derived* from that vector uncertainty with a shared delta
method. Nothing in the rest of the codebase should re-derive the delta method
or recombine polar uncertainty by hand.

Three primitives form the public surface:

1. ``per_component_total_std`` — law-of-total-variance per u/v component
   (epistemic + aleatoric) across latent draws, in physical kt.
2. ``uv_to_speed_dir_std`` — convert per-component u/v std to wind-speed and
   wind-direction std via the delta method (linearisation about the mean
   vector).
3. ``combined_vector_std`` — rotation-invariant scalar for the whole 2D
   covariance: sqrt(trace(Sigma)) = sqrt(sigma_u^2 + sigma_v^2).

Plus ``epistemic_aleatoric_uv_std`` which splits the law-of-total-variance into
its two sources (used only where multiple latent draws are available).

All functions work on NumPy arrays and take ``NormParams`` only where physical
conversion is required.
"""

import numpy as np

from wind_map.preprocess import NormParams


# ---------------------------------------------------------------------------
# Delta method: u/v std -> speed / direction std
# ---------------------------------------------------------------------------

def uv_to_speed_dir_std(u_mean, v_mean, sigma_u, sigma_v):
    """Derive speed and direction std from per-component u/v std.

    Linearises the nonlinear polar transforms
        speed = sqrt(u^2 + v^2)
        dir   = atan2(-u, -v)
    about the mean vector (u_mean, v_mean) -- the standard delta method. The
    predictable covariance is assumed diagonal in (u, v) (independent sigma),
    which matches the decoder's output.

    Args:
        u_mean, v_mean: physical mean wind components (kt). Same shape.
        sigma_u, sigma_v: physical per-component std (kt). Same shape as means.

    Returns:
        (sigma_speed_kt, sigma_dir_deg) both same shape as means.
    """
    u_mean = np.asarray(u_mean, dtype=np.float64)
    v_mean = np.asarray(v_mean, dtype=np.float64)
    sigma_u = np.asarray(sigma_u, dtype=np.float64)
    sigma_v = np.asarray(sigma_v, dtype=np.float64)

    speed = np.sqrt(u_mean**2 + v_mean**2)
    speed_safe = np.maximum(speed, 1e-6)

    # dp/d(u,v): speed = sqrt(u^2+v^2) -> dp/du = u/s, dp/dv = v/s
    sigma_speed = np.sqrt(
        (u_mean / speed_safe)**2 * sigma_u**2
        + (v_mean / speed_safe)**2 * sigma_v**2)

    # d(theta)/d(u,v) for theta = atan2(-u,-v): dtheta/du = -v/s^2,
    # dtheta/dv = u/s^2 (up to sign; variance is sign-agnostic).
    sigma_dir = np.degrees(np.sqrt(
        (v_mean / speed_safe**2)**2 * sigma_u**2
        + (u_mean / speed_safe**2)**2 * sigma_v**2))

    return sigma_speed, sigma_dir


# ---------------------------------------------------------------------------
# Vector recombinations
# ---------------------------------------------------------------------------

def combined_vector_std(sigma_u, sigma_v):
    """Total 2D vector std = sqrt(sigma_u^2 + sigma_v^2).

    This is sqrt(trace(Sigma)) of the predictive covariance and is invariant
    under rotation of the (u, v) axes. It is the canonical single scalar that
    mixes speed and direction uncertainty consistently, and it never blows up
    at low wind speed the way polar direction std does.
    """
    sigma_u = np.asarray(sigma_u, dtype=np.float64)
    sigma_v = np.asarray(sigma_v, dtype=np.float64)
    return np.sqrt(np.maximum(sigma_u**2 + sigma_v**2, 0.0))


# ---------------------------------------------------------------------------
# Law of total variance over latent draws
# ---------------------------------------------------------------------------

def _as_float(values, name):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(
            f"{name} must be [n_z, N, 2], got shape {arr.shape}")
    if arr.shape[-1] != 2:
        raise ValueError(
            f"{name} must have trailing dim 2 (u, v), got shape {arr.shape}")
    return arr


def per_component_total_std(mu_stack, sigma_stack, params: NormParams):
    """Per-component total std in physical kt via the law of total variance.

        var_u = Var_z(mu_u)     + E_z[sigma_u^2]     (epistemic + aleatoric)
        var_v = Var_z(mu_v)     + E_z[sigma_v^2]

    where mu_stack[i] is the decoder mean for latent draw i and
    sigma_stack[i] the decoder std. Epistemic variance uses sample variance
    (ddof=1) across draws, matching the codebase convention.

    Args:
        mu_stack, sigma_stack: [n_z, N, 2] normalised (u, v) arrays.
        params: NormParams for scaling to physical kt.

    Returns:
        (sigma_u, sigma_v) both [N] physical std in kt.
    """
    mu_stack = _as_float(mu_stack, "mu_stack")
    sigma_stack = _as_float(sigma_stack, "sigma_stack")

    n_z = mu_stack.shape[0]
    ddof = 1 if n_z > 1 else 0

    var_u = mu_stack[..., 0].var(axis=0, ddof=ddof
                                 ) + (sigma_stack[..., 0]**2).mean(axis=0)
    var_v = mu_stack[..., 1].var(axis=0, ddof=ddof
                                 ) + (sigma_stack[..., 1]**2).mean(axis=0)

    sigma_u = np.sqrt(np.maximum(var_u, 0.0)) * params.u_std
    sigma_v = np.sqrt(np.maximum(var_v, 0.0)) * params.v_std
    return sigma_u, sigma_v


def epistemic_aleatoric_uv_std(mu_stack, sigma_stack, params: NormParams):
    """Split per-component u/v uncertainty into epistemic and aleatoric std.

    Epistemic = std of the decoded means across latent draws (spread of z).
    Aleatoric = sqrt of the mean decoder variance (irreducible noise).

    Both returned in physical kt. Requires multiple latent draws (n_z >= 2)
    for a meaningful epistemic term.

    Args:
        mu_stack, sigma_stack: [n_z, N, 2] normalised (u, v) arrays.
        params: NormParams.

    Returns:
        (sigma_u_epi, sigma_v_epi, sigma_u_ale, sigma_v_ale) each [N] in kt.
    """
    mu_stack = _as_float(mu_stack, "mu_stack")
    sigma_stack = _as_float(sigma_stack, "sigma_stack")

    n_z = mu_stack.shape[0]
    ddof = 1 if n_z > 1 else 0

    var_u_epi = mu_stack[..., 0].var(axis=0, ddof=ddof)
    var_v_epi = mu_stack[..., 1].var(axis=0, ddof=ddof)
    var_u_ale = (sigma_stack[..., 0]**2).mean(axis=0)
    var_v_ale = (sigma_stack[..., 1]**2).mean(axis=0)

    sigma_u_epi = np.sqrt(np.maximum(var_u_epi, 0.0)) * params.u_std
    sigma_v_epi = np.sqrt(np.maximum(var_v_epi, 0.0)) * params.v_std
    sigma_u_ale = np.sqrt(np.maximum(var_u_ale, 0.0)) * params.u_std
    sigma_v_ale = np.sqrt(np.maximum(var_v_ale, 0.0)) * params.v_std
    return sigma_u_epi, sigma_v_epi, sigma_u_ale, sigma_v_ale
