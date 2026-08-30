"""
Unit tests for wind_map.uncertainty — the single source of truth for deriving
speed/direction uncertainty and combined vector uncertainty from the model's
2D u/v wind-vector output.
"""

import math

import numpy as np
import pytest

from wind_map.uncertainty import (
    combined_vector_std,
    uv_to_speed_dir_std,
    per_component_total_std,
    epistemic_aleatoric_uv_std,
)


# ---------------------------------------------------------------------------
# combined_vector_std
# ---------------------------------------------------------------------------

def test_combined_vector_std_sqrt_sum_of_squares():
    su = np.array([3.0, 0.0, 5.0])
    sv = np.array([4.0, 2.0, 0.0])
    out = combined_vector_std(su, sv)
    assert np.allclose(out, np.array([5.0, 2.0, 5.0]))
    assert (out >= 0).all()
    assert np.isfinite(out).all()


def test_combined_vector_std_zero():
    assert combined_vector_std(0.0, 0.0) == 0.0


def test_combined_vector_std_rotation_invariant_scale():
    # The magnitude of a rotated vector is unchanged.
    rng = np.random.default_rng(0)
    su = rng.uniform(0.1, 10, 200)
    sv = rng.uniform(0.1, 10, 200)
    theta = 0.7
    ru = su * math.cos(theta) - sv * math.sin(theta)
    rv = su * math.sin(theta) + sv * math.cos(theta)
    assert np.allclose(combined_vector_std(su, sv),
                       combined_vector_std(ru, rv))


# ---------------------------------------------------------------------------
# uv_to_speed_dir_std
# ---------------------------------------------------------------------------

def test_delta_method_along_u():
    # Wind purely along +u: speed std == sigma_u, direction undefined -> 0
    # because d(dir)/dv is zero when v == 0 (the along-axis Jacobian).
    sigma_speed, sigma_dir = uv_to_speed_dir_std(20.0, 0.0, 5.0, 2.0)
    assert np.isclose(sigma_speed, 5.0)
    # delta_dir = degrees(sqrt( (v/s^2)^2 su^2 + (u/s^2)^2 sv^2 )) with v=0
    #          = degrees(sqrt( (u/s^2)^2 sv^2 )) = degrees(sv/s)
    assert np.isclose(sigma_dir, math.degrees(2.0 / 20.0))


def test_delta_method_zero_sigma():
    sigma_speed, sigma_dir = uv_to_speed_dir_std(10.0, 10.0, 0.0, 0.0)
    assert sigma_speed == 0.0
    assert sigma_dir == 0.0


def test_delta_method_matches_existing_gold_formula():
    # Mirror the hand-written formula previously duplicated across
    # train/test/gp.
    rng = np.random.default_rng(1)
    u = rng.uniform(-40, 40, 50)
    v = rng.uniform(-40, 40, 50)
    su = rng.uniform(0.5, 10, 50)
    sv = rng.uniform(0.5, 10, 50)
    s = np.maximum(np.sqrt(u**2 + v**2), 1e-6)

    exp_speed = np.sqrt((u / s)**2 * su**2 + (v / s)**2 * sv**2)
    exp_dir = np.degrees(np.sqrt((v / s**2)**2 * su**2
                                 + (u / s**2)**2 * sv**2))

    got_speed, got_dir = uv_to_speed_dir_std(u, v, su, sv)
    assert np.allclose(got_speed, exp_speed)
    assert np.allclose(got_dir, exp_dir)


# ---------------------------------------------------------------------------
# per_component_total_std
# ---------------------------------------------------------------------------

def test_per_component_total_std_zero_spread(params):
    # Identical means across draws -> epistemic component zero; total is just
    # sqrt(E[sigma^2]).
    n_z, n_pts = 4, 5
    mu_stack = np.zeros((n_z, n_pts, 2))
    mu_stack[:, :, 0] = 0.6
    mu_stack[:, :, 1] = 0.8
    sigma_stack = np.full((n_z, n_pts, 2), 0.2)

    su, sv = per_component_total_std(mu_stack, sigma_stack, params)
    assert np.allclose(su, 0.2 * params.u_std, atol=1e-12)
    assert np.allclose(sv, 0.2 * params.v_std, atol=1e-12)


def test_per_component_total_std_matches_manual(params):
    n_z, n_pts = 5, 3
    rng = np.random.default_rng(7)
    mu_u = rng.normal(0.2, 0.5, (n_z, n_pts))
    mu_v = rng.normal(-0.1, 0.4, (n_z, n_pts))
    sig_u = rng.uniform(0.1, 1.0, (n_z, n_pts))
    sig_v = rng.uniform(0.1, 1.0, (n_z, n_pts))

    mu_stack = np.stack([mu_u, mu_v], axis=-1)
    sigma_stack = np.stack([sig_u, sig_v], axis=-1)

    su, sv = per_component_total_std(mu_stack, sigma_stack, params)

    exp_su = np.sqrt(mu_u.var(axis=0, ddof=1) + (sig_u**2).mean(axis=0)
                     ) * params.u_std
    exp_sv = np.sqrt(mu_v.var(axis=0, ddof=1) + (sig_v**2).mean(axis=0)
                     ) * params.v_std
    assert np.allclose(su, exp_su, atol=1e-12)
    assert np.allclose(sv, exp_sv, atol=1e-12)


def test_per_component_total_std_single_draw_reduces_to_aleatoric(params):
    # n_z == 1 -> var(ddof=0) of a single sample is 0 -> total = sigma.
    mu_stack = np.zeros((1, 4, 2))
    sigma_stack = np.full((1, 4, 2), 0.5)
    su, sv = per_component_total_std(mu_stack, sigma_stack, params)
    assert np.allclose(su, 0.5 * params.u_std)
    assert np.allclose(sv, 0.5 * params.v_std)


def test_per_component_total_std_rejects_bad_shape(params):
    with pytest.raises(ValueError):
        per_component_total_std(np.zeros((5, 4)), np.zeros((5, 4)), params)
    with pytest.raises(ValueError):
        per_component_total_std(np.zeros((5, 4, 3)), np.zeros((5, 4, 3)),
                                params)


# ---------------------------------------------------------------------------
# epistemic_aleatoric_uv_std
# ---------------------------------------------------------------------------

def test_epistemic_aleatoric_split(params):
    n_z, n_pts = 4, 3
    mu_u = np.zeros((n_z, n_pts))
    mu_v = np.zeros((n_z, n_pts))
    # Means vary across draws -> epistemic > 0.
    mu_u[0] = -1.0
    mu_u[1] = 1.0
    sig_u = np.full((n_z, n_pts), 0.3)
    sig_v = np.full((n_z, n_pts), 0.5)

    mu_stack = np.stack([mu_u, mu_v], axis=-1)
    sigma_stack = np.stack([sig_u, sig_v], axis=-1)

    epi_u, epi_v, ale_u, ale_v = epistemic_aleatoric_uv_std(
        mu_stack, sigma_stack, params)

    # Aleatoric: sqrt(mean(sigma^2))
    assert np.allclose(ale_u, 0.3 * params.u_std, atol=1e-12)
    assert np.allclose(ale_v, 0.5 * params.v_std, atol=1e-12)
    # Epistemic: std of means across draws (sample std ddof=1)
    assert np.allclose(epi_u, mu_u.std(axis=0, ddof=1) * params.u_std)
    # v means are constant -> zero epistemic in v
    assert np.allclose(epi_v, 0.0, atol=1e-12)


def test_epistemic_and_aleatoric_recombine_to_total(params):
    # law of total variance: var_epi + var_ale == var_total (per component)
    rng = np.random.default_rng(11)
    n_z, n_pts = 6, 4
    mu_u = rng.normal(0, 0.4, (n_z, n_pts))
    mu_v = rng.normal(0, 0.3, (n_z, n_pts))
    sig_u = rng.uniform(0.1, 0.8, (n_z, n_pts))
    sig_v = rng.uniform(0.1, 0.8, (n_z, n_pts))

    mu_stack = np.stack([mu_u, mu_v], axis=-1)
    sigma_stack = np.stack([sig_u, sig_v], axis=-1)

    su_tot, sv_tot = per_component_total_std(mu_stack, sigma_stack, params)
    epi_u, epi_v, ale_u, ale_v = epistemic_aleatoric_uv_std(
        mu_stack, sigma_stack, params)

    assert np.allclose(su_tot**2, epi_u**2 + ale_u**2, atol=1e-12)
    assert np.allclose(sv_tot**2, epi_v**2 + ale_v**2, atol=1e-12)
