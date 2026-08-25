"""
infer.py — Predict wind vectors at arbitrary query points from
a trained Wind ANP.

Example:
    predictor = WindPredictor('checkpoint/best_model.pth.tar')
    result = predictor.predict(context, queries, n_samples=50)
    # result['wind_dir_deg'], result['wind_speed_kt']
    # result['wind_dir_std'], result['wind_speed_std']
"""

import torch
import numpy as np
from wind_map.network import LatentModel
from wind_map.gp import norm_params_from_dict
from wind_map.preprocess import (
    normalise_coords, encode_wind, NormParams, LEGACY_PARAMS,
)
from wind_map.utils import circular_std


# ---------------------------------------------------------------------------
# Shared helpers used by infer, test, and visualise_uncertainty
# ---------------------------------------------------------------------------

def load_model_checkpoint(checkpoint_path, device, num_hidden=None,
                          num_latents=None,
                          latent_layers=None, deterministic_layers=None,
                          dropout=None,
                          num_decoder_layers=None):
    """Load a LatentModel from a checkpoint file.

    Returns (model, ckpt_dict).
    """
    ckpt = torch.load(checkpoint_path, map_location=device,
                      weights_only=False)
    hp = ckpt.get('hparams', {})
    num_hidden = num_hidden or hp.get('num_hidden')
    num_latents = num_latents or hp.get('num_latents')
    latent_layers = (
        latent_layers or hp.get('latent_layers', hp.get('layers', 4)))
    deterministic_layers = (
        deterministic_layers
        or hp.get('deterministic_layers', hp.get('layers', 4)))
    dropout = dropout if dropout is not None else hp.get('dropout', 0.0)

    if num_hidden is None:
        raise ValueError(
            "No 'hparams' in checkpoint — pass num_hidden explicitly.")

    num_decoder_layers = (
        num_decoder_layers if num_decoder_layers is not None
        else hp.get('num_decoder_layers', 3))
    use_dist_bias = hp.get('use_dist_bias', False)
    model = LatentModel(
        num_hidden, num_latents=num_latents, x_dim=3,
        latent_layers=latent_layers,
        deterministic_layers=deterministic_layers,
        dropout=dropout,
        use_dist_bias=use_dist_bias,
        num_decoder_layers=num_decoder_layers,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt


def observations_to_tensors(observations, device, params: NormParams):
    """Convert observation dicts to (x, y) tensors with batch dimension."""
    xs, ys = [], []
    for obs in observations:
        lat_n, lon_n, alt_n = normalise_coords(
            obs['lat'], obs['lon'], obs['alt_ft'], params
        )
        u_n, v_n = encode_wind(
            obs['wind_dir'], obs['wind_speed'], params
        )
        xs.append([lat_n, lon_n, alt_n])
        ys.append([u_n, v_n])
    x = torch.FloatTensor(xs).unsqueeze(0).to(device)
    y = torch.FloatTensor(ys).unsqueeze(0).to(device)
    return x, y


def queries_to_tensor(queries, device, params: NormParams):
    """Convert query dicts to target_x tensor with batch dimension."""
    xs = []
    for q in queries:
        lat_n, lon_n, alt_n = normalise_coords(
            q['lat'], q['lon'], q['alt_ft'], params
        )
        xs.append([lat_n, lon_n, alt_n])
    return torch.FloatTensor(xs).unsqueeze(0).to(device)


def _uv_to_physical(u_norm, v_norm, params: NormParams):
    """Decode normalised (u, v) to physical wind (direction, speed).

    Both inputs are numpy arrays. Returns (direction_deg, speed_kt).
    """
    u = u_norm * params.u_std + params.u_mean
    v = v_norm * params.v_std + params.v_mean
    speed = np.sqrt(u**2 + v**2)
    direction = np.degrees(np.arctan2(-u, -v)) % 360
    return direction, speed


def compute_uncertainty_components(mu_stack, sigma_stack, params: NormParams,
                                   mc_samples=10, seed=0):
    """Given mu_stack [n_z, N, 2] and sigma_stack [n_z, N, 2]
    from multiple latent draws, return epistemic and aleatoric uncertainty.

    Aleatoric direction std is computed via MC integration of the
    predictive distribution. Aleatoric speed std uses the exact Jacobian.
    Epistemic std is the standard deviation of z-draw means.

    Returns dict with keys:
        epistemic_dir_std, epistemic_speed_std,
        aleatoric_dir_std, aleatoric_speed_std
    """
    n_z, n_pts = mu_stack.shape[:2]
    u_mu = mu_stack[..., 0]
    v_mu = mu_stack[..., 1]

    # Decode each z-draw's mu to physical wind
    sample_dirs = np.empty_like(u_mu)
    sample_speeds = np.empty_like(u_mu)
    for z in range(n_z):
        d, s = _uv_to_physical(u_mu[z], v_mu[z], params)
        sample_dirs[z] = d
        sample_speeds[z] = s

    # --- Epistemic: variance of z-draw means ---
    epistemic_dir_std = circular_std(sample_dirs, axis=0)
    epistemic_speed_std = sample_speeds.std(axis=0, ddof=1)

    # --- Aleatoric speed: Jacobian-based ---
    u = u_mu * params.u_std + params.u_mean
    v = v_mu * params.v_std + params.v_mean
    speed = np.maximum(np.sqrt(u**2 + v**2), 1e-6)
    sigma_u = sigma_stack[..., 0] * params.u_std
    sigma_v = sigma_stack[..., 1] * params.v_std

    ale_speed_var = ((u / speed)**2 * sigma_u**2
                     + (v / speed)**2 * sigma_v**2)
    aleatoric_speed_std = np.sqrt(ale_speed_var.mean(axis=0))

    # --- Aleatoric direction: MC integration ---
    rng = np.random.default_rng(seed)
    K = min(mc_samples, 10)
    within_z_std = np.empty((n_z, n_pts))
    for z in range(n_z):
        eps_u = rng.normal(0, 1, (K, n_pts)) * sigma_u[z]
        eps_v = rng.normal(0, 1, (K, n_pts)) * sigma_v[z]
        mc_u = u[z] + eps_u
        mc_v = v[z] + eps_v
        mc_dirs = np.degrees(np.arctan2(-mc_u, -mc_v)) % 360
        within_z_std[z] = circular_std(mc_dirs, axis=0)

    aleatoric_dir_std = within_z_std.mean(axis=0)

    return {
        'epistemic_dir_std': epistemic_dir_std,
        'epistemic_speed_std': epistemic_speed_std,
        'aleatoric_dir_std': aleatoric_dir_std,
        'aleatoric_speed_std': aleatoric_speed_std,
    }


# ---------------------------------------------------------------------------
# Inference API
# ---------------------------------------------------------------------------

class WindPredictor:
    def __init__(self, checkpoint_path, num_hidden=None,
                 num_latents=None,
                 latent_layers=None, deterministic_layers=None,
                 dropout=None,
                 num_decoder_layers=None,
                 device=None, params: NormParams = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model, ckpt = load_model_checkpoint(
            checkpoint_path, self.device, num_hidden,
            num_latents=num_latents,
            latent_layers=latent_layers,
            deterministic_layers=deterministic_layers,
            dropout=dropout,
            num_decoder_layers=num_decoder_layers)
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
              f"(val_loss={ckpt.get('val_loss', float('nan')):.4f})")

        if params is not None:
            self.params = params
        elif 'norm_params' in ckpt:
            # Checkpoints store to_dict() output
            self.params = norm_params_from_dict(ckpt['norm_params'])
        else:
            self.params = LEGACY_PARAMS
            print("Warning: checkpoint has no norm_params; "
                  "using legacy approximate defaults.")

    def _obs_to_tensors(self, observations):
        return observations_to_tensors(observations, self.device, self.params)

    def _queries_to_tensor(self, queries):
        return queries_to_tensor(queries, self.device, self.params)

    @torch.no_grad()
    def predict(self, context_observations, query_points, n_samples=50):
        """
        Predict wind at query_points given context observations.

        For each of n_samples latent z-draws from the prior, computes
        mu(z) and sigma(z). The predictive mean is E[mu(z)] over draws.
        Total uncertainty (sqrt(Var(mu) + E[sigma^2])) combines epistemic
        and aleatoric components analytically — no MC sampling noise.
        """
        context_x, context_y = self._obs_to_tensors(context_observations)
        target_x = self._queries_to_tensor(query_points)

        mu_samples, sigma_samples = [], []
        for _ in range(n_samples):
            mu, sigma, _, _, _ = self.model(
                context_x, context_y, target_x,
                target_y=None
            )
            mu_samples.append(mu.squeeze(0).cpu().numpy())
            sigma_samples.append(sigma.squeeze(0).cpu().numpy())

        mu_stack = np.stack(mu_samples, axis=0)
        sigma_stack = np.stack(sigma_samples, axis=0)

        # Mean predictions from mu across z-draws (no aleatoric noise)
        mean_mu = mu_stack.mean(axis=0)
        dirs, speeds = _uv_to_physical(
            mean_mu[..., 0], mean_mu[..., 1], self.params)
        mean_dirs = dirs
        mean_speeds = speeds

        # Epistemic & aleatoric components
        components = compute_uncertainty_components(
            mu_stack, sigma_stack, self.params)

        # Total uncertainty: sqrt(Var(mu) + E[sigma^2]) — analytic
        total_dir_std = np.sqrt(
            components['epistemic_dir_std']**2
            + components['aleatoric_dir_std']**2)
        total_speed_std = np.sqrt(
            components['epistemic_speed_std']**2
            + components['aleatoric_speed_std']**2)

        return {
            'wind_dir_deg': mean_dirs,
            'wind_speed_kt': mean_speeds,
            'wind_dir_std': total_dir_std,
            'wind_speed_std': total_speed_std,
            **components,
        }
