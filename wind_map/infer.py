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
from wind_map.preprocess import (
    normalise_coords, encode_wind, WIND_SPEED_MEAN, WIND_SPEED_STD
)
from wind_map.utils import circular_mean, circular_std


# ---------------------------------------------------------------------------
# Shared helpers used by infer, test, and visualise_uncertainty
# ---------------------------------------------------------------------------

def load_model_checkpoint(checkpoint_path, device, num_hidden=None,
                          num_layers=None, dropout=None):
    """Load a LatentModel from a checkpoint file.

    Returns (model, ckpt_dict).
    """
    ckpt = torch.load(checkpoint_path, map_location=device,
                      weights_only=False)
    hp = ckpt.get('hparams', {})
    num_hidden = num_hidden or hp.get('num_hidden')
    num_layers = num_layers or hp.get('num_layers', 4)
    dropout = dropout if dropout is not None else hp.get('dropout', 0.0)

    if num_hidden is None:
        raise ValueError(
            "No 'hparams' in checkpoint — pass num_hidden explicitly.")

    model = LatentModel(
        num_hidden, num_layers=num_layers,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt


def observations_to_tensors(observations, device):
    """Convert observation dicts to (x, y) tensors with batch dimension."""
    xs, ys = [], []
    for obs in observations:
        lat_n, lon_n, alt_n = normalise_coords(
            obs['lat'], obs['lon'], obs['alt_ft']
        )
        sin_w, cos_w, spd_n = encode_wind(
            obs['wind_dir'], obs['wind_speed']
        )
        xs.append([lat_n, lon_n, alt_n])
        ys.append([sin_w, cos_w, spd_n])
    x = torch.FloatTensor(xs).unsqueeze(0).to(device)
    y = torch.FloatTensor(ys).unsqueeze(0).to(device)
    return x, y


def queries_to_tensor(queries, device):
    """Convert query dicts to target_x tensor with batch dimension."""
    xs = []
    for q in queries:
        lat_n, lon_n, alt_n = normalise_coords(
            q['lat'], q['lon'], q['alt_ft']
        )
        xs.append([lat_n, lon_n, alt_n])
    return torch.FloatTensor(xs).unsqueeze(0).to(device)


def compute_uncertainty_components(mu_stack, sigma_stack):
    """Given mu_stack [n_samples, N, 3] and sigma_stack [n_samples, N, 3]
    from multiple latent draws, return epistemic and aleatoric uncertainty.

    Returns dict with keys:
        epistemic_dir_std, epistemic_speed_std,
        aleatoric_dir_std, aleatoric_speed_std
    """
    sin_mu = mu_stack[..., 0]
    cos_mu = mu_stack[..., 1]
    spd_mu = mu_stack[..., 2]

    sample_dirs = np.degrees(np.arctan2(sin_mu, cos_mu)) % 360
    sample_speeds = (
        np.clip(spd_mu, 0, None) * WIND_SPEED_STD + WIND_SPEED_MEAN
    )

    epistemic_dir_std = circular_std(sample_dirs, axis=0)
    epistemic_speed_std = sample_speeds.std(axis=0, ddof=1)

    sin_sig = sigma_stack[..., 0]
    cos_sig = sigma_stack[..., 1]
    spd_sig = sigma_stack[..., 2]
    R2 = sin_mu ** 2 + cos_mu ** 2 + 1e-6
    aleatoric_dir_var_rad2 = (
        (cos_mu / R2) ** 2 * sin_sig ** 2
        + (sin_mu / R2) ** 2 * cos_sig ** 2
    )
    aleatoric_dir_var_deg2 = np.degrees(np.sqrt(aleatoric_dir_var_rad2)) ** 2
    aleatoric_dir_std = np.sqrt(aleatoric_dir_var_deg2.mean(axis=0))

    aleatoric_speed_var = (spd_sig * WIND_SPEED_STD) ** 2
    aleatoric_speed_std = np.sqrt(aleatoric_speed_var.mean(axis=0))

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
                 num_layers=None, dropout=None,
                 device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model, ckpt = load_model_checkpoint(
            checkpoint_path, self.device, num_hidden, num_layers, dropout)
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
              f"(val_loss={ckpt.get('val_loss', float('nan')):.4f})")

    def _obs_to_tensors(self, observations):
        return observations_to_tensors(observations, self.device)

    def _queries_to_tensor(self, queries):
        return queries_to_tensor(queries, self.device)

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
            mu, sigma, _, _ = self.model(
                context_x, context_y, target_x,
                target_y=None
            )
            mu_samples.append(mu.squeeze(0).cpu().numpy())
            sigma_samples.append(sigma.squeeze(0).cpu().numpy())

        mu_stack = np.stack(mu_samples, axis=0)
        sigma_stack = np.stack(sigma_samples, axis=0)

        # Mean predictions from mu across z-draws (no aleatoric noise)
        sin_mu = mu_stack[..., 0]
        cos_mu = mu_stack[..., 1]
        spd_mu = mu_stack[..., 2]
        mu_dirs = np.degrees(np.arctan2(sin_mu, cos_mu)) % 360
        mu_speeds = np.clip(spd_mu, 0, None) * WIND_SPEED_STD + WIND_SPEED_MEAN
        mean_dirs = circular_mean(mu_dirs, axis=0)
        mean_speeds = mu_speeds.mean(axis=0)

        # Epistemic & aleatoric components
        components = compute_uncertainty_components(mu_stack, sigma_stack)

        # Total uncertainty: sqrt(Var(mu) + E[sigma^2]) — analytic, no MC noise
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
