"""
infer.py — Predict wind vectors at arbitrary query points from
a trained Wind ANP.

Example:
    predictor = WindPredictor('checkpoint/best_model.pth.tar')
    result = predictor.predict(context, queries, n_samples=50)
    # result['wind_dir_deg'], result['wind_speed_kt']
    # result['wind_dir_std'], result['wind_speed_std']
    # result['combined_vector_std']   (total 2D vector std)

All uncertainty is computed in the u/v vector representation
(wind_map.uncertainty) and speed / direction derived from it, keeping this
consistent with training, validation, testing and the GP baseline.
"""

import torch
import numpy as np
from wind_map.network import LatentModel
from wind_map.gp import norm_params_from_dict
from wind_map.preprocess import (
    normalise_coords, encode_wind, NormParams, LEGACY_PARAMS,
)
from wind_map.uncertainty import (
    per_component_total_std,
    epistemic_aleatoric_uv_std,
    uv_to_speed_dir_std,
    combined_vector_std,
)

# Re-export the shared uncertainty helpers so ``infer`` remains the public
# home of all uncertainty derivation (single source of truth lives in
# wind_map.uncertainty).
__all__ = [
    'WindPredictor',
    'compute_uncertainty_components',
    'per_component_total_std',
    'epistemic_aleatoric_uv_std',
    'uv_to_speed_dir_std',
    'combined_vector_std',
]


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

    Uncertainty is first decomposed in the u/v vector representation
    (epistemic = spread of z-draw means; aleatoric = mean decoder variance),
    then speed / direction std are *derived* from those vector quantities with
    the shared delta method (see wind_map.uncertainty). This keeps the
    decomposition consistent with training, validation, testing and the GP.

    ``mc_samples`` and ``seed`` are kept for backward compatibility but are no
    longer used: direction uncertainty now flows from the vector covariance
    (deterministically) rather than a circular/directional MC integration.

    Returns dict with keys:
        epistemic_dir_std, epistemic_speed_std,
        aleatoric_dir_std, aleatoric_speed_std,
        and the raw vector components:
        epistemic_u_std, epistemic_v_std, aleatoric_u_std, aleatoric_v_std
    """
    # Decompose in u/v vector space (physical units).
    epi_u, epi_v, ale_u, ale_v = epistemic_aleatoric_uv_std(
        mu_stack, sigma_stack, params)

    # Mean physical vector across latent draws (linear in u/v, so the mean of
    # the means).
    u_mean = mu_stack[..., 0].mean(axis=0) * params.u_std + params.u_mean
    v_mean = mu_stack[..., 1].mean(axis=0) * params.v_std + params.v_mean

    epi_speed, epi_dir = uv_to_speed_dir_std(u_mean, v_mean, epi_u, epi_v)
    ale_speed, ale_dir = uv_to_speed_dir_std(u_mean, v_mean, ale_u, ale_v)

    return {
        'epistemic_dir_std': epi_dir,
        'epistemic_speed_std': epi_speed,
        'aleatoric_dir_std': ale_dir,
        'aleatoric_speed_std': ale_speed,
        'epistemic_u_std': epi_u,
        'epistemic_v_std': epi_v,
        'aleatoric_u_std': ale_u,
        'aleatoric_v_std': ale_v,
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
        Uncertainty is computed in the u/v vector representation via the law
        of total variance (epistemic + aleatoric analytically -- no MC sampling
        noise), then speed / direction std are derived from it with the shared
        delta method.

        Returns dict with keys:
            wind_dir_deg, wind_speed_kt,
            wind_dir_std, wind_speed_std,
            wind_u_std, wind_v_std, combined_vector_std,
            epistemic_dir_std, epistemic_speed_std,
            aleatoric_dir_std, aleatoric_speed_std,
            epistemic_u_std, epistemic_v_std,
            aleatoric_u_std, aleatoric_v_std,
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
        u_mean = mean_mu[..., 0] * self.params.u_std + self.params.u_mean
        v_mean = mean_mu[..., 1] * self.params.v_std + self.params.v_mean
        dirs, speeds = _uv_to_physical(
            mean_mu[..., 0], mean_mu[..., 1], self.params)

        # Total per-component u/v std (law of total variance, physical units)
        sigma_u, sigma_v = per_component_total_std(
            mu_stack, sigma_stack, self.params)
        # Derive speed / direction std from the vector uncertainty
        wind_speed_std, wind_dir_std = uv_to_speed_dir_std(
            u_mean, v_mean, sigma_u, sigma_v)
        combined_vec_std = combined_vector_std(sigma_u, sigma_v)

        # Epistemic & aleatoric decomposition (also in u/v then derived)
        components = compute_uncertainty_components(
            mu_stack, sigma_stack, self.params)

        return {
            'wind_dir_deg': dirs,
            'wind_speed_kt': speeds,
            'wind_dir_std': wind_dir_std,
            'wind_speed_std': wind_speed_std,
            'wind_u_std': sigma_u,
            'wind_v_std': sigma_v,
            'combined_vector_std': combined_vec_std,
            **components,
        }
