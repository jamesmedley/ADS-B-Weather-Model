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
from wind_map.preprocess import normalise_coords, encode_wind, MAX_WIND_KT
from wind_map.utils import circular_mean, circular_std


class WindPredictor:
    def __init__(self, checkpoint_path, num_hidden=128,
                 num_layers=2, ffn_expansion=2, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        self.model = LatentModel(
            num_hidden, num_layers=num_layers,
            ffn_expansion=ffn_expansion
        ).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
              f"(val_loss={ckpt.get('val_loss', float('nan')):.4f})")

    def _obs_to_tensors(self, observations):
        """Convert observation dicts to (x, y) tensors."""
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
        x = torch.FloatTensor(xs).unsqueeze(0).to(self.device)
        y = torch.FloatTensor(ys).unsqueeze(0).to(self.device)
        return x, y

    def _queries_to_tensor(self, queries):
        """Convert query dicts to target_x tensor."""
        xs = []
        for q in queries:
            lat_n, lon_n, alt_n = normalise_coords(
                q['lat'], q['lon'], q['alt_ft']
            )
            xs.append([lat_n, lon_n, alt_n])
        return torch.FloatTensor(xs).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, context_observations, query_points, n_samples=50):
        """
        Predict wind at query_points given context observations.

        Draws n_samples z-samples from the prior, decodes mu(z) for each
        (no injected noise — keeps spatial smoothness), and combines
        epistemic (spread of mu across z) + aleatoric (delta-method sigma)
        uncertainty.
        """
        context_x, context_y = self._obs_to_tensors(context_observations)
        target_x = self._queries_to_tensor(query_points)

        mu_samples, sigma_samples = [], []
        for _ in range(n_samples):
            mu, sigma, _, _ = self.model(
                context_x, context_y, target_x,
                target_y=None
            )
            mu_samples.append(mu.squeeze(0).cpu())
            sigma_samples.append(sigma.squeeze(0).cpu())

        mu_stack = torch.stack(mu_samples, dim=0).numpy()
        sigma_stack = torch.stack(sigma_samples, dim=0).numpy()

        sin_mu = mu_stack[..., 0]
        cos_mu = mu_stack[..., 1]
        spd_mu = mu_stack[..., 2]

        sample_dirs = np.degrees(np.arctan2(sin_mu, cos_mu)) % 360
        sample_speeds = np.clip(spd_mu, 0, None) * MAX_WIND_KT

        mean_dirs = circular_mean(sample_dirs, axis=0)
        mean_speeds = sample_speeds.mean(axis=0)

        # Epistemic: spread of mu(z) across z-draws
        epistemic_dir_std = circular_std(sample_dirs, axis=0)
        epistemic_speed_var = sample_speeds.var(axis=0)

        # Aleatoric: delta-method propagation of sigma
        # through atan2, per z-sample
        sin_sig = sigma_stack[..., 0]
        cos_sig = sigma_stack[..., 1]
        spd_sig = sigma_stack[..., 2]
        R2 = sin_mu ** 2 + cos_mu ** 2 + 1e-6
        aleatoric_dir_var_rad2 = (
            (cos_mu / R2) ** 2 * sin_sig ** 2
            + (sin_mu / R2) ** 2 * cos_sig ** 2
        )
        aleatoric_dir_var_deg2 = (
            np.degrees(np.sqrt(aleatoric_dir_var_rad2)) ** 2
        )
        mean_aleatoric_dir_var = aleatoric_dir_var_deg2.mean(axis=0)

        aleatoric_speed_var = (spd_sig * MAX_WIND_KT) ** 2
        mean_aleatoric_speed_var = aleatoric_speed_var.mean(axis=0)

        # Total variance = epistemic + aleatoric
        dir_stds = np.sqrt(epistemic_dir_std ** 2 + mean_aleatoric_dir_var)
        speed_stds = np.sqrt(epistemic_speed_var + mean_aleatoric_speed_var)

        return {
            'wind_dir_deg': mean_dirs,
            'wind_speed_kt': mean_speeds,
            'wind_dir_std': dir_stds,
            'wind_speed_std': speed_stds,
        }
