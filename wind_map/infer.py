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


class WindPredictor:
    def __init__(self, checkpoint_path, num_hidden=None,
                 num_layers=None, dropout=None,
                 device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)

        ckpt = torch.load(
            checkpoint_path, map_location=self.device,
            weights_only=False)
        hp = ckpt.get('hparams', {})

        num_hidden = num_hidden or hp.get('num_hidden', 128)
        num_layers = num_layers or hp.get('num_layers', 4)
        dropout = dropout if dropout is not None else hp.get('dropout', 0.0)

        self.model = LatentModel(
            num_hidden, num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)
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

        For each of n_samples z-draws from the prior, samples
        y ~ Normal(mu(z), sigma(z)) to capture both epistemic
        and aleatoric uncertainty in the prediction spread.
        """
        context_x, context_y = self._obs_to_tensors(context_observations)
        target_x = self._queries_to_tensor(query_points)

        mu_samples, sigma_samples = [], []
        pred_dirs, pred_speeds = [], []
        for _ in range(n_samples):
            mu, sigma, _, _ = self.model(
                context_x, context_y, target_x,
                target_y=None
            )
            mu_np = mu.squeeze(0).cpu().numpy()
            sigma_np = sigma.squeeze(0).cpu().numpy()
            mu_samples.append(mu_np)
            sigma_samples.append(sigma_np)

            # Sample from predictive distribution
            noise = np.random.randn(*mu_np.shape).astype(np.float32)
            y_sample = mu_np + sigma_np * noise
            sin_s = y_sample[..., 0]
            cos_s = y_sample[..., 1]
            spd_s = np.clip(y_sample[..., 2], 0, None)
            pred_dirs.append(np.degrees(np.arctan2(sin_s, cos_s)) % 360)
            pred_speeds.append(spd_s * WIND_SPEED_STD + WIND_SPEED_MEAN)

        mu_stack = np.stack(mu_samples, axis=0)
        sigma_stack = np.stack(sigma_samples, axis=0)
        pred_dir_stack = np.stack(pred_dirs, axis=0)
        pred_spd_stack = np.stack(pred_speeds, axis=0)

        # Main predictions from sampled y's
        mean_dirs = circular_mean(pred_dir_stack, axis=0)
        mean_speeds = pred_spd_stack.mean(axis=0)

        # Total uncertainty: spread of y-samples
        total_dir_std = circular_std(pred_dir_stack, axis=0)
        total_speed_std = pred_spd_stack.std(axis=0, ddof=1)

        # Epistemic: spread of mu(z) across z-draws
        sin_mu = mu_stack[..., 0]
        cos_mu = mu_stack[..., 1]
        sample_dirs = np.degrees(np.arctan2(sin_mu, cos_mu)) % 360
        sample_speeds = (
            np.clip(mu_stack[..., 2], 0, None)
            * WIND_SPEED_STD + WIND_SPEED_MEAN
        )
        epistemic_dir_std = circular_std(sample_dirs, axis=0)
        epistemic_speed_std = sample_speeds.std(axis=0, ddof=1)

        # Aleatoric: delta-method propagation of sigma
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

        aleatoric_speed_var = (spd_sig * WIND_SPEED_STD) ** 2
        mean_aleatoric_speed_var = aleatoric_speed_var.mean(axis=0)

        return {
            'wind_dir_deg': mean_dirs,
            'wind_speed_kt': mean_speeds,
            'wind_dir_std': total_dir_std,
            'wind_speed_std': total_speed_std,
            'epistemic_dir_std': epistemic_dir_std,
            'epistemic_speed_std': epistemic_speed_std,
            'aleatoric_dir_std': np.sqrt(mean_aleatoric_dir_var),
            'aleatoric_speed_std': np.sqrt(mean_aleatoric_speed_var),
        }
