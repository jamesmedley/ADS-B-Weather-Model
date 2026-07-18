"""
wind_map — Attentive Neural Process for wind prediction
from ADS-B aircraft observations.
"""

from wind_map.infer import WindPredictor
from wind_map.network import LatentModel
from wind_map.train import train

__all__ = ['WindPredictor', 'LatentModel', 'train']
