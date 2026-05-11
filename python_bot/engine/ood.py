import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from scipy.spatial.distance import mahalanobis
from python_bot.common.types import MarketRegime

class AdvancedOODDetector:
    """
    Solves "Silent Distribution Shift".
    Uses Mahalanobis Distance and Latent Covariance Monitoring.
    """
    def __init__(self, feature_dim: int = 20, threshold_sigma: float = 3.0):
        self.dim = feature_dim
        self.threshold = threshold_sigma
        
        # Baseline distributions per regime
        self._baselines: Dict[MarketRegime, Dict[str, np.ndarray]] = {}
        self._is_calibrated = False

    def calibrate(self, regime: MarketRegime, training_data: np.ndarray):
        """Computes mean and inverse covariance from training vectors."""
        mean = np.mean(training_data, axis=0)
        cov = np.cov(training_data, rowvar=False) + np.eye(self.dim) * 1e-6
        inv_cov = np.linalg.inv(cov)
        
        self._baselines[regime] = {
            'mean': mean,
            'inv_cov': inv_cov
        }
        self._is_calibrated = True

    def calculate_score(self, vector: np.ndarray, regime: MarketRegime) -> float:
        """
        Returns the Mahalanobis distance. 
        Values > critical threshold indicate Out-of-Distribution (OOD).
        """
        if not self._is_calibrated or regime not in self._baselines:
            return 1.0 # Default High-Risk if unknown
            
        base = self._baselines[regime]
        try:
            m_dist = mahalanobis(vector, base['mean'], base['inv_cov'])
            # Normalize by dim to make it less sensitive to dimensionality
            return float(m_dist / np.sqrt(self.dim))
        except:
            return 10.0 # Numerical failure implies extreme OOD

    def is_ood(self, vector: np.ndarray, regime: MarketRegime) -> bool:
        score = self.calculate_score(vector, regime)
        return score > self.threshold

    @classmethod
    def load_mock_calibration(cls, dim: int = 20):
        # In production, this is loaded from pre-trained metadata
        detector = cls(feature_dim=dim)
        mock_data = np.random.normal(0, 1, (1000, dim))
        for r in MarketRegime:
            detector.calibrate(r, mock_data)
        return detector
