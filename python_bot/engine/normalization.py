import numpy as np
from typing import Dict, Optional
from python_bot.common.types import MarketRegime

class FrozenQuantileNormalizer:
    """
    Solves Floating-Point Instability and Scaler Contamination.
    Uses pre-computed percentiles per regime to map features to [-1, 1].
    Strictly deterministic across platforms.
    """
    def __init__(self, regime_stats: Dict[MarketRegime, Dict[str, np.ndarray]]):
        self.stats = regime_stats # regime -> {'q05': arr, 'q50': arr, 'q95': arr}

    def normalize(self, vector: np.ndarray, regime: MarketRegime) -> np.ndarray:
        if regime not in self.stats:
            regime = MarketRegime.SIDEWAYS
            
        s = self.stats[regime]
        
        # Robust Scaling: (x - median) / (q95 - q05)
        # Prevents outlier explosion while maintaining ordinal information
        iqr = s['q95'] - s['q05']
        norm = (vector - s['q50']) / (iqr + 1e-8)
        
        # Symmetrical Tanh Squashing for Neural Input Stability
        return np.tanh(norm)

    @classmethod
    def load_defaults(cls, dim: int = 20):
        # Mock stats for prototype. 
        # In Institutional setup, these are exported from the Training Pipeline metadata.
        mock = {
            r: {
                'q05': np.full(dim, -2.0),
                'q50': np.zeros(dim),
                'q95': np.full(dim, 2.0)
            } for r in MarketRegime
        }
        return cls(mock)
