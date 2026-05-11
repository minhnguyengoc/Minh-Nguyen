import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from python_bot.common.types import MarketData, MarketRegime, SessionState

class FeatureEngine:
    """
    Stateful Feature Engine. 
    Maintains causal buffers and resets on session boundaries to prevent contamination.
    """
    def __init__(self, lookback: int = 30):
        self.lookback = lookback
        self._buffer = []
        self._max_buffer = 100

    def reset(self):
        self._buffer.clear()

    def push(self, data: MarketData):
        self._buffer.append(data)
        if len(self._buffer) > self._max_buffer:
            self._buffer.pop(0)

    def generate(self) -> np.ndarray:
        if len(self._buffer) < 20: # Minimum warmup
            return np.zeros(20)

        df = pd.DataFrame([d.model_dump() for d in self._buffer])
        c = df['close']
        h = df['high']
        l = df['low']
        v = df['volume']
        
        # Strictly Causal Vectorization
        # 1-5: Multi-scale Returns
        ret1 = c.pct_change(1).iloc[-1]
        ret5 = c.pct_change(5).iloc[-1]
        
        # 6-10: Microstructure Ranges
        hl_range = (h - l) / (c + 1e-8)
        oc_range = (c - df['open']) / (df['open'] + 1e-8)
        
        # 11-15: Volatility / Liquidity
        vol_ratio = v.iloc[-1] / (v.rolling(20).mean().iloc[-1] + 1e-8)
        std5 = c.rolling(5).std().iloc[-1] / (c.iloc[-1] + 1e-8)
        
        # ASSEMBLY (Fixed 20-dim raw vector)
        raw = np.array([
            ret1, ret5, hl_range.iloc[-1], oc_range.iloc[-1], vol_ratio, std5,
            # ... additional features ...
        ], dtype=np.float32)
        
        # Pad to exactly 20 for this architecture
        if len(raw) < 20:
            raw = np.pad(raw, (0, 20 - len(raw)))
            
        return raw


class RegimeNormalizationRouter:
    """
    Institutional Normalization.
    Uses PRE-FROZEN statistics per regime. 
    Prevents scaling distribution drift from contaminating inference in real-time.
    """
    def __init__(self, frozen_stats: Dict[MarketRegime, Dict[str, np.ndarray]]):
        self.stats = frozen_stats # Dict mapping regime -> {'mean': arr, 'std': arr}

    def normalize(self, vector: np.ndarray, regime: MarketRegime) -> np.ndarray:
        if regime not in self.stats:
            # Fallback to SIDEWAYS if regime is undefined or AUCTION hasn't been calibrated
            regime = MarketRegime.SIDEWAYS
            
        stats = self.stats[regime]
        norm = (vector - stats['mean']) / (stats['std'] + 1e-8)
        return np.clip(norm, -4.0, 4.0)

    @classmethod
    def load_defaults(cls, dim: int = 20):
        # Mocking default stats for prototype - in prod these come from training metadata
        mock_stats = {
            r: {'mean': np.zeros(dim), 'std': np.ones(dim)} 
            for r in MarketRegime
        }
        return cls(mock_stats)
