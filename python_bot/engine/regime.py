import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from python_bot.common.types import MarketRegime, MarketData

class InstitutionalRegimeDetector:
    """
    Advanced Regime Detection Engine for VN Equities.
    Detects Volatility Clusters, Trend Persistence, and Liquidity Entropies.
    Used for routing normalization scalers and modulating risk.
    """
    def __init__(self, window: int = 40):
        self.window = window
        self._history: List[MarketData] = []
        
    def update(self, data: MarketData) -> MarketRegime:
        self._history.append(data)
        if len(self._history) > self.window:
            self._history.pop(0)
            
        if len(self._history) < self.window:
            return MarketRegime.SIDEWAYS # Warmup default
            
        df = pd.DataFrame([d.model_dump() for d in self._history])
        c = df['close'].values
        v = df['volume'].values
        h = df['high'].values
        l = df['low'].values
        
        # 1. Volatility Regime (Normalized ATR)
        tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
        atr = np.mean(tr)
        atr_std = np.std(tr)
        current_vol = tr[-1]
        
        is_panic = current_vol > (atr + 2.5 * atr_std)
        if is_panic: return MarketRegime.PANIC
        
        # 2. Liquidity Regime (Volume Entropy / Decay)
        avg_vol = np.mean(v)
        if v[-1] < (avg_vol * 0.3):
            return MarketRegime.LIQUIDITY_CRUNCH
            
        # 3. Trend Regime (EMA Cross + RSI Strength)
        ema10 = df['close'].ewm(span=10).mean().iloc[-1]
        ema30 = df['close'].ewm(span=30).mean().iloc[-1]
        
        # Combined Signal
        if ema10 > ema30 * 1.002: # 20bps threshold to prevent flicker
            return MarketRegime.BULL
        elif ema10 < ema30 * 0.998:
            return MarketRegime.BEAR
            
        return MarketRegime.SIDEWAYS

    def get_context_metadata(self) -> Dict[str, Any]:
        """Provides raw statistical metrics for the observation metadata."""
        if len(self._history) < self.window: return {}
        
        c = np.array([d.close for d in self._history])
        v = np.array([d.volume for d in self._history])
        
        return {
            "volatility_zscore": (c[-1] - np.mean(c)) / (np.std(c) + 1e-8),
            "volume_participation": v[-1] / (np.mean(v) + 1e-8),
            "trend_strength": (c[-1] / c[0]) - 1.0
        }
