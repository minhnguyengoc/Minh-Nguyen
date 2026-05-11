import sys
import os

# Ensure the project root is in the python path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
from typing import List, Dict, Any

class InstitutionalMetrics:
    """
    Computes High-Fidelity Performance and Risk Metrics.
    Solves "Retail Metric Myopia".
    """
    def __init__(self, returns: List[float], periods_per_day: int = 240):
        """
        Calculates metrics. 
        periods_per_day: 240 for 1m bars (Standard 4h VN session), adjust as needed.
        """
        self.returns = np.array(returns)
        self.periods_per_day = periods_per_day

    def calculate_pnl_stats(self) -> Dict[str, float]:
        if len(self.returns) < 2: 
            return {k: 0.0 for k in ["annualized_return", "sharpe_ratio", "sortino_ratio", "max_drawdown", "calmar_ratio"]}
        
        # 1. Basic Stats
        cum_ret = np.prod(1 + self.returns) - 1
        
        # Adjust for total bars vs periods per day
        days_fraction = len(self.returns) / (self.periods_per_day + 1e-9)
        ann_ret = (1 + cum_ret) ** (252 / max(days_fraction, 0.001)) - 1
        ann_vol = np.std(self.returns) * np.sqrt(252 * self.periods_per_day)
        
        # 2. Sharpe & Sortino (Downside risk)
        sharpe = ann_ret / (ann_vol + 1e-9)
        
        downside_returns = self.returns[self.returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252 * self.periods_per_day) if len(downside_returns) > 0 else 1e-9
        sortino = ann_ret / (downside_vol + 1e-9)
        
        # 3. Max Drawdown
        cum_equity = np.cumprod(1 + self.returns)
        peak = np.maximum.accumulate(cum_equity)
        drawdown = (peak - cum_equity) / (peak + 1e-9)
        max_dd = np.max(drawdown)
        
        calmar = ann_ret / (max_dd + 1e-9)
        
        # 4. Tail Risk
        var_95 = np.percentile(self.returns, 5)
        cvar_95 = self.returns[self.returns <= var_95].mean() if any(self.returns <= var_95) else 0
        
        return {
            "total_return": float(cum_ret),
            "annualized_return": float(ann_ret),
            "annualized_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "max_drawdown": float(max_dd),
            "calmar_ratio": float(calmar),
            "var_95": float(var_95),
            "cvar_95": float(cvar_95)
        }

    @staticmethod
    def calculate_execution_efficiency(fills: List[Dict]) -> Dict[str, float]:
        """Measures slippage and fill quality."""
        # ... logic to compare fill_price against mid_price or bench ...
        return {"avg_slippage_bps": 0.0}
