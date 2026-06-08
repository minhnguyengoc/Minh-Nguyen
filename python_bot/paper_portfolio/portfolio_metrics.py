import numpy as np
import pandas as pd
from typing import Dict, Any

class PortfolioMetricsSuite:
    """Computes sophisticated performance statistics and risk metrics."""
    
    @classmethod
    def calculate_drawdown(cls, equity_curve: pd.Series) -> Tuple = None:
        """Computes peak-to-trough drawdowns and returns (max_drawdown, current_drawdown)."""
        if equity_curve.empty:
            return 0.0, 0.0
        peaks = equity_curve.cummax()
        drawdowns = (peaks - equity_curve) / (peaks + 1e-9)
        return float(drawdowns.max()), float(drawdowns.iloc[-1])

    @classmethod
    def analyze(cls, returns: pd.Series, equity_curve: pd.Series) -> Dict[str, Any]:
        """Runs overall performance statistics evaluation."""
        if equity_curve.empty:
            return {}
            
        initial_eq = equity_curve.iloc[0]
        final_eq = equity_curve.iloc[-1]
        total_return = (final_eq - initial_eq) / (initial_eq + 1e-9)
        
        max_dd, current_dd = cls.calculate_drawdown(equity_curve)
        
        # Sharpe ratio
        daily_std = returns.std()
        daily_mean = returns.mean()
        sharpe = (daily_mean / (daily_std + 1e-9)) * np.sqrt(252) if daily_std > 0 else 0.0
        
        return {
            "total_return": float(total_return),
            "max_drawdown": float(max_dd),
            "current_drawdown": float(current_dd),
            "sharpe_ratio": float(sharpe),
            "volatility": float(daily_std),
            "starting_equity": float(initial_eq),
            "ending_equity": float(final_eq)
        }
