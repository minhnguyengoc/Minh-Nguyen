import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

class NAVReportGenerator:
    """
    Generates institutional-grade backtest reports.
    """
    def __init__(self, output_dir: str = "reports/backtest"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def calculate_metrics(self, nav_series: pd.Series, trade_log: List[Dict]) -> Dict[str, Any]:
        """Calculates core quantitative metrics."""
        returns = nav_series.pct_change().dropna()
        
        # Risk Ratios
        total_ret = (nav_series.iloc[-1] / nav_series.iloc[0]) - 1.0
        ann_vol = returns.std() * np.sqrt(252 * 240) # Assumes 1m bars in trading day
        sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252 * 240)
        
        # Drawdown logic
        peak = nav_series.cummax()
        dd = (nav_series - peak) / peak
        max_dd = dd.min()
        
        # Trade metrics
        df_trades = pd.DataFrame(trade_log)
        if df_trades.empty:
            return {"status": "NO_TRADES_EXECUTED", "return": total_ret}
            
        wins = df_trades[df_trades['cost'] < 0] # Re-selling at gain (cost in log usually positive)
        # For simplicity, calculate from trade sequence
        
        return {
            "ending_equity": nav_series.iloc[-1],
            "total_return_pct": total_ret * 100,
            "ann_vol_pct": ann_vol * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd * 100,
            "calmar_ratio": abs(total_ret / (max_dd + 1e-9)),
            "trade_count": len(df_trades),
            "win_rate": 0.0 # Placeholder
        }

    def plot_nav(self, nav_series: pd.Series, symbol_name: str = "VN30"):
        """Generates equity curve and drawdown charts."""
        plt.figure(figsize=(12, 8))
        
        # 1. Equity Curve
        plt.subplot(2, 1, 1)
        plt.plot(nav_series, label="NAV", color='blue')
        plt.title(f"Institutional NAV Curve - {symbol_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 2. Drawdown
        plt.subplot(2, 1, 2)
        peak = nav_series.cummax()
        dd = (nav_series - peak) / peak
        plt.fill_between(dd.index, dd, 0, color='red', alpha=0.3, label="Drawdown")
        plt.ylim(-0.5, 0)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        save_path = os.path.join(self.output_dir, f"nav_{symbol_name.lower()}.png")
        plt.savefig(save_path)
        plt.close()
        return save_path
