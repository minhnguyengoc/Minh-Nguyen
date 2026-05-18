import pandas as pd
import numpy as np
from typing import List, Dict, Any

class EntryQualityAnalyzer:
    """
    Analyzes alpha decay and predictive edge for trade entries.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.entries = []

    def log_entry(self, step: int, symbol: str, side: str, price: float):
        """Records an entry for future analysis."""
        self.entries.append({
            "step": step,
            "symbol": symbol,
            "side": side,
            "entry_price": price
        })

    def analyze(self) -> Dict[str, Any]:
        """Calculates forward metrics for all logged entries."""
        if not self.entries:
            return {"status": "NO_ENTRIES"}
            
        results = []
        for entry in self.entries:
            idx = entry['step']
            symbol = entry['symbol']
            price = entry['entry_price']
            side = 1 if entry['side'] == 'BUY' else -1
            
            # Forward Windows
            sub_df = self.df.iloc[idx:idx+100] # Look ahead 100 bars
            if sub_df.empty: continue
            
            # Relative returns
            f_rets = (sub_df['close'] - price) / price * side
            
            results.append({
                "symbol": symbol,
                "f_ret_5": f_rets.iloc[min(5, len(f_rets)-1)],
                "f_ret_20": f_rets.iloc[min(20, len(f_rets)-1)],
                "mfe": f_rets.max(),
                "mae": f_rets.min(),
                "efficiency": f_rets.max() / (abs(f_rets.min()) + 1e-9)
            })
            
        df_res = pd.DataFrame(results)
        return {
            "avg_f_ret_20": df_res['f_ret_20'].mean(),
            "avg_mfe": df_res['mfe'].mean(),
            "avg_mae": df_res['mae'].mean(),
            "entry_efficiency_score": df_res['efficiency'].mean(),
            "positive_edge_ratio": (df_res['f_ret_20'] > 0).mean()
        }
