import numpy as np
import pandas as pd
from typing import Dict, Any

class EntryLabelGenerator:
    """
    Generates training-only labels for trade entry quality.
    Used for curriculum reward shaping and diagnostic validation.
    """
    @staticmethod
    def calculate_labels(df: pd.DataFrame, 
                         current_idx: int, 
                         round_trip_cost_bps: float = 30.0) -> Dict[str, Any]:
        """
        Computes forward-looking metrics to determine 'optimal' direction.
        """
        cost_threshold = round_trip_cost_bps / 10000.0
        
        # Lookahead windows
        win_3 = 3
        win_5 = 5
        win_10 = 10
        
        def get_ret(w):
            if current_idx + w >= len(df):
                return 0.0
            p1 = df.iloc[current_idx]['close']
            p2 = df.iloc[current_idx + w]['close']
            return (p2 - p1) / (p1 + 1e-9)

        ret3 = get_ret(win_3)
        ret5 = get_ret(win_5)
        ret10 = get_ret(win_10)
        
        # Best direction based on 5-bar window
        best_direction = 0 # HOLD
        if ret5 > (cost_threshold * 1.5):
            best_direction = 1 # BUY
        elif ret5 < (-cost_threshold * 1.5):
            best_direction = 2 # SELL/EXIT
            
        return {
            "future_return_3": ret3,
            "future_return_5": ret5,
            "future_return_10": ret10,
            "net_future_return_after_cost": ret5 - cost_threshold,
            "best_direction": best_direction,
            "no_trade_zone": abs(ret5) < (cost_threshold * 0.8),
            "tradable_edge_after_cost": max(0.0, abs(ret5) - cost_threshold)
        }
