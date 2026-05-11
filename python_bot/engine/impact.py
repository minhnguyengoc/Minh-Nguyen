import numpy as np
from typing import Dict, Tuple, Optional
from python_bot.common.types import OrderSide

class InstitutionalImpactModel:
    """
    Models Endogenous Market Impact.
    Prevents the agent from learning unrealistic size scaling.
    Implements Square-Root Impact Law.
    """
    def __init__(self, 
                 temp_impact_beta: float = 0.1, 
                 perm_impact_gamma: float = 0.01,
                 avg_daily_vol: float = 1_000_000.0):
        self.beta = temp_impact_beta     # Coefficient for temporary impact
        self.gamma = perm_impact_gamma   # Coefficient for permanent impact
        self.adv = avg_daily_vol         # Average Daily Volume context
        
        self._cumulative_permanent_impact = 0.0

    def calculate_fill_impact(self, 
                             order_qty: int, 
                             order_side: OrderSide, 
                             current_vol: float, 
                             current_price: float) -> Tuple[float, float]:
        """
        Returns (Execution_Price_With_Impact, New_Permanent_Impact_Baseline).
        
        Temp Impact: I_temp = sigma * beta * (qty / vol_interval)^0.5
        Perm Impact: I_perm = sigma * gamma * (qty / ADV)
        """
        direction = 1 if order_side == OrderSide.BUY else -1
        
        participation = order_qty / (current_vol + 1e-8)
        
        # 1. Temporary Impact (Slippage during execution)
        # Using a simplified version of square root law
        temp_impact_bps = self.beta * np.sqrt(max(0, participation)) * 100 
        temp_impact_abs = (temp_impact_bps / 10000.0) * current_price
        
        # 2. Permanent Impact (Price shift after execution)
        perm_impact_bps = self.gamma * (order_qty / (self.adv + 1e-8)) * 100
        perm_impact_abs = (perm_impact_bps / 10000.0) * current_price
        
        self._cumulative_permanent_impact += direction * perm_impact_abs
        
        impacted_price = current_price + (direction * temp_impact_abs)
        
        return impacted_price, self._cumulative_permanent_impact

    def get_market_adjustment(self) -> float:
        """Returns the total price drift caused by the agent's historical actions."""
        return self._cumulative_permanent_impact

    def reset(self):
        self._cumulative_permanent_impact = 0.0
