import numpy as np
from enum import Enum
from typing import Dict, Any, Tuple, Optional

class CostMode(Enum):
    NORMAL = "NORMAL"
    STRESS = "STRESS"
    HELL = "HELL"

class InstitutionalCostModel:
    """
    Advanced Institutional Cost Model for Vietnam Stock Market (HOSE).
    Implements multi-regime friction, square-root market impact, and liquidity penalties.
    """
    def __init__(self, mode: CostMode = CostMode.NORMAL):
        self.mode = mode
        # Baseline institutional configs (bps = basis points, 1/100 of 1%)
        self.params = {
            CostMode.NORMAL: {
                "broker_fee_bps": 15,
                "sell_tax_bps": 10,
                "half_spread_bps": 5,
                "fixed_slippage_bps": 5,
                "impact_coef": 0.1,  # Square-root law coefficient
                "min_fill_ratio": 0.8,
                "latency_bps": 1
            },
            CostMode.STRESS: {
                "broker_fee_bps": 20,
                "sell_tax_bps": 10,
                "half_spread_bps": 15,
                "fixed_slippage_bps": 20,
                "impact_coef": 0.25,
                "min_fill_ratio": 0.5,
                "latency_bps": 5
            },
            CostMode.HELL: {
                "broker_fee_bps": 30,
                "sell_tax_bps": 10,
                "half_spread_bps": 40,
                "fixed_slippage_bps": 50,
                "impact_coef": 0.5,
                "min_fill_ratio": 0.2,
                "latency_bps": 20
            }
        }
        self.current = self.params[mode]

    def calculate_execution(self, 
                            side: str, 
                            price: float, 
                            quantity: int, 
                            bar_volume: float, 
                            volatility: float = 0.01) -> Dict[str, Any]:
        """
        Calculates impact-aware execution.
        Side: 'BUY' or 'SELL'
        """
        p = self.current
        side_mult = 1 if side.upper() == 'BUY' else -1
        
        # 1. Market Impact (Square-root Law)
        # impact = impact_coef * sigma * sqrt(order_val / volume_val)
        bar_value = bar_volume * price
        order_value = quantity * price
        
        participation_rate = order_value / (bar_value + 1e-9)
        impact_bps = p["impact_coef"] * volatility * 10000 * np.sqrt(participation_rate + 1e-8)
        
        # 2. Hybrid Slippage (Fixed + Latency)
        slippage_bps = p["fixed_slippage_bps"] + p["latency_bps"]
        
        # 3. Total Non-Fee Friction
        friction_bps = p["half_spread_bps"] + slippage_bps + impact_bps
        friction_ratio = friction_bps / 10000.0
        
        effective_price = price * (1 + (side_mult * friction_ratio))
        
        # 4. Fill Simulation
        fill_ratio = 1.0
        if participation_rate > (0.1 if self.mode == CostMode.NORMAL else 0.05):
            # Penalize Large Orders relative to Volume
            fill_ratio = max(p["min_fill_ratio"], 1.0 - (participation_rate * 2.0))
        
        # Random rejection under stress
        if self.mode != CostMode.NORMAL and np.random.random() < (0.01 if self.mode == CostMode.STRESS else 0.05):
            return {"rejected": True, "reason": "STRESS_LIQUIDITY_TIMEOUT", "fill_ratio": 0}

        # 5. Fee & Tax
        broker_fee = order_value * (p["broker_fee_bps"] / 10000.0)
        sell_tax = (order_value * (p["sell_tax_bps"] / 10000.0)) if side.upper() == 'SELL' else 0.0
        
        total_costs = broker_fee + sell_tax + (abs(effective_price - price) * quantity)
        
        return {
            "rejected": False,
            "effective_price": effective_price,
            "quantity_filled": int(quantity * fill_ratio),
            "fee": broker_fee,
            "tax": sell_tax,
            "slippage_cost": friction_ratio * order_value,
            "impact_cost": (impact_bps / 10000.0) * order_value,
            "total_cost": total_costs,
            "fill_ratio": fill_ratio
        }
