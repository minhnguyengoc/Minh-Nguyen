import logging
from typing import Tuple
from python_bot.risk.risk_limits import RiskLimits

logger = logging.getLogger("VNStockBot.TradeGuard")

class TradeGuard:
    """Standard pre-trade risk filter assessing size, short-limits, and instrument status."""
    
    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()
        
    def check_trade(self, ticker: str, action: str, qty: float, price: float, available_balance: float, current_pos: float) -> Tuple[bool, str]:
        """
        Validates individual order structure against pre-trade limits.
        Returns (is_allowed, rejected_reason).
        """
        action = action.upper()
        if action == "HOLD":
            return True, "NONE"
            
        # 1. Block multi-symbol live trading
        if ticker.upper() != "STB":
            return False, "FORBIDDEN_MULTI_SYMBOL_ORDER"
            
        # 2. Check quantity constraint
        if qty <= 0:
            return False, "INVALID_ORDER_QUANTITY"
            
        # 3. Lot size constraint (HOSE minimum lot size is 100 shares)
        if qty % self.limits.min_lot_size != 0:
            return False, f"LOT_SIZE_LOTTERY_VIOLATION: order qty {qty} is not a multiple of {self.limits.min_lot_size}"
            
        # 4. Short-selling security check
        if action in ["SELL", "CLOSE"] and not self.limits.allow_short:
            if qty > current_pos:
                return False, f"SHORT_SELLING_FORBIDDEN: sells {qty} but only has {current_pos} shares"
                
        # 5. Margin buying check (no negative cash)
        if action == "BUY":
            required_cash = qty * price
            if required_cash > available_balance:
                return False, f"INSUFFICIENT_CASH: requires {required_cash:.2f} but only has {available_balance:.2f}"
                
        return True, "NONE"
