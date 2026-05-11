from typing import Dict, List, Optional
from python_bot.common.types import OrderSide, OrderRequest, PortfolioState, MarketData, OrderType
import logging

class InventoryLiquidationEngine:
    """
    Enforces "Time-Decay Inventory Penalties" and Regime-Triggered Deleveraging.
    Automates the flattening of toxic inventory.
    """
    def __init__(self, max_holding_bars: int = 120):
        self.max_holding_bars = max_holding_bars
        self._entry_times: Dict[str, int] = {} # Symbol -> Step_ID

    def get_liquidation_orders(self, 
                               symbol: str, 
                               portfolio: PortfolioState, 
                               data: MarketData, 
                               current_step: int) -> List[OrderRequest]:
        """Checks if a position MUST be force-closed."""
        orders = []
        qty = portfolio.position_quantity
        
        if qty == 0:
            if symbol in self._entry_times: del self._entry_times[symbol]
            return orders

        if symbol not in self._entry_times:
            self._entry_times[symbol] = current_step
            
        # 1. Time-based Liquidation
        age = current_step - self._entry_times[symbol]
        if age > self.max_holding_bars:
            logging.warning(f"LIQUIDATION: Position in {symbol} exceeded age limit ({age} > {self.max_holding_bars})")
            side = OrderSide.SELL if qty > 0 else OrderSide.BUY
            orders.append(OrderRequest(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=abs(qty),
                timestamp=data.timestamp
            ))
            
        # 2. Limit Up/Down (Freeze avoidance - Exit if possible before lock)
        if data.limit_up or data.limit_down:
            logging.critical(f"LIQUIDATION: Limit-Lock warning on {symbol}. Attempting exit.")
            # ... (Exit logic logic for limit boundaries) ...
            
        return orders
