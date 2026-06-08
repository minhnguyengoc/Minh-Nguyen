import logging
from typing import Dict, Any

logger = logging.getLogger("VNStockBot.PositionManager")

class Position:
    """Manages raw volume shares, average purchase cost and PnL breakdown of a single ticker holding."""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.qty = 0.0
        self.avg_cost = 0.0
        self.realized_pnl = 0.0
        
    def buy(self, price: float, qty: float) -> Tuple = None:
        """Adds shares to position and adjusts average cost dynamically."""
        if qty <= 0:
            return
            
        total_value_before = self.qty * self.avg_cost
        new_value = qty * price
        
        self.qty += qty
        self.avg_cost = (total_value_before + new_value) / self.qty
        logger.info(f"Position BUY on {self.ticker}: quantity={qty}, price={price}, avg_cost={self.avg_cost:.2f}")

    def sell(self, price: float, qty: float) -> float:
        """
        Reduces position volume and returns the realized PnL.
        Raises ValueError if selling more than available.
        """
        if qty <= 0:
            return 0.0
            
        if qty > self.qty:
            raise ValueError(f"Cannot sell {qty} shares of {self.ticker}. Only {self.qty} available.")
            
        realized = qty * (price - self.avg_cost)
        self.qty -= qty
        self.realized_pnl += realized
        
        if self.qty == 0:
            self.avg_cost = 0.0
            
        logger.info(f"Position SELL on {self.ticker}: qty={qty}, price={price}, realized_pnl={realized:.2f}")
        return realized

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculates current paper profit or loss of the active position."""
        return self.qty * (current_price - self.avg_cost)
