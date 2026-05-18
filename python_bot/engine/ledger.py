from typing import Dict, List, Optional
from python_bot.common.types import FillEvent, PortfolioState, OrderSide
import logging

class ExposureLedger:
    """
    Truth-Source for Portfolio Exposure.
    ONLY mutates via FillEvents or Reconciliation directives.
    """
    def __init__(self, initial_cash: float):
        self._available_cash = initial_cash
        self._position = 0 # Net quantity
        self._avg_price = 0.0
        self._realized_pnl = 0.0
        
        self._execution_history: List[FillEvent] = []

    def apply_fill(self, fill: FillEvent):
        """Processes a confirmed fill event."""
        self._execution_history.append(fill)
        
        direction = 1 if fill.side == OrderSide.BUY else -1
        notional = fill.fill_quantity * fill.fill_price
        
        if self._position == 0:
            # New position
            self._position = fill.fill_quantity * direction
            self._avg_price = fill.fill_price
        elif (self._position > 0 and direction > 0) or (self._position < 0 and direction < 0):
            # Adding to position
            new_qty = self._position + (fill.fill_quantity * direction)
            self._avg_price = ((abs(self._position) * self._avg_price) + notional) / abs(new_qty)
            self._position = new_qty
        else:
            # Reducing / Closing
            # Calculate realized PnL
            reduced_qty = min(abs(self._position), fill.fill_quantity)
            pnl_per_unit = (fill.fill_price - self._avg_price) * (1 if self._position > 0 else -1)
            self._realized_pnl += reduced_qty * pnl_per_unit
            
            self._position += fill.fill_quantity * direction
            if self._position == 0:
                self._avg_price = 0.0

        # Adjust cash
        self._available_cash -= notional * direction
        self._available_cash -= fill.fee

    def get_state(self, current_price: float) -> PortfolioState:
        unrealized = (current_price - self._avg_price) * self._position if self._position != 0 else 0.0
        return PortfolioState(
            available_cash=self._available_cash,
            locked_shares_t2=0, # Simplified T2 logic
            position_quantity=self._position,
            average_entry_price=self._avg_price,
            unrealized_pnl=unrealized,
            realized_pnl_today=self._realized_pnl
        )

    @property
    def total_fills(self) -> int:
        return len(self._execution_history)

class ReconciliationEngine:
    """
    Verifies that internal ledger matches external broker state (ExecutionLedger).
    Detects mismatch / drift in accounting.
    """
    def __init__(self, ledger: ExposureLedger):
        self.ledger = ledger

    def reconcile(self, broker_position: int, broker_cash: float):
        state = self.ledger.get_state(0.0) # Price irrelevant for qty/cash recon
        if state.position_quantity != broker_position or abs(state.available_cash - broker_cash) > 0.01:
            logging.critical(f"LEDGER MISMATCH: Local({state.position_quantity}, {state.available_cash}) != Broker({broker_position}, {broker_cash})")
            # In institutional systems, this would trigger an atomic halt or force-sync.
            return False
        return True
