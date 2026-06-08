import logging
from python_bot.paper_portfolio.ledger import PaperLedger

logger = logging.getLogger("VNStockBot.LivePaperLedger")

# Wrap the core PaperLedger for live module references
class LivePaperLedger(PaperLedger):
    """Refined Live Paper Ledger module wrapping core portfolio accounting services."""
    def apply_signal_to_ledger(self, signal_row: dict) -> dict:
        """Applies a verified, executed trading signal to the ledger account."""
        action = signal_row.get("action_name", "HOLD")
        if action == "SELL/CLOSE":
            action = "SELL"
        return self.process_order(
            timestamp=signal_row.get("timestamp"),
            action=action,
            price=signal_row.get("executed_price", 0.0) or signal_row.get("close", 0.0),
            qty=signal_row.get("executed_qty", 0.0),
            transaction_cost=signal_row.get("transaction_cost", 0.0)
        )

def load_live_ledger(ticker: str, initial_cash: float = 100000000.0) -> LivePaperLedger:
    logger.info(f"Loading paper-live portfolio ledger for {ticker}...")
    return LivePaperLedger(ticker=ticker, initial_cash=initial_cash)
