import pandas as pd
from typing import Dict, Any
from python_bot.signal_engine.signal_state import SignalStateManager
from python_bot.signal_engine.signal_store import SignalStore
from python_bot.paper_portfolio.ledger import PaperLedger

class MetricsTracker:
    """Calculates active trading metrics and gathers portfolio status snapshots."""
    
    @classmethod
    def get_live_metrics(cls, ticker: str = "STB") -> Dict[str, Any]:
        """Assembles comprehensive current performance vectors and signal counts."""
        state = SignalStateManager.load_state(ticker)
        signals = SignalStore.load_signals(ticker)
        
        # Load ledger
        ledger = PaperLedger(ticker)
        
        total_signals = len(signals)
        total_trades = len(signals[signals["action"] != 0]) if total_signals > 0 else 0
        
        # Get rejected reasons distribution
        rejected_reasons = {}
        if total_signals > 0 and "rejected_reason" in signals.columns:
            rejected_reasons = signals["rejected_reason"].value_counts().to_dict()
            
        latest_close = 0.0
        latest_timestamp = None
        if total_signals > 0:
            latest_close = float(signals["close"].iloc[-1])
            latest_timestamp = str(signals["timestamp"].iloc[-1])
            
        metrics = {
            "latest_data_timestamp": latest_timestamp or state.get("last_processed_timestamp"),
            "latest_close": latest_close,
            "total_signals": total_signals,
            "total_paper_trades": total_trades,
            "paper_equity": float(ledger.cash + (ledger.position_qty * latest_close)) if latest_close > 0 else ledger.cash,
            "paper_cash": ledger.cash,
            "position_shares": ledger.position_qty,
            "cumulative_pnl": state.get("cumulative_pnl", 0.0),
            "rejected_reasons": rejected_reasons,
            "last_error": None
        }
        return metrics
