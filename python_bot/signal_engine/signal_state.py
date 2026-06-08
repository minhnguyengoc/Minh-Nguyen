import os
import json
import logging
from pathlib import Path
from python_bot.core.paths import PAPER_LOGS_DIR

logger = logging.getLogger("VNStockBot.SignalState")

class SignalStateManager:
    """Manages persistent paper trading states to ensure idempotency and sequence tracking."""
    
    @classmethod
    def load_state(cls, ticker: str) -> dict:
        """Loads state dictionary for the given ticker."""
        state_file = PAPER_LOGS_DIR / f"paper_live_state_{ticker.lower()}_1m.json"
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load signal state file: {e}")
                
        # Return clean default state
        return {
            "ticker": ticker,
            "last_processed_timestamp": None,
            "current_position": 0.0,
            "cash": 100000000.0,
            "unsettled_t1": 0.0,
            "unsettled_t2": 0.0,
            "cumulative_pnl": 0.0
        }

    @classmethod
    def save_state(cls, ticker: str, state: dict):
        """Saves state dictionary for the given ticker."""
        os.makedirs(PAPER_LOGS_DIR, exist_ok=True)
        state_file = PAPER_LOGS_DIR / f"paper_live_state_{ticker.lower()}_1m.json"
        try:
            with open(state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=4)
            logger.debug(f"Saved state to: {state_file}")
        except Exception as e:
            logger.error(f"Failed to write signal state file: {e}")
