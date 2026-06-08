import os
import pandas as pd
import logging
from pathlib import Path
from python_bot.core.paths import PAPER_LOGS_DIR

logger = logging.getLogger("VNStockBot.SignalStore")

class SignalStore:
    """Stores generated signals persistently inside a CSV file for analytical logging."""
    
    COLUMNS = [
        "timestamp", "ticker", "close", "action", "action_name",
        "equity", "position", "executed_qty", "executed_price",
        "transaction_cost", "rejected_reason"
    ]
    
    @classmethod
    def append_signal(cls, ticker: str, signal_data: dict):
        """Appends a new signal record to the CSV file representation."""
        os.makedirs(PAPER_LOGS_DIR, exist_ok=True)
        file_path = PAPER_LOGS_DIR / f"paper_live_signals_{ticker.upper()}_1m.csv"
        
        # Ensure all columns present
        row_dict = {col: signal_data.get(col, None) for col in cls.COLUMNS}
        # Add timestamp in human format
        if isinstance(row_dict["timestamp"], pd.Timestamp):
            row_dict["timestamp"] = str(row_dict["timestamp"])
            
        new_df = pd.DataFrame([row_dict])
        
        if file_path.exists():
            try:
                new_df.to_csv(file_path, mode='a', header=False, index=False)
            except Exception as e:
                logger.error(f"Failed to append signal data structure: {e}")
        else:
            try:
                new_df.to_csv(file_path, index=False)
            except Exception as e:
                logger.error(f"Failed to create and write signal data structure: {e}")

    @classmethod
    def load_signals(cls, ticker: str) -> pd.DataFrame:
        """Loads all recorded raw signals for verification."""
        file_path = PAPER_LOGS_DIR / f"paper_live_signals_{ticker.upper()}_1m.csv"
        if file_path.exists():
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                logger.error(f"Failed to read signal store file: {e}")
        return pd.DataFrame(columns=cls.COLUMNS)
