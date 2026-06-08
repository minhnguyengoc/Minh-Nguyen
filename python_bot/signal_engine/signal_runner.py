import logging
from pathlib import Path
import pandas as pd
import json
import os
from python_bot.live.signal_engine import run_signal
from python_bot.core.paths import PAPER_LOGS_DIR

logger = logging.getLogger("VNStockBot.SignalRunner")

class SignalRunner:
    """Orchestrates signal generation cycles in research and automated paper environments."""
    
    @classmethod
    def execute_live_run(cls, ticker: str = "STB") -> dict:
        """Executes current signal poll and guarantees copying of target file names."""
        result = run_signal(ticker=ticker, timeframe="1m")
        
        # Ensure exact output filenames as requested under Phase 5
        # paper_live_signals_STB.csv, paper_live_snapshot_STB.json, paper_live_replay_log_STB.csv
        try:
            base_logs = PAPER_LOGS_DIR
            
            # Map standard output paths
            src_signals = base_logs / f"paper_live_signals_{ticker.upper()}_1m.csv"
            dst_signals = base_logs / f"paper_live_signals_{ticker.upper()}.csv"
            if src_signals.exists():
                pd.read_csv(src_signals).to_csv(dst_signals, index=False)
                
            src_snapshot = base_logs / f"paper_live_snapshot_{ticker.upper()}_1m.json"
            dst_snapshot = base_logs / f"paper_live_snapshot_{ticker.upper()}.json"
            if src_snapshot.exists():
                with open(src_snapshot, 'r', encoding='utf-8') as sf_in:
                    data = json.load(sf_in)
                with open(dst_snapshot, 'w', encoding='utf-8') as sf_out:
                    json.dump(data, sf_out, indent=4)
                    
            src_replay = base_logs / f"paper_live_replay_log_{ticker.upper()}_1m.csv"
            dst_replay = base_logs / f"paper_live_replay_log_{ticker.upper()}.csv"
            if src_replay.exists():
                pd.read_csv(src_replay).to_csv(dst_replay, index=False)
                
            logger.info("Copied Phase 5 exact filenames successfully.")
        except Exception as e:
            logger.error(f"Error transferring exact filename copies: {e}")
            
        return result
