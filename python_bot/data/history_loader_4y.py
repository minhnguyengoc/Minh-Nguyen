import os
import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict
import datetime as dt

class HistoryLoader4Y:
    """
    Standardized loader for 4-year historical dataset for VNStock.
    Supports multi-symbol loading, data cleaning, and regime tagging.
    """
    def __init__(self, 
                 symbols: List[str] = ["FPT", "MWG", "HPG", "SSI", "VCB"], 
                 base_dir: str = "historical_data"):
        self.symbols = symbols
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("HistoryLoader4Y")

    def validate_integrity(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Enforces institutional data quality standards."""
        if df.empty:
            return df
            
        # 1. Monotonic Timestamps
        df = df.sort_values('timestamp').drop_duplicates('timestamp')
        
        # 2. OHLC Logic Guards
        df = df[df['high'] >= df['low']]
        df = df[df['high'] >= df['open']]
        df = df[df['high'] >= df['close']]
        df = df[df['low'] <= df['open']]
        df = df[df['low'] <= df['close']]
        df = df[df['volume'] >= 0]
        
        # 3. Handle NaNs
        df = df.ffill().bfill()
        
        return df

    def tag_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds regime metadata for behavioral analysis."""
        # Session State
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        
        def get_session(row):
            h, m = row['hour'], row['minute']
            time_val = h * 100 + m
            if 900 <= time_val <= 915: return "ATO"
            if 1430 <= time_val <= 1445: return "ATC"
            if 1130 <= time_val <= 1300: return "LUNCH"
            return "CONTINUOUS"
            
        df['session'] = df.apply(get_session, axis=1)
        
        # Volatility Regime (20-bar rolling std)
        returns = df['close'].pct_change()
        vol = returns.rolling(20).std()
        df['vol_regime'] = np.where(vol > vol.quantile(0.8), "HIGH", 
                           np.where(vol < vol.quantile(0.2), "LOW", "NORMAL"))
        
        # Trend Regime (SMA 20/50)
        sma20 = df['close'].rolling(20).mean()
        sma50 = df['close'].rolling(50).mean()
        df['trend_regime'] = np.where(df['close'] > sma20, "BULL", "BEAR")
        
        # Liquidity Bucket
        vol_ma = df['volume'].rolling(50).mean()
        df['liq_bucket'] = np.where(df['volume'] > vol_ma * 1.5, "CLIMAX", 
                           np.where(df['volume'] < vol_ma * 0.5, "THIN", "NORMAL"))
        
        return df.fillna(method='bfill')

    def load_combined(self) -> pd.DataFrame:
        """Loads and merges all symbols into a long-format DataFrame."""
        all_dfs = []
        for sym in self.symbols:
            path = os.path.join(self.base_dir, f"{sym}_1m.parquet")
            if not os.path.exists(path):
                self.logger.warning(f"Data missing for {sym} at {path}")
                continue
                
            try:
                df = pd.read_parquet(path)
                df['symbol'] = sym
                df = self.validate_integrity(df, sym)
                df = self.tag_regimes(df)
                all_dfs.append(df)
                self.logger.info(f"Loaded {len(df)} bars for {sym}")
            except Exception as e:
                self.logger.error(f"Failed to load {sym}: {e}")
                
        if not all_dfs:
            raise RuntimeError("No historical data found. Root cause: Missing .parquet files in historical_data/")
            
        return pd.concat(all_dfs, ignore_index=True).sort_values('timestamp')

if __name__ == "__main__":
    # Test stub
    loader = HistoryLoader4Y()
    # Mock data if needed for testing (Optional)
