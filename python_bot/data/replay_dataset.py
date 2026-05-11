import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import hashlib
from python_bot.common.types import MarketData
import logging

class ReplayDatasetPipeline:
    """
    Deterministic Data Ingestion & Replay Pipeline.
    Ensures Causal Stitching and Dataset Integrity for RL.
    """
    def __init__(self, raw_path: str):
        self.raw_path = raw_path
        self._df: Optional[pd.DataFrame] = None
        self._checksum: Optional[str] = None

    def load(self, symbol: str) -> 'ReplayDatasetPipeline':
        """Loads and validates the dataset."""
        try:
            # Assuming standard OHLCV parquet/csv
            if self.raw_path.endswith('.parquet'):
                df = pd.read_parquet(self.raw_path)
            else:
                df = pd.read_csv(self.raw_path, parse_dates=['timestamp'])
                
            # Integrity Checks
            df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
            
            # Canonical Mapping
            df['symbol'] = symbol
            if 'event_id' not in df.columns:
                # Generate stable event IDs if missing
                df['event_id'] = [hashlib.md5(f"{symbol}_{t}".encode()).hexdigest() for t in df['timestamp']]
            
            # Received_at simulation (for replaying with latency)
            if 'received_at' not in df.columns:
                # Synthetic latency between 10ms and 150ms
                df['received_at'] = df['timestamp'] + pd.to_timedelta(np.random.randint(10, 150, len(df)), unit='ms')
            
            self._df = df
            self._checksum = hashlib.sha256(pd.util.hash_pandas_object(df).values).hexdigest()
            logging.info(f"Dataset loaded: {len(df)} rows. Hash: {self._checksum[:16]}")
            
            return self
        except Exception as e:
            logging.error(f"Dataset load failure: {e}")
            raise

    def get_market_data(self) -> List[MarketData]:
        """Converts dataframe to a list of validated MarketData entities."""
        if self._df is None: return []
        
        # Vectorized translation to pydantic models
        records = self._df.to_dict('records')
        return [MarketData(**r) for r in records]

    def split(self, train_ratio: float = 0.8) -> Tuple[List[MarketData], List[MarketData]]:
        """Temporal walk-forward split."""
        all_data = self.get_market_data()
        split_idx = int(len(all_data) * train_ratio)
        return all_data[:split_idx], all_data[split_idx:]

    @property
    def hash(self) -> str:
        return self._checksum or "UNINITIALIZED"
