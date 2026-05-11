import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import hashlib
import logging

class DatasetIntegrityAuditor:
    """
    Institutional Data Integrity Guard.
    Detects anomalies, discontinuities, and corruption in historical exchange data.
    """
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.logger = logging.getLogger(f"Integrity_{ticker}")

    def audit(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Performs a comprehensive quality audit of the dataset."""
        report = {}
        
        # 1. Monotonicity & Continuity
        df = df.sort_values('timestamp')
        ts_diff = df['timestamp'].diff()
        
        # Detect gaps > expected interval (e.g. 1 min)
        # Expected interval calculated from median of diffs
        median_interval = ts_diff.median()
        gaps = ts_diff > (median_interval * 2)
        # Exclude lunch breaks (11:30 - 13:00) and overnight
        
        # 2. Price Invariants
        # High >= Open, High >= Close, High >= Low etc.
        inv_violation = (df['high'] < df['low']) | (df['high'] < df['open']) | (df['high'] < df['close']) | \
                         (df['low'] > df['open']) | (df['low'] > df['close'])
        
        # 3. Volume & Liquidity 
        zero_vol = df['volume'] <= 0
        
        # 4. Limit Locks
        # VN Market: +/- 7% for HOSE, 10% HNX, 15% Upcom
        # Identification of price ceilings/floors
        df['pct_change'] = df['close'].pct_change()
        limit_locks = (df['pct_change'].abs() > 0.069) # Proxy for HOSE
        
        # 5. Deterministic Hash
        content_hash = hashlib.sha256(pd.util.hash_pandas_object(df).values).hexdigest()

        report = {
            "rows": len(df),
            "hash": content_hash,
            "gap_count": int(gaps.sum()),
            "invariant_violations": int(inv_violation.sum()),
            "zero_volume_bars": int(zero_vol.sum()),
            "limit_lock_count": int(limit_locks.sum()),
            "continuity_score": 1.0 - (gaps.sum() / len(df)),
            "is_training_safe": (inv_violation.sum() == 0) and (gaps.sum() / len(df) < 0.01)
        }
        
        if not report['is_training_safe']:
            self.logger.error(f"DATASET CORRUPTED: {report}")
            
        return report

    def segment_walkforward(self, df: pd.DataFrame, windows: int = 5) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Splits data into overlapping walk-forward folds."""
        n = len(df)
        fold_size = n // windows
        folds = []
        for i in range(windows - 1):
            train_end = (i + 1) * fold_size
            test_end = (i + 2) * fold_size
            folds.append((df.iloc[:train_end], df.iloc[train_end:test_end]))
        return folds
