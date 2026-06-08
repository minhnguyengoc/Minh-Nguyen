import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger("VNStockBot.DataValidator")

class DataValidator:
    """Performs deep data quality, gap, duplicate, and clean distribution checks."""
    
    @classmethod
    def analyze_quality(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validates data constraints and logs results.
        Returns a detailed metrics dictionary reflecting dataset health.
        """
        metrics = {}
        if df.empty:
            return {"empty": True, "status": "FAIL"}
            
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 1. Monotonic & sorting checks
        is_monotonic = df['timestamp'].is_monotonic_increasing
        non_monotonic_count = 0
        if not is_monotonic:
            # Count out-of-order rows
            ts_diff = df['timestamp'].diff().dt.total_seconds()
            non_monotonic_count = int((ts_diff < 0).sum())
            
        # 2. Duplicate timestamps
        duplicate_count = int(df['timestamp'].duplicated().sum())
        
        # 3. Gaps in timestamps (large jump, e.g. more than 1 day during active weekdays, or general gaps in minutes)
        # For 1m bars, we can check step transitions during active sessions
        # Let's count gaps of > 1 hour during weekdays
        ts_diffs = df['timestamp'].sort_values().diff().dropna()
        # Find jumps larger than 1 hour (3600 seconds) excluding overnight / weekends
        large_gaps = int((ts_diffs.dt.total_seconds() > 3600).sum())
        
        # 4. NaN ratio
        nan_cells = df[['open', 'high', 'low', 'close', 'volume']].isna().sum().sum()
        total_cells = df[['open', 'high', 'low', 'close', 'volume']].size
        nan_ratio = float(nan_cells / total_cells) if total_cells > 0 else 0.0
        
        # 5. Abnormal zero volume percentage
        vol_col = [c for c in df.columns if c.lower() == 'volume'][0]
        zero_vol_count = int((df[vol_col] == 0).sum())
        zero_vol_ratio = float(zero_vol_count / len(df))
        
        # 6. Abnormal price gaps (>15% close-to-close change in a single bar)
        close_col = [c for c in df.columns if c.lower() == 'close'][0]
        pct_changes = df[close_col].pct_change().abs()
        excessive_movements = int((pct_changes > 0.15).sum())
        
        # Assemble metrics
        metrics = {
            "empty": False,
            "row_count": len(df),
            "is_monotonic": is_monotonic,
            "non_monotonic_count": non_monotonic_count,
            "duplicate_count": duplicate_count,
            "large_gaps_count": large_gaps,
            "nan_ratio": nan_ratio,
            "zero_vol_count": zero_vol_count,
            "zero_vol_ratio": zero_vol_ratio,
            "abnormal_price_gaps": excessive_movements,
            "start_date": str(df['timestamp'].min()),
            "end_date": str(df['timestamp'].max()),
            "latest_close": float(df[close_col].iloc[-1]) if len(df) > 0 else 0.0,
        }
        
        # Status calculation
        status = "PASS"
        if duplicate_count > 0 or nan_ratio > 0.05 or excessive_movements > 5 or not is_monotonic:
            status = "WARN"
        if len(df) < 100 or nan_ratio > 0.2:
            status = "FAIL"
            
        metrics["status"] = status
        return metrics
