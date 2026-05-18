import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

FORBIDDEN_COLS = [
    'timestamp', 'date', 'time', 'datetime',
    'ticker', 'symbol', 'code', 'stock', 'stock_code',
    'ticker_name', 'name', 'company_name'
]

def get_numeric_feature_columns(df: pd.DataFrame, forbidden_cols: Optional[List[str]] = None) -> List[str]:
    """
    Returns a deterministic list of numeric-only columns from the dataframe.
    Excludes forbidden columns.
    """
    if forbidden_cols is None:
        forbidden_cols = FORBIDDEN_COLS
        
    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c]) # Usually want floats/ints
        and c not in forbidden_cols
    ]
    
    # Ensure deterministic order
    return sorted(numeric_cols)

def feature_schema_hash(columns: List[str]) -> str:
    """
    Returns a stable hash of the joined column names.
    Useful for detecting schema drift.
    """
    import hashlib
    schema_str = ",".join(sorted(columns))
    return hashlib.sha256(schema_str.encode()).hexdigest()

def validate_env_observation(env):
    """
    Validates that the environment's observation matches its defined space.
    """
    obs, _ = env.reset()
    expected_shape = env.observation_space.shape[0]
    actual_shape = obs.shape[0]
    
    if actual_shape != expected_shape:
        raise RuntimeError(f"OBSERVATION_DIM_MISMATCH: got {actual_shape}, expected {expected_shape}")
        
    if not np.isfinite(obs).all():
        raise RuntimeError("OBSERVATION_HAS_NON_FINITE_VALUES: Observation contains NaN or Inf.")
        
    if obs.dtype != np.float32:
        logger.warning(f"Observation dtype is {obs.dtype}, expected float32. Converting...")
        obs = obs.astype(np.float32)
    
    return True
