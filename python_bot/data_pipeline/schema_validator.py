import pandas as pd
import logging
from typing import List, Tuple

logger = logging.getLogger("VNStockBot.SchemaValidator")

class SchemaValidator:
    """Validates structural and typing constraints of the incoming stock dataset."""
    
    REQUIRED_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    @classmethod
    def validate_schema(cls, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validates column names and element types.
        Returns (is_valid, error_messages).
        """
        errors = []
        df_cols_lower = [c.lower() for c in df.columns]
        
        # Check required columns
        for col in cls.REQUIRED_COLUMNS:
            if col not in df_cols_lower:
                errors.append(f"Missing required lower-case column: '{col}'")
                
        if errors:
            return False, errors
            
        # Match types
        # Open, High, Low, Close should be floating numeric types. Volume should be integer/float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            orig_col = [c for c in df.columns if c.lower() == col][0]
            if not pd.api.types.is_numeric_dtype(df[orig_col]):
                errors.append(f"Column '{orig_col}' is not numeric (type: {df[orig_col].dtype})")
                
        return len(errors) == 0, errors
