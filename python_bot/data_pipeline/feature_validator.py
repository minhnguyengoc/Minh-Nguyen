import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger("VNStockBot.FeatureValidator")

class FeatureValidator:
    """Detects predictive future leakage, backward shifts, and distribution drift in features."""
    
    LEAKAGE_KEYWORDS = ['future', 'forward', 'target', 'label', 'next']
    
    @classmethod
    def check_leakage_columns(cls, df: pd.DataFrame) -> List[str]:
        """Identifies columns that have names indicating classification labels or future leakage."""
        leaking = []
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in cls.LEAKAGE_KEYWORDS):
                leaking.append(col)
        return leaking

    @classmethod
    def audit_codefiles_for_leakage(cls, bot_dir: Path) -> List[Dict[str, Any]]:
        """
        Scans code files inside the repository for suspicious lookahead references.
        Specifically looking for negative shift operations like `shift(-X)`.
        """
        findings = []
        # Walk python repository files
        for path in bot_dir.glob("**/*.py"):
            try:
                content = path.read_text(encoding='utf-8')
                # Find occurrences of .shift(-
                matches = re.findall(r'\.shift\s*\(\s*-\s*\d+', content)
                if matches:
                    findings.append({
                        "file": str(path.relative_to(bot_dir.parent)),
                        "matches": matches,
                        "description": "Suspicious lookahead shift operation `.shift(-x)` detected"
                    })
            except Exception as e:
                pass
        return findings

    @classmethod
    def check_distributions(cls, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        """Analyzes mean, std, min, max, and infinite counts for numeric feature columns."""
        report = {}
        for col in feature_cols:
            if col in df.columns:
                series = df[col]
                finite_vals = series[np.isfinite(series)]
                inf_count = int(np.isinf(series).sum())
                nan_count = int(series.isna().sum())
                
                report[col] = {
                    "mean": float(finite_vals.mean()) if len(finite_vals) > 0 else 0.0,
                    "std": float(finite_vals.std()) if len(finite_vals) > 1 else 0.0,
                    "min": float(finite_vals.min()) if len(finite_vals) > 0 else 0.0,
                    "max": float(finite_vals.max()) if len(finite_vals) > 0 else 0.0,
                    "inf_count": inf_count,
                    "nan_count": nan_count
                }
        return report
