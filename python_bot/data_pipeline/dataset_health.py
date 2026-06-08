import os
import json
import logging
import pandas as pd
from pathlib import Path
from python_bot.core.paths import DATA_HEALTH_REPORTS_DIR, ROOT_DIR
from python_bot.data_pipeline.schema_validator import SchemaValidator
from python_bot.data_pipeline.data_validator import DataValidator
from python_bot.data_pipeline.feature_validator import FeatureValidator

logger = logging.getLogger("VNStockBot.DatasetHealth")

class DatasetHealthSuite:
    """Consolidates schema, quality, and leakage checks into structured JSON/Markdown reports."""
    
    @classmethod
    def audit_dataset(cls, df: pd.DataFrame, ticker: str) -> dict:
        """
        Runs comprehensive data pipeline health checks.
        Saves audit report file and returns summary metrics dictionary.
        """
        schema_ok, schema_errors = SchemaValidator.validate_schema(df)
        quality_metrics = DataValidator.analyze_quality(df)
        
        # Identify numeric features (excluding OHLVC metadata)
        from python_bot.features.feature_schema import get_numeric_feature_columns
        num_cols = get_numeric_feature_columns(df)
        
        leakage_cols = FeatureValidator.check_leakage_columns(df)
        code_findings = FeatureValidator.audit_codefiles_for_leakage(ROOT_DIR / "python_bot")
        distributions = FeatureValidator.check_distributions(df, num_cols)
        
        # Decide final health grading
        final_status = "PASS"
        if quality_metrics.get("status") == "FAIL" or not schema_ok:
            final_status = "FAIL"
        elif quality_metrics.get("status") == "WARN" or leakage_cols or code_findings:
            final_status = "WARN"
            
        report = {
            "ticker": ticker,
            "timestamp": pd.Timestamp.now().isoformat(),
            "schema_check": {
                "valid": schema_ok,
                "errors": schema_errors
            },
            "quality_metrics": quality_metrics,
            "leakage_analysis": {
                "leaking_columns": leakage_cols,
                "code_leakage_findings": code_findings
            },
            "feature_distributions": distributions,
            "overall_status": final_status
        }
        
        # Save to reports/data_health/
        os.makedirs(DATA_HEALTH_REPORTS_DIR, exist_ok=True)
        report_file = DATA_HEALTH_REPORTS_DIR / f"report_{ticker.lower()}.json"
        try:
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4)
            logger.info(f"✅ Data health report written to: {report_file}")
        except IOError as e:
            logger.error(f"Failed to record data health report file: {e}")
            
        return report
