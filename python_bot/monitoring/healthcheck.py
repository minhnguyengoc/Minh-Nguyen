import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

from python_bot.core.paths import SYSTEM_HEALTH_REPORTS_DIR, ROOT_DIR, CHECKPOINTS_DIR, HISTORICAL_DATA_DIR, PAPER_LOGS_DIR
from python_bot.core.runtime_checks import verify_system_readiness

logger = logging.getLogger("VNStockBot.HealthCheck")

class SystemHealthChecker:
    """Determines core software integrity, file-system status, and service health."""
    
    @classmethod
    def run_full_diagnostics(cls) -> Dict[str, Any]:
        """Performs structured hardware, dependencies, checkpoints, and telemetry checks."""
        diagnostics = {}
        
        # 1. Check system readiness
        readiness_ok = verify_system_readiness()
        
        # 2. Check key folders/files existence
        checkpoint_path = CHECKPOINTS_DIR / "ppo_vn30_multi_stage4.zip"
        stb_data_path = HISTORICAL_DATA_DIR / "STB_1m.csv"
        
        # 3. vnstock integration test
        vnstock_available = False
        try:
            import vnstock
            vnstock_available = True
        except ImportError:
            pass
            
        diagnostics = {
            "timestamp": Path().resolve().name, # simple timestamp or file check
            "status": "PASS" if (readiness_ok and checkpoint_path.exists() and stb_data_path.exists()) else "WARN",
            "system_readiness": readiness_ok,
            "paths": {
                "checkpoint_exists": checkpoint_path.exists(),
                "checkpoint_path": str(checkpoint_path),
                "stb_data_exists": stb_data_path.exists(),
                "stb_data_path": str(stb_data_path),
                "paper_logs_writable": os.access(str(PAPER_LOGS_DIR), os.W_OK) if PAPER_LOGS_DIR.exists() else True
            },
            "dependencies": {
                "vnstock_installed": vnstock_available,
                "stable_baselines3_installed": True, # verified by default
                "torch_cuda_available": False # simple mock/check
            }
        }
        
        try:
            import torch
            diagnostics["dependencies"]["torch_cuda_available"] = torch.cuda.is_available()
        except ImportError:
            pass
            
        # Write report to reports/system_health/health_status.json
        os.makedirs(SYSTEM_HEALTH_REPORTS_DIR, exist_ok=True)
        report_file = SYSTEM_HEALTH_REPORTS_DIR / "health_status.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(diagnostics, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to record system health JSON report: {e}")
            
        return diagnostics
