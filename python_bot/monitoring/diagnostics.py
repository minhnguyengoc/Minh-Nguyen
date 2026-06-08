import logging
from typing import List, Dict, Any
from python_bot.core.paths import LOGS_DIR

logger = logging.getLogger("VNStockBot.Diagnostics")

class DiagnosticsEngine:
    """Analyzes system logging logs to detect runtime faults and thread starvation logs."""
    
    @classmethod
    def get_recent_errors(cls, max_lines: int = 50) -> List[str]:
        """Scans the system.log for error logs and warnings to surface on consoles."""
        log_file = LOGS_DIR / "system.log"
        errors = []
        if log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in reversed(lines):
                    if "ERROR" in line or "CRITICAL" in line or "WARNING" in line:
                        errors.append(line.strip())
                        if len(errors) >= max_lines:
                            break
            except Exception as e:
                logger.error(f"Failed to parse diagnostics line: {e}")
        return errors
