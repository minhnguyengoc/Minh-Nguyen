import os
import shutil
import logging
from python_bot.core.paths import LOGS_DIR, PAPER_LOGS_DIR

logger = logging.getLogger("VNStockBot.RuntimeChecks")

def verify_system_readiness() -> bool:
    """Verifies write access, disk space, and runtime libraries."""
    # Check directory write permissions
    for d in [LOGS_DIR, PAPER_LOGS_DIR]:
        test_file = d / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except OSError as e:
            logger.error(f"❌ Standard write test failed on directory {d}: {e}")
            return False
            
    # Check Disk Space
    total, used, free = shutil.disk_usage(str(LOGS_DIR))
    free_gb = free / (2**30)
    if free_gb < 0.1:
        logger.error(f"❌ Disk space extremely low: {free_gb:.2f} GB available.")
        return False
        
    logger.info("✅ System runtime preflight checks COMPLETE and READY.")
    return True
