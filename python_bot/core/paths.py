import os
from pathlib import Path

# Absolute path of the project root
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Centralized directories
HISTORICAL_DATA_DIR = ROOT_DIR / "historical_data"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
LOGS_DIR = ROOT_DIR / "logs"
PAPER_LOGS_DIR = ROOT_DIR / "paper_live_logs"
REPORTS_DIR = ROOT_DIR / "reports"
DATA_HEALTH_REPORTS_DIR = REPORTS_DIR / "data_health"
SYSTEM_HEALTH_REPORTS_DIR = REPORTS_DIR / "system_health"

def ensure_directories():
    """Ensure all required directories exist on the filesystem."""
    for directory in [
        HISTORICAL_DATA_DIR,
        CHECKPOINTS_DIR,
        LOGS_DIR,
        PAPER_LOGS_DIR,
        REPORTS_DIR,
        DATA_HEALTH_REPORTS_DIR,
        SYSTEM_HEALTH_REPORTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

# Run setup
ensure_directories()
