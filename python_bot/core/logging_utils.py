import logging
import sys
from python_bot.core.paths import LOGS_DIR

def setup_logger(name: str, log_file: str = "app.log", level=logging.INFO):
    """Sets up a standardized logger with both console and file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if setup was called multiple times
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_path = LOGS_DIR / log_file
    file_handler = logging.FileHandler(str(file_path), encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Root level bot logger configuration
system_logger = setup_logger("VNStockBot", "system.log")
