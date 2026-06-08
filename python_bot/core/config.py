import os
import logging
from typing import List

logger = logging.getLogger("VNStockBot.CoreConfig")

# Environment Variables with Robust Validation
BOT_TICKERS_RAW = os.getenv("BOT_TICKERS", "STB")
BOT_TICKERS: List[str] = [t.strip().upper() for t in BOT_TICKERS_RAW.split(",") if t.strip()]

BOT_TIMEFRAME = os.getenv("BOT_TIMEFRAME", "1m").strip()

try:
    PAPER_MAX_STEPS = int(os.getenv("PAPER_MAX_STEPS", "50000"))
except ValueError:
    logger.warning("Invalid PAPER_MAX_STEPS. Using default: 50000")
    PAPER_MAX_STEPS = 50000

try:
    INITIAL_CASH = float(os.getenv("INITIAL_CASH", "100000000.0"))
except ValueError:
    logger.warning("Invalid INITIAL_CASH. Using default: 100000000.0")
    INITIAL_CASH = 100000000.0

try:
    LIVE_POLL_SECONDS = int(os.getenv("LIVE_POLL_SECONDS", "60"))
except ValueError:
    logger.warning("Invalid LIVE_POLL_SECONDS. Using default: 60")
    LIVE_POLL_SECONDS = 60

# Base defaults check
if not BOT_TICKERS:
    BOT_TICKERS = ["STB"]

# Validate configurations
def validate_config():
    if INITIAL_CASH <= 0:
        raise ValueError("INITIAL_CASH must be positive.")
    if LIVE_POLL_SECONDS <= 0:
        raise ValueError("LIVE_POLL_SECONDS must be positive.")
    if PAPER_MAX_STEPS <= 0:
        raise ValueError("PAPER_MAX_STEPS must be positive.")

validate_config()
