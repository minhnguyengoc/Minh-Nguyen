import re
from pathlib import Path
import logging
from python_bot.core.exceptions import BotBaseException

logger = logging.getLogger("VNStockBot.SecurityInputValidation")

class SecurityInputValidator:
    """Rigorous defense against input injections, traversal attacks, and corrupted files."""
    
    TICKER_PATTERN = re.compile(r'^[a-zA-Z0-9]{3,6}$')
    
    @classmethod
    def sanitize_ticker(cls, ticker: str) -> str:
        """Enforces alphanumeric character structure between 3 and 6 characters."""
        ticker = ticker.strip().upper()
        if not cls.TICKER_PATTERN.match(ticker):
            raise ValueError(f"SECURITY_VIOLATION: Malformed symbol code input: {ticker}")
        return ticker
        
    @classmethod
    def block_path_traversal(cls, base_path: Path, target_relative: str) -> Path:
        """
        Locks down file-access boundaries.
        Throws a ValueError if relative paths point outside base directories.
        """
        # Resolve reference path
        base_absolute = base_path.resolve()
        
        # Check for path traversals
        if ".." in target_relative or target_relative.startswith("/") or target_relative.startswith("\\"):
            raise ValueError(f"SECURITY_VIOLATION: Suspicious path traversal input: {target_relative}")
            
        resolved_target = (base_path / target_relative).resolve()
        
        if not resolved_target.is_relative_to(base_absolute):
            raise ValueError(f"SECURITY_VIOLATION: Relative destination {target_relative} lies outside folder bounds.")
            
        return resolved_target
        
    @classmethod
    def validate_csv_path(cls, file_path: Path) -> Path:
        """Ensures that the target CSV exists, is readable, and is a file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a valid file: {file_path}")
        if file_path.suffix.lower() != '.csv':
            raise ValueError(f"SECURITY_VIOLATION: Unexpected file format: {file_path}")
        return file_path
