class BotBaseException(Exception):
    """Base exception for all trading bot components."""
    pass

class DataIntegrityError(BotBaseException):
    """Raised when data validation fails or features have future leakage."""
    pass

class ModelIncompatibilityError(BotBaseException):
    """Raised when loaded checkpoint is incompatible with env space/features."""
    pass

class RiskValidationError(BotBaseException):
    """Raised when a proposed trade violates pre-trade risk thresholds."""
    pass

class LedgerAccountingError(BotBaseException):
    """Raised on illegal portfolio actions (e.g. negative cash, position mismatch)."""
    pass

class LiveFeedException(BotBaseException):
    """Raised when quote APIs fail, timeout, or return corrupted structures."""
    pass
