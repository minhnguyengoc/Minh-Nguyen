from dataclasses import dataclass

@dataclass
class RiskLimits:
    """Enterprise risk control limit thresholds."""
    max_position_value_pct: float = 0.2
    max_daily_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.05
    min_lot_size: int = 100
    allow_short: bool = False
    cooldown_seconds: int = 60
    stale_data_limit_seconds: int = 300
