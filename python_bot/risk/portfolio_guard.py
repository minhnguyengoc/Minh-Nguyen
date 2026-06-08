import logging
from typing import Tuple, Dict, Any
from datetime import datetime, time, timezone
from python_bot.risk.risk_limits import RiskLimits

logger = logging.getLogger("VNStockBot.PortfolioGuard")

class PortfolioGuard:
    """Manages aggregate portfolio concentration, drawdown levels, and session boundaries."""
    
    def __init__(self, limits: RiskLimits = None):
        self.limits = limits or RiskLimits()
        
    def check_portfolio_limits(self, 
                               current_equity: float, 
                               peak_equity: float, 
                               daily_pnl: float, 
                               proposed_trade_val: float,
                               last_bar_time: datetime,
                               is_live_deployment: bool = False) -> Tuple[bool, str]:
        """
        Validates risk constraints on portfolio equity metrics.
        Returns (is_allowed, rejected_reason).
        """
        # 1. Check drawdown
        if peak_equity > 0:
            drawdown = (peak_equity - current_equity) / peak_equity
            if drawdown > self.limits.max_drawdown_pct:
                return False, f"MAX_DRAWDOWN_BREACH: current drawdown {drawdown*100:.2f}% exceeds limit {self.limits.max_drawdown_pct*100:.2f}%"
                
        # 2. Check daily loss
        if daily_pnl < 0:
            loss_pct = abs(daily_pnl) / max(current_equity, 1e-9)
            if loss_pct > self.limits.max_daily_loss_pct:
                return False, f"MAX_DAILY_LOSS_BREACH: daily loss {loss_pct*100:.2f}% exceeds limit {self.limits.max_daily_loss_pct*100:.2f}%"
                
        # 3. Position value concentration check
        if current_equity > 0:
            concentration = proposed_trade_val / current_equity
            if concentration > self.limits.max_position_value_pct:
                return False, f"MAX_POSITION_CONCENTRATION_BREACH: order value ratio {concentration*100:.2f}% exceeds limit {self.limits.max_position_value_pct*100:.2f}%"
                
        # 4. Market session & stale data (only enforced in real live production deployments)
        if is_live_deployment:
            # Check market closed (Vietnam market hours: 09:00-11:30 and 13:00-15:00)
            vn_now = datetime.now() # and localize/derive
            # Basic hour check
            current_hour_time = vn_now.time()
            morn_start = time(9, 0)
            morn_end = time(11, 30)
            aft_start = time(13, 0)
            aft_end = time(15, 0)
            
            is_morn = (morn_start <= current_hour_time <= morn_end)
            is_aft = (aft_start <= current_hour_time <= aft_end)
            is_weekday = (vn_now.weekday() < 5)
            
            if not ((is_morn or is_aft) and is_weekday):
                return False, "MARKET_CLOSED_PROTECTION: trading is disabled outside standard HOSE sessions."
                
            # Stale data check
            age = (vn_now - last_bar_time).total_seconds()
            if age > self.limits.stale_data_limit_seconds:
                return False, f"STALE_DATA_PROTECTION: latest data is {age:.0f}s old, limit is {self.limits.stale_data_limit_seconds}s."
                
        return True, "NONE"
