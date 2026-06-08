import unittest
from datetime import datetime
from python_bot.risk.risk_limits import RiskLimits
from python_bot.risk.trade_guard import TradeGuard
from python_bot.risk.portfolio_guard import PortfolioGuard

class TestRiskControls(unittest.TestCase):
    
    def setUp(self):
        self.limits = RiskLimits(
            max_position_value_pct=0.2,
            max_daily_loss_pct=0.02,
            max_drawdown_pct=0.05,
            min_lot_size=100,
            allow_short=False
        )
        self.trade_guard = TradeGuard(self.limits)
        self.port_guard = PortfolioGuard(self.limits)
        
    def test_multi_symbol_order_exclusion(self):
        # Only STB is allowed in paper-live
        allowed, reason = self.trade_guard.check_trade("FPT", "BUY", 100, 50000.0, 100000000.0, 0)
        self.assertFalse(allowed)
        self.assertEqual(reason, "FORBIDDEN_MULTI_SYMBOL_ORDER")
        
    def test_stb_allowed(self):
        allowed, reason = self.trade_guard.check_trade("STB", "BUY", 100, 20000.0, 100000000.0, 0)
        self.assertTrue(allowed)
        self.assertEqual(reason, "NONE")
        
    def test_lot_size_enforcements(self):
        # 150 is not multiple of 100 lot size
        allowed, reason = self.trade_guard.check_trade("STB", "BUY", 150, 20000.0, 100000000.0, 0)
        self.assertFalse(allowed)
        self.assertIn("LOT_SIZE_LOTTERY_VIOLATION", reason)
        
    def test_drawdown_limits(self):
        # Equity is 90k, peak was 100k -> 10% drawdown (limit is 5%)
        allowed, reason = self.port_guard.check_portfolio_limits(
            current_equity=90000.0,
            peak_equity=100000.0,
            daily_pnl=0.0,
            proposed_trade_val=5000.0,
            last_bar_time=datetime.now()
        )
        self.assertFalse(allowed)
        self.assertIn("MAX_DRAWDOWN_BREACH", reason)
        
    def test_concentration_violation(self):
        # Proposed trade value is 30,000, equity is 100,000 -> 30% concentration (limit is 20%)
        allowed, reason = self.port_guard.check_portfolio_limits(
            current_equity=100000.0,
            peak_equity=100000.0,
            daily_pnl=0.0,
            proposed_trade_val=30000.0,
            last_bar_time=datetime.now()
        )
        self.assertFalse(allowed)
        self.assertIn("MAX_POSITION_CONCENTRATION_BREACH", reason)
        
if __name__ == "__main__":
    unittest.main()
