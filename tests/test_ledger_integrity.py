import unittest
import shutil
import tempfile
from pathlib import Path
import pandas as pd

from python_bot.paper_portfolio.ledger import PaperLedger
from python_bot.core.exceptions import LedgerAccountingError

class TestLedgerIntegrity(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        # Override the PAPER_LOGS_DIR inside Path so it writes to temp directory
        import python_bot.paper_portfolio.ledger as l_module
        l_module.PAPER_LOGS_DIR = Path(self.temp_dir)
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        
    def test_initial_state(self):
        ledger = PaperLedger("STB", initial_cash=1000.0)
        self.assertEqual(ledger.cash, 1000.0)
        self.assertEqual(ledger.position_qty, 0.0)
        
    def test_buy_order_success(self):
        ledger = PaperLedger("STB", initial_cash=5000.0)
        entry = ledger.process_order("2026-06-01 09:00:00", "BUY", price=10.0, qty=100, transaction_cost=5.0)
        
        self.assertEqual(entry["rejected_reason"], "NONE")
        self.assertEqual(ledger.cash, 5000.0 - 1005.0)
        self.assertEqual(ledger.position_qty, 100)
        self.assertEqual(ledger.avg_cost, 10.0)
        
    def test_insufficient_cash_leads_to_rejection(self):
        ledger = PaperLedger("STB", initial_cash=500.0)
        entry = ledger.process_order("2026-06-01 09:00:00", "BUY", price=10.0, qty=100)
        
        self.assertEqual(entry["rejected_reason"], "NEGATIVE_CASH_VIOLATION")
        self.assertEqual(ledger.cash, 500.0)
        self.assertEqual(ledger.position_qty, 0.0)
        
    def test_negative_position_violation_rejection(self):
        ledger = PaperLedger("STB", initial_cash=1000.0)
        entry = ledger.process_order("2026-06-01 09:00:00", "SELL", price=15.0, qty=50)
        
        self.assertEqual(entry["rejected_reason"], "NEGATIVE_POSITION_VIOLATION")
        self.assertEqual(ledger.cash, 1000.0)

    def test_idempotent_order_behavior(self):
        ledger = PaperLedger("STB", initial_cash=5000.0)
        # Process BUY
        ledger.process_order("2026-06-01 09:00:00", "BUY", price=10.0, qty=100)
        self.assertEqual(ledger.cash, 4000.0)
        
        # Reprocess with same timestamp should trigger idempotent reject/ignore
        ret = ledger.process_order("2026-06-01 09:00:00", "BUY", price=10.0, qty=100)
        self.assertEqual(ret["rejected_reason"], "IDEMPOTENT_IGNORE")
        self.assertEqual(ledger.cash, 4000.0)
        
if __name__ == "__main__":
    unittest.main()
